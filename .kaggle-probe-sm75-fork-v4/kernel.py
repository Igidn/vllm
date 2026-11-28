#!/usr/bin/env python
"""Stage 4 probe: 100k context on 2xT4 without DSpark.

The Stage 0-2 probe capped the cache at 1536 blocks because the extensible
sizing withheld a 1.44 GiB margin and 2.16 GiB of torch reservation growth.
Smaller batches shrink both, and the hybrid allocator only reserves the
sliding window for local/conv groups, so 100k should fit. Spec decode stays
off: explicit fp16 KV + speculative_config is rejected.

Pushed by: kaggle kernels push -p .kaggle-probe-sm75-fork-v4
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

VENV = Path("/kaggle/tmp/vllm-sm75-venv")
PY = VENV / "bin" / "python"
SRC = Path("/kaggle/tmp/vllm-sm75-src")
MODEL = Path("/kaggle/tmp/inkling")
PORT = 8016
LOG = Path("/kaggle/working/inkling-vllm-sm75.log")
DIAG = Path("/kaggle/tmp/sm75-diag.json")
RUN_LOG = Path("/kaggle/tmp/sm75-run.log")
RESULT_PATH = Path("/kaggle/working/probe-sm75-fork-v4-result.json")

FORK_URL = os.environ.get("KAGGLE_FORK_URL", "https://github.com/Igidn/vllm.git")
FORK_REF = os.environ.get(
    "KAGGLE_FORK_REF", "4a66ebfc50f5151a7b4a79a6d142bd897110e992"
)
PRECOMPILED_COMMIT = os.environ.get(
    "KAGGLE_PRECOMPILED_COMMIT", "8c97cff059fc0d142cbc4acef9fbba62c0e08aed"
)
PRECOMPILED_VARIANT = os.environ.get("KAGGLE_PRECOMPILED_VARIANT", "cu132")

RESULT: dict = {
    "probe": "sm75-fork-stage4-context",
    "fork_url": FORK_URL,
    "fork_ref": FORK_REF,
    "precompiled_commit": PRECOMPILED_COMMIT,
    "precompiled_variant": PRECOMPILED_VARIANT,
    "venv": str(VENV),
}

DIAG_BODY = r'''
import json
import torch
import vllm
from vllm.platforms import current_platform
from vllm.models.inkling2.quant_config import Inkling2QuantConfig

r = {
    "vllm_version": vllm.__version__,
    "capability": list(torch.cuda.get_device_capability()),
    "min_capability": Inkling2QuantConfig.get_min_capability(),
    "act_dtypes": [str(d) for d in Inkling2QuantConfig.get_supported_act_dtypes()],
    "gpu_free_bytes": torch.cuda.mem_get_info()[0],
}
with open("/kaggle/tmp/sm75-diag.json", "w") as f:
    json.dump(r, f, indent=2, default=str)
print(json.dumps(r, indent=2, default=str))
'''


def run(argv, timeout=1200, env=None):
    cmd = " ".join(str(a) for a in argv)
    print("+", cmd, flush=True)
    p = subprocess.run(
        [str(a) for a in argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env or dict(os.environ),
    )
    text = ((p.stdout or "") + (p.stderr or "")).strip()
    with RUN_LOG.open("a") as handle:
        handle.write(f"+ {cmd}\n{text}\n")
    print(text[-4000:], flush=True)
    return p.returncode, text


def finish():
    RESULT_PATH.write_text(json.dumps(RESULT, indent=2, default=str))
    print("=== RESULT ===", flush=True)
    print(json.dumps(RESULT, indent=2, default=str)[-14000:], flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def clone_fork():
    if not (SRC / ".git").is_dir():
        run(["git", "init", str(SRC)], timeout=300)
        run(["git", "-C", str(SRC), "remote", "add", "origin", FORK_URL], timeout=300)
    run(
        ["git", "-C", str(SRC), "fetch", "--depth", "1", "origin", FORK_REF],
        timeout=1800,
    )
    run(["git", "-C", str(SRC), "checkout", "-f", "FETCH_HEAD"], timeout=300)
    code, out = run(["git", "-C", str(SRC), "rev-parse", "HEAD"], timeout=300)
    RESULT["checked_out"] = out.strip()
    if RESULT["checked_out"] != FORK_REF:
        raise RuntimeError(f"expected {FORK_REF}, checked out {RESULT['checked_out']}")


def install_fork(uv):
    env = {
        **os.environ,
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_USE_PRECOMPILED": "1",
        "VLLM_PRECOMPILED_WHEEL_COMMIT": PRECOMPILED_COMMIT,
        "VLLM_PRECOMPILED_WHEEL_VARIANT": PRECOMPILED_VARIANT,
    }
    code, out = run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(PY),
            "--no-cache",
            "--torch-backend=auto",
            "-e",
            str(SRC),
        ],
        timeout=5400,
        env=env,
    )
    RESULT["install_code"] = code
    RESULT["install_tail"] = out[-8000:]
    if code != 0:
        raise RuntimeError(f"fork install failed with code {code}; see install_tail")


def serve_and_chat(env):
    from huggingface_hub import snapshot_download

    MODEL.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="thinkingmachines/Inkling-2-Mini", local_dir=str(MODEL))
    RESULT["model_config"] = json.loads((MODEL / "config.json").read_text())
    RESULT["quant_config"] = json.loads((MODEL / "hf_quant_config.json").read_text())

    # 2000-token prompt: long enough that prompt throughput and KV growth are
    # visible, short enough to prefill in 512-token chunks within the kernel
    # time budget.
    prompt = (
        "You are testing a long-context configuration. Read the numbered "
        "facts below, then answer the question at the end.\n"
    )
    for i in range(120):
        prompt += (
            f"{i}. Fact {i}: the value for record {i} is {i * 7 % 97} and its "
            f"category is {'alpha' if i % 3 == 0 else 'beta' if i % 3 == 1 else 'gamma'}.\n"
        )
    prompt += (
        "Question: what is the value for record 42, and what category is "
        "record 43 in? Answer with just the number and the category."
    )

    argv = [
        str(VENV / "bin" / "vllm"),
        "serve",
        "--model",
        str(MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--tensor-parallel-size",
        "2",
        "--enable-expert-parallel",
        "--dtype",
        "half",
        "--max-model-len",
        "102400",
        # Small in-flight budget: the hybrid allocator reserves
        # min(window + in_flight, max_model_len) tokens for local/conv groups.
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.98",
        "--limit-mm-per-prompt",
        '{"image": 0}',
        "--enforce-eager",
        "--reasoning-parser",
        "inkling",
        "--tool-call-parser",
        "inkling",
        "--enable-auto-tool-choice",
        "--served-model-name",
        "inkling-sm75",
    ]
    RESULT["command"] = argv
    RESULT["prompt_chars"] = len(prompt)
    started = time.time()
    with LOG.open("wb") as handle:
        proc = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, env=env)
        try:
            ready = False
            while time.time() - started < 2400:
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{PORT}/health", timeout=5
                    ) as response:
                        if response.status == 200:
                            ready = True
                            break
                except Exception:
                    time.sleep(10)
            RESULT["ready"] = ready
            RESULT["boot_seconds"] = round(time.time() - started, 1)
            RESULT["exit_code"] = proc.poll()
            if ready:
                payload = json.dumps(
                    {
                        "model": "inkling-sm75",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 64,
                        "temperature": 0.0,
                    }
                ).encode()
                request = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(request, timeout=900) as response:
                        body = json.loads(response.read().decode())
                        RESULT["chat"] = body["choices"][0]["message"]
                        RESULT["chat_usage"] = body.get("usage")
                except Exception as exc:
                    RESULT["chat_error"] = repr(exc)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
    RESULT["log_path"] = str(LOG)
    log = LOG.read_text(errors="replace")
    RESULT["log_tail"] = log[-40000:]

    interesting = (
        "kv cache group sizes",
        "GPU KV cache size",
        "Maximum concurrency",
        "Available KV cache memory",
        "Memory after warmup",
        "Remapping the KV cache",
        "Extended KV cache",
        "caps this instance",
        "does not fit",
        "estimated maximum model length",
        "Avg prompt throughput",
    )
    RESULT["log_markers"] = [
        line[:300] for line in log.splitlines() if any(m in line for m in interesting)
    ]
    forbidden = (
        "ptxas",
        "cvt.bf16.f32",
        "griddepcontrol",
        "illegal instruction",
        "illegal memory access",
        "no kernel image",
        "CUDA error",
        "Traceback (most recent call last)",
        "ValueError",
        "RuntimeError",
    )
    RESULT["forbidden_log_lines"] = [
        line[:300] for line in log.splitlines() if any(m in line for m in forbidden)
    ]

    usage = RESULT.get("chat_usage") or {}
    content = (RESULT.get("chat") or {}).get("content") or ""
    RESULT["gate"] = {
        "ready": bool(RESULT.get("ready")),
        "chat_nonempty": bool(content.strip()),
        "prompt_tokens_over_1000": (usage.get("prompt_tokens") or 0) > 1000,
        "no_forbidden_log_lines": not RESULT["forbidden_log_lines"],
    }
    RESULT["gate"]["passed"] = all(
        value for key, value in RESULT["gate"].items() if key != "passed"
    )


def main():
    try:
        uv = shutil.which("uv")
        if uv is None:
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--no-cache-dir",
                    "uv",
                ]
            )
            uv = shutil.which("uv") or str(Path(sys.executable).with_name("uv"))
        run([uv, "venv", "--python", "3.12", str(VENV)], timeout=600)

        clone_fork()
        install_fork(uv)

        env = {**os.environ, "VLLM_NO_USAGE_STATS": "1"}
        diag_script = Path("/kaggle/tmp/sm75-diag-body.py")
        diag_script.write_text(DIAG_BODY)
        code, out = run([str(PY), str(diag_script)], timeout=600, env=env)
        RESULT["diag_code"] = code
        if DIAG.is_file():
            RESULT["diag"] = json.loads(DIAG.read_text())
        else:
            RESULT["diag_error"] = out[-2500:]

        serve_and_chat(env)
    except BaseException as exc:
        RESULT["fatal"] = repr(exc)
        RESULT["fatal_traceback"] = traceback.format_exc()[-5000:]
        raise
    finally:
        finish()


if __name__ == "__main__":
    main()