#!/usr/bin/env python
"""Performance probe: CUDA graphs + batch scaling on 2xT4.

Boots without --enforce-eager (breakable CUDA graphs are the default for
Inkling2Loop) and measures decode tok/s at concurrency 1, 4 and 8 with DSpark
on, then once with DSpark off. The point is to see whether graph capture
succeeds on sm75 and how much batching amortises the expert-weight reads.

Pushed by: kaggle kernels push -p .kaggle-probe-sm75-fork-v5
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
RUN_LOG = Path("/kaggle/tmp/sm75-run.log")
RESULT_PATH = Path("/kaggle/working/probe-sm75-fork-v5-result.json")

FORK_URL = os.environ.get("KAGGLE_FORK_URL", "https://github.com/Igidn/vllm.git")
FORK_REF = os.environ.get(
    "KAGGLE_FORK_REF", "ae2a04ecc258870722ca92c5ed6201167a8cd7c8"
)
PRECOMPILED_COMMIT = os.environ.get(
    "KAGGLE_PRECOMPILED_COMMIT", "8c97cff059fc0d142cbc4acef9fbba62c0e08aed"
)
PRECOMPILED_VARIANT = os.environ.get("KAGGLE_PRECOMPILED_VARIANT", "cu132")

SPEC_CONFIG = json.dumps(
    {
        "method": "dspark",
        "num_speculative_tokens": 7,
        "draft_sample_method": "probabilistic",
    },
    separators=(",", ":"),
)

PROMPT = (
    "Write a detailed technical explanation of how a paged KV cache works in "
    "an LLM inference server, covering block allocation, prefix caching and "
    "eviction. Aim for about 150 words."
)

RESULT: dict = {
    "probe": "sm75-fork-perf",
    "fork_url": FORK_URL,
    "fork_ref": FORK_REF,
    "precompiled_commit": PRECOMPILED_COMMIT,
    "precompiled_variant": PRECOMPILED_VARIANT,
    "venv": str(VENV),
}


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


def chat_batch(n: int, max_tokens: int = 128, concurrent: int = 1) -> dict:
    payload = json.dumps(
        {
            "model": "inkling-sm75",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    started = time.time()
    total = 0
    chars = 0
    spec = None
    if concurrent <= 1:
        request = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1200) as response:
            body = json.loads(response.read().decode())
        total = body["usage"]["completion_tokens"]
        chars = len(body["choices"][0]["message"].get("content") or "")
        spec = (body.get("metrics") or {}).get("speculative_decoding")
    else:
        import concurrent.futures as cf

        def one(_):
            request = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=1200) as response:
                return json.loads(response.read().decode())

        with cf.ThreadPoolExecutor(max_workers=concurrent) as pool:
            bodies = list(pool.map(one, range(concurrent)))
        total = sum(b["usage"]["completion_tokens"] for b in bodies)
        chars = sum(len(b["choices"][0]["message"].get("content") or "") for b in bodies)
        spec = (bodies[0].get("metrics") or {}).get("speculative_decoding")
    elapsed = time.time() - started
    return {
        "n": n,
        "concurrent": concurrent,
        "elapsed_s": round(elapsed, 2),
        "completion_tokens": total,
        "tok_s": round(total / elapsed, 2) if elapsed else None,
        "spec": spec,
        "text_chars": chars,
    }


def serve_once(env, spec: bool, batches: list[int]) -> dict:
    from huggingface_hub import snapshot_download

    MODEL.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="thinkingmachines/Inkling-2-Mini", local_dir=str(MODEL))

    argv = [
        str(VENV / "bin" / "vllm"),
        "serve",
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
        "1024",
        "--max-num-batched-tokens",
        "512",
        "--gpu-memory-utilization",
        "0.98",
        "--max-num-seqs",
        "8",
        "--limit-mm-per-prompt",
        '{"image": 0}',
        # No --enforce-eager: breakable CUDA graphs capture the text decode.
        "--cudagraph-capture-sizes",
        "1",
        "2",
        "4",
        "8",
        "--hf-overrides",
        '{"speculators_config": null}',
        "--reasoning-parser",
        "inkling",
        "--tool-call-parser",
        "inkling",
        "--enable-auto-tool-choice",
        "--served-model-name",
        "inkling-sm75",
    ]
    if spec:
        argv += [
            "--per-request-spec-decode-metrics",
            "summary",
            "--speculative-config",
            SPEC_CONFIG,
        ]
    started = time.time()
    with LOG.open("ab" if spec else "wb") as handle:
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
            out = {
                "spec": spec,
                "command": argv,
                "ready": ready,
                "boot_seconds": round(time.time() - started, 1),
                "exit_code": proc.poll(),
            }
            if ready:
                out["batches"] = []
                for n in batches:
                    # n>1: concurrent requests, not the n= parameter (the
                    # server rejects n>1 with 400).
                    try:
                        out["batches"].append(chat_batch(n, concurrent=n))
                    except Exception as exc:
                        out["batches"].append({"n": n, "error": repr(exc)})
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
    return out


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
        RESULT["runs"] = []
        RESULT["runs"].append(serve_once(env, spec=True, batches=[1, 4, 8]))
        RESULT["runs"].append(serve_once(env, spec=False, batches=[1, 4, 8]))

        log = LOG.read_text(errors="replace")
        markers = (
            "Capturing",
            "cudagraph",
            "CUDA graph",
            "breakable",
            "eager",
            "GPU KV cache size",
            "Memory after warmup",
            "SpecDecoding metrics",
            "Avg generation throughput",
        )
        RESULT["log_markers"] = [
            line[:300] for line in log.splitlines() if any(m in line for m in markers)
        ][-200:]
        forbidden = (
            "ptxas",
            "cvt.bf16.f32",
            "griddepcontrol",
            "illegal instruction",
            "illegal memory access",
            "no kernel image",
            "CUDA error",
            "Traceback (most recent call last)",
        )
        RESULT["forbidden_log_lines"] = [
            line[:300] for line in log.splitlines() if any(m in line for m in forbidden)
        ]
        RESULT["log_path"] = str(LOG)
    except BaseException as exc:
        RESULT["fatal"] = repr(exc)
        RESULT["fatal_traceback"] = traceback.format_exc()[-5000:]
        raise
    finally:
        finish()


if __name__ == "__main__":
    main()