#!/usr/bin/env python
"""Stage 0/1 probe: boot the pinned sm75 Inkling2 fork on 2xT4 and chat once.

Clones the fork by commit (never an unpinned nightly), installs it with the
base commit's precompiled extensions (all fork changes are pure Python),
checks the sm75 dtype/PDL guards, then serves Inkling-2-Mini with EP2 and
--enforce-eager. DSpark is intentionally not enabled yet.

Pushed by: kaggle kernels push -p .kaggle-probe-sm75-fork
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
RESULT_PATH = Path("/kaggle/working/probe-sm75-fork-result.json")

FORK_URL = os.environ.get("KAGGLE_FORK_URL", "https://github.com/Igidn/vllm.git")
FORK_REF = os.environ.get(
    "KAGGLE_FORK_REF", "c21d1c8835d1cad12d8c4cc8fd4b0eae91ace10b"
)
# Upstream commit the fork was cut from. Its nightly wheel supplies the CUDA
# extensions; the fork itself only changes Python files.
PRECOMPILED_COMMIT = os.environ.get(
    "KAGGLE_PRECOMPILED_COMMIT", "8c97cff059fc0d142cbc4acef9fbba62c0e08aed"
)
PRECOMPILED_VARIANT = os.environ.get("KAGGLE_PRECOMPILED_VARIANT", "cu132")

RESULT: dict = {
    "probe": "sm75-fork-stage0",
    "fork_url": FORK_URL,
    "fork_ref": FORK_REF,
    "precompiled_commit": PRECOMPILED_COMMIT,
    "precompiled_variant": PRECOMPILED_VARIANT,
    "venv": str(VENV),
}

# Runs in the fork's venv; dumps the guards that must hold on Turing.
DIAG_BODY = r'''
import json
import torch
import vllm
from vllm.platforms import current_platform
from vllm.models.inkling2.quant_config import Inkling2QuantConfig

r = {
    "vllm_version": vllm.__version__,
    "vllm_file": vllm.__file__,
    "torch": torch.__version__,
    "capability": list(torch.cuda.get_device_capability()),
    "arch_list": torch.cuda.get_arch_list(),
    "pdl": current_platform.is_arch_support_pdl(),
    "min_capability": Inkling2QuantConfig.get_min_capability(),
    "act_dtypes": [str(d) for d in Inkling2QuantConfig.get_supported_act_dtypes()],
    "gpu_free_bytes": torch.cuda.mem_get_info()[0],
}
try:
    from vllm.models.inkling2.nvidia.ops.moe_sm90 import (
        _humming_activation_dtype,
        _normalize_input_schema,
    )
    from vllm.utils.humming import HummingInputSchema

    r["humming_dtype"] = str(_humming_activation_dtype())
    r["humming_schema_a_dtype"] = str(
        _normalize_input_schema(HummingInputSchema()).a_dtype
    )
except Exception as e:
    r["humming_error"] = repr(e)

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
        "1024",
        "--gpu-memory-utilization",
        "0.95",
        "--max-num-seqs",
        "2",
        # Vision kernels are bf16-only and sm75 cannot emit bf16 PTX. The
        # target preset is text-only, so skip image profiling entirely.
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
                        "messages": [
                            {"role": "user", "content": "Say hello in one sentence."}
                        ],
                        "max_tokens": 512,
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
                except Exception as exc:
                    RESULT["chat_error"] = repr(exc)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
    RESULT["log_path"] = str(LOG)
    RESULT["log_tail"] = LOG.read_text(errors="replace")[-40000:]


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
