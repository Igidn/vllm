#!/usr/bin/env python
"""Vision probe: does the Inkling2 image encoder run on 2xT4 (sm75)?

Two stages, both self-contained (no serve repo, no relay):

1. The fork's own vision tests on the T4 -- `test_vision.py` (fp16 and bf16
   parametrizations) and `test_sm75_fork.py`. This is what proves the Triton
   kernels compile for sm_75 and still match the eager reference.
2. A real `vllm serve` with vision enabled at 100k context, then a chat
   request carrying a generated PNG. Records the KV-cache memory the
   replicated encoder leaves behind.

Pushed by: kaggle kernels push -p .kaggle-probe-sm75-vision
"""
from __future__ import annotations

import base64
import io
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
PORT = 8017
LOG = Path("/kaggle/working/inkling-vision-sm75.log")
RUN_LOG = Path("/kaggle/tmp/sm75-vision-run.log")
RESULT_PATH = Path("/kaggle/working/probe-sm75-vision-result.json")

FORK_URL = os.environ.get("KAGGLE_FORK_URL", "https://github.com/Igidn/vllm.git")
FORK_REF = os.environ.get(
    "KAGGLE_FORK_REF", "05a41d94df40efa731a36d818c2358f95cc16b7d"
)
PRECOMPILED_COMMIT = os.environ.get(
    "KAGGLE_PRECOMPILED_COMMIT", "8c97cff059fc0d142cbc4acef9fbba62c0e08aed"
)
PRECOMPILED_VARIANT = os.environ.get("KAGGLE_PRECOMPILED_VARIANT", "cu132")

RESULT: dict = {
    "probe": "sm75-vision",
    "fork_url": FORK_URL,
    "fork_ref": FORK_REF,
    "precompiled_commit": PRECOMPILED_COMMIT,
    "precompiled_variant": PRECOMPILED_VARIANT,
    "venv": str(VENV),
}


def run(argv, timeout=1200, env=None, cwd=None):
    cmd = " ".join(str(a) for a in argv)
    print("+", cmd, flush=True)
    p = subprocess.run(
        [str(a) for a in argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env or dict(os.environ),
        cwd=cwd,
    )
    text = ((p.stdout or "") + (p.stderr or "")).strip()
    with RUN_LOG.open("a") as handle:
        handle.write(f"+ {cmd}\n{text}\n")
    print(text[-4000:], flush=True)
    return p.returncode, text


def finish():
    RESULT_PATH.write_text(json.dumps(RESULT, indent=2, default=str))
    print("=== RESULT ===", flush=True)
    print(json.dumps(RESULT, indent=2, default=str)[-16000:], flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def clone_fork():
    if not (SRC / ".git").is_dir():
        run(["git", "init", str(SRC)], timeout=300)
        run(["git", "-C", str(SRC), "remote", "add", "origin", FORK_URL], timeout=300)
    code, out = run(
        ["git", "-C", str(SRC), "fetch", "--depth", "1", "origin", FORK_REF],
        timeout=1800,
    )
    if code != 0:
        raise RuntimeError(f"fetch of {FORK_REF} failed: {out[-2000:]}")
    code, out = run(["git", "-C", str(SRC), "checkout", "-f", "FETCH_HEAD"], timeout=300)
    if code != 0:
        raise RuntimeError(f"checkout failed: {out[-2000:]}")
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


def make_image(color: tuple[int, int, int], size: int = 256) -> bytes:
    """A solid colour, so 'what colour' has exactly one right answer."""
    from PIL import Image

    image = Image.new("RGB", (size, size), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def ask_vision(question: str, png: bytes, max_tokens: int = 96) -> dict:
    payload = json.dumps(
        {
            "model": "inkling-vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri(png)}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            # The default effort (0.9) spends the whole budget on reasoning and
            # emits no content; the colour answer needs a low effort.
            "chat_template_kwargs": {"reasoning_effort": 0.0},
        }
    ).encode()
    started = time.time()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        body = json.loads(response.read().decode())
    message = body["choices"][0]["message"]
    return {
        "elapsed_s": round(time.time() - started, 2),
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "completion_tokens": body["usage"]["completion_tokens"],
        "content": message.get("content"),
        "reasoning": message.get("reasoning_content"),
        "finish_reason": body["choices"][0].get("finish_reason"),
    }


def serve_once(env) -> dict:
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
        "102400",
        "--max-num-batched-tokens",
        "512",
        "--gpu-memory-utilization",
        "0.98",
        "--max-num-seqs",
        "8",
        # Vision on: the replicated encoder costs ~0.59 GiB per TP rank.
        "--limit-mm-per-prompt",
        '{"image": 4}',
        "--cudagraph-capture-sizes",
        "1",
        "2",
        "4",
        "8",
        "--disable-custom-all-reduce",
        "--kv-offloading-size",
        "8",
        "--hf-overrides",
        '{"speculators_config": null}',
        "--reasoning-parser",
        "inkling",
        "--tool-call-parser",
        "inkling",
        "--enable-auto-tool-choice",
        "--served-model-name",
        "inkling-vision",
    ]
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
            out = {
                "command": argv,
                "ready": ready,
                "boot_seconds": round(time.time() - started, 1),
                "exit_code": proc.poll(),
            }
            if ready:
                out["red"] = ask_vision("What colour is this image? One word.", make_image((200, 30, 30)))
                out["blue"] = ask_vision("What colour is this image? One word.", make_image((30, 30, 200)))
                out["text_only"] = ask_vision(
                    "Reply with one word: ok", make_image((30, 200, 30), size=64), max_tokens=32
                )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
    return out


ENCODER_PEAK_SCRIPT = r"""
import json, math, torch
from vllm.models.inkling2.nvidia.vision import Inkling2VisionEncoder

from huggingface_hub import hf_hub_download
path = hf_hub_download("thinkingmachines/Inkling-2-Mini", "config.json")
cfg = json.load(open(path))["vision_config"]
enc = Inkling2VisionEncoder(cfg, text_hidden_size=2304, dtype=torch.float16).cuda()
for p in enc.parameters():
    p.data.normal_(0, 0.05)
side = enc.patch_size
for n in (676, 2704):
    g = math.isqrt(n)
    pixels = torch.randint(-1, 256, (n, side, side, 3)).cuda().to(torch.float16)
    meta = enc.metadata([[1, g, g]], pixels.device)
    torch.cuda.synchronize()
    before = torch.cuda.max_memory_allocated()
    with torch.inference_mode():
        out = enc.encode(pixels, meta)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - before) / 2**30
    print(f"tokens={n} out={tuple(out.shape)} peak={peak:.3f} GiB")
"""


def encoder_peak() -> dict:
    """The encoder's own peak allocation for a max-size profiling image."""
    code, out = run([str(PY), "-c", ENCODER_PEAK_SCRIPT], timeout=1200)
    return {"code": code, "out": out[-3000:]}


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
        # The slim venv has no test deps; install what tests/conftest.py and
        # the vision tests need rather than the full test requirement set.
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(PY),
                "pytest",
                "tblib",
                "pytest-timeout",
                "pytest-asyncio",
                "pytest-forked",
                "pytest-shard",
                "pytest-rerunfailures",
            ],
            timeout=1200,
        )
        # Stage 1: the fork's vision tests on the actual T4.
        code, out = run(
            [
                str(PY),
                "-m",
                "pytest",
                "tests/models/inkling2/test_vision.py",
                "tests/models/inkling2/test_sm75_fork.py",
                "-v",
            ],
            timeout=2400,
            cwd=SRC,
        )
        RESULT["pytest_code"] = code
        RESULT["pytest_tail"] = out[-6000:]


        # Stage 2: a real server with vision enabled at 100k context.
        env = {
            **os.environ,
            "VLLM_NO_USAGE_STATS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        }
        RESULT["serve"] = serve_once(env)

        log = LOG.read_text(errors="replace")
        markers = (
            "Available KV cache memory",
            "Memory after warmup",
            "GPU KV cache size",
            "Maximum concurrency",
            "Capturing",
            "cudagraph",
            "Traceback",
        )
        RESULT["log_markers"] = [
            line[:300] for line in log.splitlines() if any(m in line for m in markers)
        ][-200:]
        forbidden = (
            "ptxas",
            "cvt.bf16.f32",
            "illegal instruction",
            "illegal memory access",
            "no kernel image",
            "CUDA error",
            "FlashAttention only supports",
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