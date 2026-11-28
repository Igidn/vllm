#!/usr/bin/env python
"""Stage 3 probe: DSpark speculator in fp16 on 2xT4.

Pins the sm75 fork by commit, installs it with the base commit's precompiled
extensions, boots Inkling-2-Mini with --dtype half and the in-checkpoint
DSpark draft, then issues one chat request and records the per-request
speculative-decoding acceptance metrics. The pass bar is a positive draft
acceptance rate with no bf16-PTX / dtype / CUDA error in the server log.

Pushed by: kaggle kernels push -p .kaggle-probe-sm75-fork-v3
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
RESULT_PATH = Path("/kaggle/working/probe-sm75-fork-v3-result.json")

FORK_URL = os.environ.get("KAGGLE_FORK_URL", "https://github.com/Igidn/vllm.git")
FORK_REF = os.environ.get(
    "KAGGLE_FORK_REF", "3251c8bfe8964c9708b3b046d0e9c167f3e28a4e"
)
# Upstream commit the fork was cut from. Its nightly wheel supplies the CUDA
# extensions; the fork itself only changes Python files.
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

RESULT: dict = {
    "probe": "sm75-fork-stage3-dspark",
    "fork_url": FORK_URL,
    "fork_ref": FORK_REF,
    "precompiled_commit": PRECOMPILED_COMMIT,
    "precompiled_variant": PRECOMPILED_VARIANT,
    "venv": str(VENV),
    "speculative_config": SPEC_CONFIG,
}

# Runs in the fork's venv; dumps the guards that must hold on Turing and the
# draft-model wiring (no separate draft checkpoint is needed).
DIAG_BODY = r'''
import json
import torch
import vllm
from vllm.platforms import current_platform
from vllm.models.inkling2.configs import Inkling2DSparkConfig, Inkling2MMConfig
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

# The draft spec must follow the target dtype; no bf16 fallback in the metadata.
try:
    cfg = Inkling2MMConfig(
        architectures=["Inkling2ForConditionalGeneration"],
        model_type="inkling2",
        text_config={
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "local_layer_ids": [0],
            "vocab_size": 1024,
            "unpadded_vocab_size": 1000,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
        },
        mtp_config={
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "intermediate_size": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1e6,
            "max_position_embeddings": 4096,
            "local_layer_ids": [0],
            "markov_rank": 32,
            "markov_head_type": "rnn_hidden",
            "target_layer_ids": [1, 2],
        },
        dtype="float16",
    )
    r["draft_spec_dtype"] = str(Inkling2DSparkConfig.from_target(cfg).dtype)
except Exception as e:
    r["draft_spec_error"] = repr(e)

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
        # No --kv-cache-dtype: "auto" resolves to fp16 under --dtype half and
        # does not trip the explicit-fp16 + spec-decode rejection.
        "--max-model-len",
        "1024",
        "--max-num-batched-tokens",
        "1024",
        "--gpu-memory-utilization",
        "0.95",
        "--max-num-seqs",
        "2",
        # Vision kernels emit bf16 PTX on sm75; the serving path is text-only.
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
        "--per-request-spec-decode-metrics",
        "summary",
        "--speculative-config",
        SPEC_CONFIG,
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
                            {
                                "role": "user",
                                "content": (
                                    "Explain in three sentences why speculative "
                                    "decoding can speed up token generation."
                                ),
                            }
                        ],
                        "max_tokens": 256,
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
                        metrics = body.get("metrics") or {}
                        RESULT["chat_metrics"] = metrics
                        RESULT["spec_metrics"] = metrics.get("speculative_decoding")
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
        "SpeculativeConfig",
        "SpecDecoding metrics",
        "Speculative decoding",
        "Draft acceptance rate",
        "Using FlashAttention",
        "kv cache group sizes",
        "GPU KV cache size",
        "Available KV cache memory",
        "Memory after warmup",
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
        "Inkling2 FP16 KV requires speculative decoding disabled",
        "Traceback (most recent call last)",
    )
    RESULT["forbidden_log_lines"] = [
        line[:300] for line in log.splitlines() if any(m in line for m in forbidden)
    ]

    spec = RESULT.get("spec_metrics") or {}
    content = (RESULT.get("chat") or {}).get("content") or ""
    RESULT["gate"] = {
        "ready": bool(RESULT.get("ready")),
        "chat_nonempty": bool(content.strip()),
        "spec_metrics_present": bool(spec),
        "num_draft_tokens_positive": (spec.get("num_draft_tokens") or 0) > 0,
        "acceptance_rate_positive": (spec.get("draft_acceptance_rate") or 0) > 0,
        "mean_acceptance_length_gt_1": (spec.get("mean_acceptance_length") or 0) > 1,
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
