# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm75 fork boundaries: config guards, dtype boundaries and DSpark norm."""

import pytest
import torch

from vllm.models.inkling2.configs import Inkling2DSparkConfig, Inkling2MMConfig
from vllm.models.inkling2.quant_config import Inkling2QuantConfig
from vllm.platforms import current_platform

_TARGET: dict = {
    "architectures": ["Inkling2ForConditionalGeneration"],
    "model_type": "inkling2",
    "text_config": {
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "local_layer_ids": [0],
        "vocab_size": 1024,
        "unpadded_vocab_size": 1000,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
    },
    "mtp_config": {
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
}


@pytest.mark.parametrize(
    "dtype", ["float16", "bfloat16", None, torch.float16, torch.bfloat16]
)
def test_dspark_config_follows_target_dtype(dtype):
    """The draft spec inherits the target dtype instead of forcing bf16."""
    overrides = {} if dtype is None else {"dtype": dtype}
    config = Inkling2MMConfig(**_TARGET, **overrides)
    draft = Inkling2DSparkConfig.from_target(config)
    expected = torch.bfloat16
    if dtype is not None:
        expected = getattr(torch, str(dtype).removeprefix("torch."))
    assert draft.dtype == expected


def test_quant_config_accepts_fp16_on_sm75():
    """The Inkling2 quantizer admits fp16 activations and Turing (cap 75)."""
    quant = Inkling2QuantConfig({"model": {"parameter_groups": []}})
    assert torch.float16 in quant.get_supported_act_dtypes()
    assert quant.get_min_capability() == 75


def test_quant_config_rejects_bf16_on_sm75(monkeypatch):
    """Config validation must reject bf16 before the MoE GEMMs reach Turing."""
    quant = Inkling2QuantConfig({"model": {"parameter_groups": []}})
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: capability == 75
    )
    assert quant.get_supported_act_dtypes() == [torch.float16]

    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: False
    )
    assert quant.get_supported_act_dtypes() == [torch.bfloat16, torch.float16]


def test_humming_dtype_is_fp16_only_on_sm75(monkeypatch, default_vllm_config):
    """Humming keeps bf16 off sm75 but follows the model dtype elsewhere."""
    from vllm.models.inkling2.nvidia.ops.moe_sm90 import _humming_activation_dtype

    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: capability == 75
    )
    assert _humming_activation_dtype() == torch.float16

    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: False
    )
    assert _humming_activation_dtype() == default_vllm_config.model_config.dtype


def test_humming_input_schema_is_fp16_only_on_sm75(monkeypatch):
    """A caller-supplied schema cannot reintroduce bf16 activations on sm75."""
    pytest.importorskip("humming")
    from vllm.models.inkling2.nvidia.ops.moe_sm90 import _normalize_input_schema
    from vllm.utils.humming import HummingInputSchema, dtypes

    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: capability == 75
    )
    assert _normalize_input_schema(HummingInputSchema()).a_dtype == dtypes.float16
    assert (
        _normalize_input_schema(HummingInputSchema(a_dtype=dtypes.bfloat16)).a_dtype
        == dtypes.float16
    )

    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: False
    )
    assert _normalize_input_schema(HummingInputSchema()).a_dtype is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_qk_norm_rope_reads_norm_weight_dtype(weight_dtype):
    """QK norm reads weights at their real dtype, not a hardcoded bf16 pointer.

    The draft's `_RMSNorm.weight` is created fp32 and loaded in place, so an
    fp32 pointer (or a bf16 one for bf16 weights) is required; reading fp32
    storage through a bf16 pointer silently produces garbage.
    """
    from vllm.models.inkling2.nvidia.ops.dspark import qk_norm_rope_

    torch.manual_seed(851)
    tokens, num_q_heads, num_k_heads, head_dim = 3, 2, 1, 16
    half = head_dim // 2
    q_size = num_q_heads * head_dim
    width = q_size + 2 * num_k_heads * head_dim
    device = "cuda"

    qkv = torch.randn(tokens, width, device=device, dtype=torch.float16)
    positions = torch.tensor([0, 2, 1], device=device, dtype=torch.int64)
    q_weight = (0.5 + torch.rand(head_dim, device=device)).to(weight_dtype)
    k_weight = (0.5 + torch.rand(head_dim, device=device)).to(weight_dtype)
    cos_sin = torch.randn(8, head_dim, device=device)
    eps = 1e-6

    expected = qkv.clone()
    for t in range(tokens):
        for h in range(num_q_heads + num_k_heads):
            is_q = h < num_q_heads
            col = h * head_dim if is_q else q_size + (h - num_q_heads) * head_dim
            weight = q_weight if is_q else k_weight
            x = qkv[t, col : col + head_dim].float()
            inv = torch.rsqrt(x.square().mean() + eps)
            x1 = (x[:half] * inv * weight[:half].float()).bfloat16().float()
            x2 = (x[half:] * inv * weight[half:].float()).bfloat16().float()
            cos = cos_sin[positions[t], :half]
            sin = cos_sin[positions[t], half:]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            expected[t, col : col + half] = o1.bfloat16().to(qkv.dtype)
            expected[t, col + half : col + head_dim] = o2.bfloat16().to(qkv.dtype)

    qk_norm_rope_(
        qkv,
        positions,
        q_weight,
        k_weight,
        cos_sin,
        num_q_heads,
        num_k_heads,
        eps,
    )
    torch.testing.assert_close(qkv, expected, atol=1e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_add_softcap_rounds_sum_to_bf16(dtype):
    """The softcap sum rounds to bf16 precision without bf16 PTX (sm75)."""
    from vllm.models.inkling2.nvidia.ops.dspark import add_softcap

    torch.manual_seed(851)
    rows, n, softcap = 5, 257, 30.0
    a = torch.randn(rows, n, device="cuda", dtype=dtype) * 4
    b = torch.randn(rows, n, device="cuda", dtype=dtype) * 4

    out = add_softcap(a, b, softcap)
    ref = (a.float() + b.float()).bfloat16().float() / softcap
    ref = (torch.tanh(ref) * softcap).bfloat16().to(dtype)

    assert out.dtype == dtype
    # The reference output is bf16-quantized, so only bf16 precision is exact.
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_markov_gate_rounds_each_step_to_bf16(dtype):
    """The Markov head rounds every step to bf16 like the reference, in fp16."""
    from vllm.models.inkling2.nvidia.ops.dspark import markov_gate

    torch.manual_seed(852)
    rows, rank = 4, 256
    raw = torch.randn(rows, 3 * rank, device="cuda", dtype=torch.float32)
    state = torch.randn(rows, rank, device="cuda", dtype=dtype)

    out, latent = markov_gate(raw, state)

    def rbf16(x):
        return x.bfloat16().float()

    raw_r = rbf16(raw)
    gate = rbf16(torch.sigmoid(raw_r[:, :rank]))
    cand = rbf16(torch.tanh(raw_r[:, rank : 2 * rank]))
    out_r = rbf16(torch.tanh(raw_r[:, 2 * rank :]))
    keep = rbf16(gate * state.float())
    update = rbf16(rbf16(1.0 - gate) * cand)
    ref_out = rbf16(keep + update).to(dtype)
    ref_latent = rbf16(torch.tanh(out_r)).to(dtype)

    assert out.dtype == latent.dtype == dtype
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(latent, ref_latent, atol=1e-2, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_swiglu_output_dtype(out_dtype):
    """SwiGLU emits the Humming activation dtype; fp16 avoids illegal bf16 PTX."""
    from vllm.models.inkling2.nvidia.ops.moe_sm90 import swiglu

    rows, n = 4, 128
    input_dtype = torch.float16 if out_dtype == torch.float16 else torch.bfloat16
    gate_up = torch.randn(rows, 2 * n, device="cuda", dtype=input_dtype)
    weights = torch.randn(rows, device="cuda", dtype=torch.float32)
    src2dst = torch.arange(rows, device="cuda", dtype=torch.int32)
    out = swiglu(gate_up, weights, src2dst, 4.0, 25.0, out_dtype=out_dtype)
    assert out.dtype == out_dtype
    assert out.shape == (rows, n)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rmsnorm_module_follows_input_dtype():
    """With no forced out_dtype the norm writes in the model (input) dtype."""
    from vllm.models.inkling2.nvidia.model import Inkling2RMSNorm

    norm = Inkling2RMSNorm(128, 1e-6)
    assert norm.out_dtype is None
    x = torch.randn(7, 128, device="cuda", dtype=torch.float16)
    out = norm(x)
    assert out.dtype == torch.float16
    ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(out, ref.to(torch.float16), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_loop_norms_honor_out_dtype(out_dtype):
    """The loop pair/blend norms write in the dtype the controller GEMMs use."""
    from vllm.models.inkling2.nvidia.ops.loop import blend_rms_norm, pair_rms_norm

    streams, tokens, width = 2, 5, 64
    device = "cuda"
    dest = torch.randn(streams, tokens, width, device=device, dtype=torch.float32)
    message = torch.randn(streams, tokens, width, device=device, dtype=torch.float32)
    gain_forget = torch.randn(
        streams, tokens, 2 * width, device=device, dtype=torch.float32
    )
    norm_weight = torch.ones(streams, width, device=device, dtype=torch.float32)

    pair = pair_rms_norm(dest, message, out_dtype=out_dtype)
    assert pair.dtype == out_dtype
    blend = blend_rms_norm(
        dest, message, gain_forget, norm_weight, 1e-6, out_dtype=out_dtype
    )
    assert blend.dtype == out_dtype


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    normed = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_dspark_rmsnorm_fp32_fallback_on_sm75(monkeypatch, dtype):
    """On sm75 `_RMSNorm` matches an fp32 reference, in the input dtype.

    The fallback computes in fp32 and casts once, so the result is exact
    against a reference that does the same; the residual path updates the
    residual in place (input dtype) and writes the norm back into `x`.
    """
    from vllm.models.inkling2.nvidia.dspark import _RMSNorm

    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: capability == 75
    )
    torch.manual_seed(851)
    hidden, eps = 128, 1e-6
    norm = _RMSNorm(hidden, eps)
    with torch.no_grad():
        norm.weight.copy_(0.5 + torch.rand(hidden))
    weight = norm.weight.detach().clone()

    atol, rtol = (0.0, 0.0) if dtype == torch.float32 else (1e-3, 1e-3)

    x = torch.randn(7, hidden, dtype=dtype)
    out = norm(x)
    assert out.dtype == dtype
    torch.testing.assert_close(out, _ref_rmsnorm(x, weight, eps), atol=atol, rtol=rtol)

    x = torch.randn(7, hidden, dtype=dtype)
    residual = torch.randn(7, hidden, dtype=dtype)
    x_orig, residual_orig = x.clone(), residual.clone()
    out = norm(x, residual)
    assert out is x and out.dtype == dtype
    torch.testing.assert_close(residual, residual_orig + x_orig, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        out, _ref_rmsnorm(residual_orig + x_orig, weight, eps), atol=atol, rtol=rtol
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dspark_rmsnorm_selects_fallback_on_sm75(monkeypatch):
    """sm75 takes the pure-PyTorch path; other capabilities use flashinfer."""
    from vllm.models.inkling2.nvidia.dspark import _RMSNorm

    norm = _RMSNorm(16, 1e-6)
    x = torch.randn(4, 16, dtype=torch.float32)

    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: capability == 75
    )
    # The fallback runs without importing flashinfer.
    assert norm(x).dtype == x.dtype

    fi_norm = pytest.importorskip("flashinfer.norm")
    used: list[str] = []
    monkeypatch.setattr(
        fi_norm, "rmsnorm", lambda x, w, eps: used.append("rmsnorm") or x
    )
    monkeypatch.setattr(
        fi_norm,
        "fused_add_rmsnorm",
        lambda *args, **kwargs: used.append("fused_add"),
    )
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: False
    )
    norm(x)
    norm(x, torch.zeros_like(x))
    assert used == ["rmsnorm", "fused_add"]


def test_vision_encoder_follows_model_dtype(default_vllm_config):
    """The vision encoder is replicated per TP rank and must not force bf16:
    Turing has no bf16 tensor cores, so `--dtype half` has to reach it too."""
    from vllm.models.inkling2.nvidia.vision import Inkling2VisionEncoder

    config = {
        "temporal_patch_size": 2,
        "native_patch_size": 20,
        "adapter_merge_size": 2,
        "projection_hidden_dim": 256,
        "projector_bias": False,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "mlp_ratio": 4.0,
        "posemb_grid_size": 32,
        "layer_norm_eps": 1e-5,
    }
    # Inside a vLLM config the dtype comes from the model; outside it (tests,
    # standalone use) the reference bf16 default is kept.
    enc = Inkling2VisionEncoder(config, text_hidden_size=192)
    assert all(p.dtype == torch.bfloat16 for p in enc.parameters())

    enc = Inkling2VisionEncoder(config, text_hidden_size=192, dtype=torch.float16)
    assert all(p.dtype == torch.float16 for p in enc.parameters())
