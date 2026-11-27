# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm75 fork boundaries: config guards and the DSpark QK-norm pointer dtype."""

import pytest
import torch

from vllm.models.inkling2.configs import Inkling2DSparkConfig, Inkling2MMConfig
from vllm.models.inkling2.quant_config import Inkling2QuantConfig

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


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", None])
def test_dspark_config_follows_target_dtype(dtype):
    """The draft spec inherits the target dtype instead of forcing bf16."""
    overrides = {} if dtype is None else {"dtype": dtype}
    config = Inkling2MMConfig(**_TARGET, **overrides)
    draft = Inkling2DSparkConfig.from_target(config)
    assert draft.dtype == (getattr(torch, dtype) if dtype else torch.bfloat16)


def test_quant_config_accepts_fp16_on_sm75():
    """The Inkling2 quantizer admits fp16 activations and Turing (cap 75)."""
    quant = Inkling2QuantConfig({"model": {"parameter_groups": []}})
    assert torch.float16 in quant.get_supported_act_dtypes()
    assert quant.get_min_capability() == 75


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
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
