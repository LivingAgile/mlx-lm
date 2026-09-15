"""DeepSeek-V4.1-Flash: exact architecture registration, config, and packed
FP8(E4M3)/E8M0 + FP4(E2M1)/E8M0 numerical primitives.

This module intentionally implements a bounded slice of the full V4.1
architecture (Plan 0051 M2). It provides:

  * ``ModelArgs`` (plus nested ``TextConfig``/``VisionConfig``/
    ``QuantizationConfig``) that mirror the official pinned HF ``config.json``
    (``deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277``)
    field-for-field. Every field that is architecturally load-bearing (Engram,
    CED/index-source cross-layer sharing, Hyper-Connections, sparse/candidate
    attention, DSpark/MTP, vision) has **no default value**, so
    ``BaseModelArgs.from_dict`` raises a ``TypeError`` instead of silently
    constructing a V4-shaped config when a required V4.1 field is missing.
  * Faithful packed-format dequantization primitives
    (``dequantize_fp8_block``, ``unpack_fp4_e2m1``/``dequantize_fp4_block``,
    ``decode_e8m0_scale``) matching the official reference numerical recipe in
    ``inference/kernel.py`` (``act_quant``/``fp4_act_quant``) and
    ``inference/convert.py`` (FP4 nibble order, E8M0 scale decode), verified
    against the pinned revision real sampled tensor bytes in
    ``m1-layout-probes.md`` (see ``tests/test_deepseek_v41.py``).
  * The mandatory ``wo_a`` FP8-at-rest -> dense-``bfloat16`` exception
    (``dequantize_wo_a``), matching ``inference/convert.py`` special case:
    every other packed Linear in the official reference stays packed and is
    dequantized only transiently per-GEMM-call, but ``wo_a`` is hardcoded
    ``bfloat16`` in ``inference/model.py`` because it is consumed through a
    block-diagonal ``einsum`` that a quantized GEMM cannot express.

Explicitly out of scope for this slice (tracked as the next M2 dependency,
not faked here): the window/candidate/index sparse-attention stack, the
Hyper-Connections Sinkhorn split/merge, 384+1 MoE routing, sparse Engram
gather, vision/aligner token-budget arithmetic, and DSpark/MTP. ``Model``
is registered (``model_type == "deepseek_v41"``, not a ``deepseek_v4`` alias)
so config validation and packed-weight loading are exercisable now, but
``Model.__call__`` raises ``NotImplementedError`` naming the deferred pieces
rather than silently producing an unfaithful forward pass.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class VisionConfig(BaseModelArgs):
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    patch_size: int
    rope_theta: float
    downsample_ratio: int
    max_image_tokens: int
    min_pixels: int
    max_wh_ratio: Optional[float] = None


@dataclass
class QuantizationConfig(BaseModelArgs):
    quant_method: str
    activation_scheme: str
    weight_block_size: List[int]
    scale_fmt: str
    expert_dtype: str


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    hidden_act: str
    swiglu_limit: float
    rms_norm_eps: float
    attention_bias: bool
    max_position_embeddings: int
    rope_theta: float
    rope_scaling: Dict[str, Any]
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: str
    topk_method: str
    norm_topk_prob: bool
    routed_scaling_factor: float
    sliding_window: int
    compress_ratios: List[int]
    compress_rope_theta: float
    kv_source_layer_ids: List[int]
    index_source_layer_ids: List[int]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    engram_layer_ids: List[int]
    engram_num_embeddings: List[int]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_token_id: int
    engram_compressed_vocab_size: int
    num_nextn_predict_layers: int
    dspark_block_size: int
    dspark_noise_token_id: int
    dspark_target_layer_ids: List[int]
    dspark_markov_rank: int
    dspark_n_routed_experts: int
    dspark_num_experts_per_tok: int
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self):
        # compress_ratios carries one entry per decode layer *plus* one per
        # MTP (num_nextn_predict_layers) layer -- confirmed against the
        # pinned official config.json (40 decode-layer entries followed by
        # num_nextn_predict_layers==3 trailing MTP zeros, 43 total). A
        # length mismatch here means a truncated/malformed config, not a
        # real checkpoint, so fail closed rather than silently truncating
        # or index-erroring deep inside per-layer construction.
        expected_len = self.num_hidden_layers + self.num_nextn_predict_layers
        if len(self.compress_ratios) != expected_len:
            raise ValueError(
                "text_config.compress_ratios must have one entry per decode "
                "layer plus one per MTP layer "
                f"(num_hidden_layers={self.num_hidden_layers} + "
                f"num_nextn_predict_layers={self.num_nextn_predict_layers} "
                f"= {expected_len}), got {len(self.compress_ratios)} entries"
            )


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: Union[TextConfig, dict]
    vision_config: Union[VisionConfig, dict]
    quantization_config: Union[QuantizationConfig, dict]
    image_token_id: int
    architectures: Optional[List[str]] = None
    dtype: str = "bfloat16"
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2
    transformers_version: Optional[str] = None

    def __post_init__(self):
        self.text_config = TextConfig.from_dict(self.text_config)
        self.vision_config = VisionConfig.from_dict(self.vision_config)
        self.quantization_config = QuantizationConfig.from_dict(
            self.quantization_config
        )


# --------------------------------------------------------------------------- #
# Packed numerical primitives                                                 #
#                                                                              #
# Faithful to inference/kernel.py (act_quant / fp4_act_quant) and             #
# inference/convert.py (cast_e2m1fn_to_e4m3fn, the wo_a special case) at the  #
# pinned revision. See tests/test_deepseek_v41.py for the fixed synthetic and #
# real-sampled-byte red-green evidence.                                       #
# --------------------------------------------------------------------------- #

# The official E2M1 magnitude/sign table, copied verbatim from
# inference/convert.py FP4_TABLE (independently confirmed byte-for-byte
# against this exact pinned-revision source in Plan 0051 M1 gap-closure).
# Index = 4-bit code; bit 3 is sign, bits 2:0 index the magnitude table
# [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0].
FP4_E2M1_TABLE = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

_WO_A_BLOCK_SIZES = (32, 128)


def decode_e8m0_scale(codes: mx.array) -> mx.array:
    """Decode an unsigned E8M0 (float8_e8m0fnu) scale byte array to float32.

    Per the OCP microscaling spec (and inference/kernel.py fast_pow2 /
    fast_round_scale), an E8M0 code stores an unbiased power-of-two
    exponent with bias 127: value = 2 ** (code - 127). Code 0xFF is the
    reserved NaN encoding. Mirrors the _scale_to_float helper already used
    for DeepSeek-V4 in this repository (deepseek_v4.py).
    """
    codes = codes.astype(mx.uint32)
    exponent = codes.astype(mx.float32) - 127.0
    scale = mx.power(mx.array(2.0, dtype=mx.float32), exponent)
    return mx.where(codes == 255, mx.array(float("nan"), dtype=mx.float32), scale)


def dequantize_fp8_block(
    weight: mx.array,
    scale: mx.array,
    block_size: Optional[int] = None,
    dtype=mx.bfloat16,
) -> mx.array:
    """Dequantize a 2-D block-quantized F8_E4M3 weight with an F8_E8M0 scale.

    weight is the raw packed byte tensor (uint8, one byte per E4M3FN
    element) and scale is the coarser E8M0 grid (uint8) where each scalar
    covers one (out_block, in_block) tile -- the scheme confirmed for
    DeepSeek-V4.1-Flash quantization_config.weight_block_size ([32, 32])
    and, at the tensor level, by inference/kernel.py fp8_gemm weight-scale
    table (shape (ceildiv(N, block), K // block)) and inference/convert.py
    wo_a special case.

    Two call modes, both failing closed on a shape mismatch instead of
    silently mis-partitioning the tensor:

      * ``block_size=None`` (default): the out/in block sizes are derived
        from an exact division of weight.shape by scale.shape, matching
        convert.py own ``out_block_size = weight.size(0) // scale.size(0)``.
        This is the historical exact-fit path and requires weight.shape to
        be evenly tiled by scale.shape on *both* axes.
      * ``block_size=<int>``: faithful pad-to-scale-grid-then-slice partial
        tail handling for the out (row/N) axis only, matching the official
        weight-scale table shape ``(ceildiv(N, block), K // block)`` -- the
        in (column/K, reduction) axis is never padded, because the pinned
        weight_block_size grid always divides it exactly; an in-axis
        mismatch fails closed rather than silently padding a reduction
        dimension. A real out-axis tail (if any) is zero-padded before the
        per-block scale multiply and sliced back off afterwards, so the
        real elements are bit-for-bit identical to the exact-fit path.

    Uses mx.from_fp8 (the available native MLX E4M3 decode) for the packed
    weight bytes, matching the pattern already used for DeepSeek-V4 in this
    repository.
    """
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "dequantize_fp8_block expects 2-D weight/scale tensors, got "
            f"weight.shape={weight.shape} scale.shape={scale.shape}"
        )
    out_dim, in_dim = weight.shape
    n_out_blocks, n_in_blocks = scale.shape

    if block_size is None:
        if out_dim % n_out_blocks or in_dim % n_in_blocks:
            raise ValueError(
                f"weight shape {weight.shape} is not evenly tiled by scale shape "
                f"{scale.shape}; pass an explicit block_size to allow a partial "
                "out-axis tail block (the pinned format never pads the in-axis)"
            )
        out_block, in_block = out_dim // n_out_blocks, in_dim // n_in_blocks
        values = mx.from_fp8(weight.astype(mx.uint8), dtype=mx.float32)
        s = decode_e8m0_scale(scale)
        values = values.reshape(n_out_blocks, out_block, n_in_blocks, in_block)
        values = values * s[:, None, :, None]
        return values.reshape(out_dim, in_dim).astype(dtype)

    if in_dim % block_size:
        raise ValueError(
            f"in_dim {in_dim} is not evenly divisible by block_size={block_size}; "
            "only the out-axis (row) tail may be padded, matching the pinned "
            "weight-scale table shape (ceildiv(N, block), K // block)"
        )
    expected_n_in_blocks = in_dim // block_size
    expected_n_out_blocks = -(-out_dim // block_size)  # ceil division
    if (n_out_blocks, n_in_blocks) != (expected_n_out_blocks, expected_n_in_blocks):
        raise ValueError(
            f"scale shape {scale.shape} does not match the expected "
            f"{(expected_n_out_blocks, expected_n_in_blocks)} ceil-div-out grid "
            f"for weight shape {weight.shape} at block_size={block_size}"
        )
    pad_out = n_out_blocks * block_size - out_dim
    values = mx.from_fp8(weight.astype(mx.uint8), dtype=mx.float32)
    if pad_out:
        values = mx.pad(values, ((0, pad_out), (0, 0)))
    s = decode_e8m0_scale(scale)
    values = values.reshape(n_out_blocks, block_size, n_in_blocks, block_size)
    values = values * s[:, None, :, None]
    values = values.reshape(n_out_blocks * block_size, in_dim)
    return values[:out_dim, :].astype(dtype)


def dequantize_wo_a(weight: mx.array, scale: mx.array) -> mx.array:
    """The mandatory wo_a FP8-at-rest -> dense-bfloat16 exception.

    Faithful to inference/convert.py wo_a.weight special case: every
    other packed Linear in the official reference stays packed FP8 and is
    dequantized only transiently per-GEMM-call inside fp8_gemm, but
    model.py hardcodes self.wo_a = ColumnParallelLinear(...,
    dtype=torch.bfloat16) because wo_a is consumed through a
    block-diagonal einsum("bsgd,grd->bsgr", o, wo_a), not a dense matmul,
    so it must be promoted to a real dense tensor once and reused, never
    kept quantized. convert.py additionally asserts the block size is one
    of 32 or 128; this function enforces the same constraint so an
    unexpected checkpoint layout fails closed. A real out-axis (row) tail
    is accepted for either official block size (a ceildiv(out_dim, block)
    scale-row count), matching dequantize_fp8_block's explicit-block_size
    tail handling; the in-axis must still divide the block size exactly.
    """
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"wo_a expects 2-D weight/scale tensors, got weight.shape="
            f"{weight.shape} scale.shape={scale.shape}"
        )
    out_dim, in_dim = weight.shape
    n_out_blocks, n_in_blocks = scale.shape
    for block in _WO_A_BLOCK_SIZES:
        if in_dim % block:
            continue
        expected_n_in_blocks = in_dim // block
        expected_n_out_blocks = -(-out_dim // block)  # ceil division
        if (n_out_blocks, n_in_blocks) == (expected_n_out_blocks, expected_n_in_blocks):
            return dequantize_fp8_block(weight, scale, block_size=block, dtype=mx.bfloat16)
    raise ValueError(
        f"wo_a scale shape {scale.shape} for weight shape {weight.shape} does not "
        f"match either official block size in {_WO_A_BLOCK_SIZES} (an out-axis "
        "tail is allowed; the in-axis must divide the block size exactly)"
    )


def unpack_fp4_e2m1(packed: mx.array) -> mx.array:
    """Unpack a raw packed-FP4 byte tensor into per-element 4-bit codes.

    Uses the official low=lower-index/high=higher-index nibble convention:
    low = x & 0x0F; high = (x >> 4) & 0x0F; the decoded pair is written in
    [low, high] order (torch.stack([...], dim=-1).flatten), i.e. byte i
    low nibble becomes logical element 2*i and its high nibble becomes
    element 2*i + 1. This was independently confirmed byte-for-byte against
    inference/convert.py cast_e2m1fn_to_e4m3fn at the pinned revision
    (Plan 0051 M1 gap-closure), not merely assumed.
    """
    x = packed.astype(mx.uint32)
    low = mx.bitwise_and(x, mx.array(0x0F, dtype=mx.uint32))
    high = mx.bitwise_and(mx.right_shift(x, mx.array(4, dtype=mx.uint32)), mx.array(0x0F, dtype=mx.uint32))
    codes = mx.stack([low, high], axis=-1)
    return codes.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def decode_fp4_e2m1_codes(codes: mx.array) -> mx.array:
    """Look up 4-bit E2M1 codes in the official FP4_TABLE magnitude/sign table."""
    table = mx.array(FP4_E2M1_TABLE, dtype=mx.float32)
    return mx.take(table, codes.astype(mx.uint32))


def dequantize_fp4_block(
    packed: mx.array,
    scale: mx.array,
    block_size: int = 32,
    dtype=mx.bfloat16,
) -> mx.array:
    """Dequantize a packed FP4 (E2M1, 2 codes/byte) row with a per-row-group
    E8M0 scale, matching inference/kernel.py fp4_act_quant (which also
    drives the routed-expert-weight-at-rest format when expert_dtype=fp4):
    one scale per contiguous block_size-element run along the last
    (logical, unpacked) axis, independent per row -- unlike the 2-D-tiled
    FP8 scheme in dequantize_fp8_block.

    The last (unpacked) axis is allowed a genuine partial tail group: the
    expected scale-grid size is ceildiv(in_dim, block_size), matching the
    official act_quant/fp4_act_quant group count. A tail group real region
    is zero-padded up to block_size before the per-group scale multiply
    and sliced back off afterwards, so the real elements are bit-for-bit
    identical to an exact-fit block.
    """
    codes = unpack_fp4_e2m1(packed)
    lead = codes.shape[:-1]
    in_dim = codes.shape[-1]
    n_blocks = -(-in_dim // block_size)  # ceil division: allow a partial tail group
    if tuple(scale.shape) != (*lead, n_blocks):
        raise ValueError(
            f"expected one E8M0 scale per {block_size}-element group "
            f"(ceil-div grid size {n_blocks}), got scale shape {scale.shape} "
            f"for unpacked shape {codes.shape}"
        )
    pad_in = n_blocks * block_size - in_dim
    values = decode_fp4_e2m1_codes(codes)
    if pad_in:
        pad_width = [(0, 0)] * (values.ndim - 1) + [(0, pad_in)]
        values = mx.pad(values, pad_width)
    s = decode_e8m0_scale(scale)
    values = values.reshape(*lead, n_blocks, block_size)
    values = values * mx.expand_dims(s, -1)
    values = values.reshape(*lead, n_blocks * block_size)
    return values[..., :in_dim].astype(dtype)


class Model(nn.Module):
    """Registration entry point for model_type == "deepseek_v41".

    This is an exact V4.1 architecture registration (not a deepseek_v4
    alias): ModelArgs fails closed on every V4.1-specific required field
    (see module docstring), and sanitize applies the real, faithful wo_a
    FP8->dense-bf16 exception at load time. The base-decode stack
    (window/candidate/index sparse attention, Hyper-Connections, MoE
    routing, sparse Engram, vision/aligner, DSpark/MTP) is explicitly out
    of scope for this slice and is not implemented here as a stub: there
    are no layers to silently produce a wrong forward pass, and __call__
    raises rather than faking a result. Loading real checkpoint weights
    against this Model will fail closed (a strict tensor-name/shape
    mismatch), which is the correct, honest outcome until the deferred
    architecture lands.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "deepseek_v41.Model does not implement a forward pass yet. "
            "Deferred (Plan 0051 M2, next dependency after this slice): "
            "128-token window attention, two-level candidate/index sparse "
            "attention with attn_sink, the four physical CED/index-source "
            "cache owners, Hyper-Connections Sinkhorn split/merge, 384+1 "
            "MoE routing, sparse Engram row-sharded gather, and "
            "vision/aligner token-budget arithmetic."
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Apply the mandatory wo_a FP8-at-rest -> dense-bf16 exception.

        Every other packed weight is passed through unchanged: the general
        dense/expert packed-weight loading and key remapping strategy is
        explicitly deferred to the base-decode-architecture slice (see
        class docstring), not faked here. Fails closed if a wo_a.weight is
        present without its paired .scale (or vice versa), and via
        dequantize_wo_a if the scale block size is not one of the
        official 32/128 block sizes (an out-axis tail is allowed).
        """
        weights = dict(weights)
        wo_a_weight_keys = [k for k in weights if k.endswith("wo_a.weight")]
        wo_a_scale_keys = [k for k in weights if k.endswith("wo_a.scale")]
        for wk in wo_a_weight_keys:
            sk = wk[: -len("weight")] + "scale"
            if sk not in weights:
                raise ValueError(
                    f"{wk} is missing its paired {sk}: wo_a is required to "
                    "be FP8-block-quantized at rest (inference/convert.py "
                    "wo_a special case); a bare bf16 wo_a.weight with no "
                    "scale is not a supported checkpoint layout."
                )
            weight = weights.pop(wk)
            scale = weights.pop(sk)
            weights[wk] = dequantize_wo_a(weight, scale)
        for sk in wo_a_scale_keys:
            if sk in weights:
                wk = sk[: -len("scale")] + "weight"
                raise ValueError(
                    f"{sk} is an orphan wo_a scale with no paired {wk}: a "
                    "bare FP8 wo_a.scale with no matching wo_a.weight is not "
                    "a supported checkpoint layout."
                )
        return weights
