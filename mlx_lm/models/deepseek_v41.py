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

  * The exact base-decode attention and cache architecture:
    ``DeepseekV41Attention`` with its 128-token sliding window, two-level
    candidate/index sparse selection with ``attn_sink``, compress-ratio
    0/1/2 CED behaviour, and the four physical ``kv_source`` cache owners
    shared by reference across all 40 layers
    (``DeepseekV41AttentionStack``, ``DeepseekV41AttentionCache``,
    ``make_deepseek_v41_attention_caches``). These are directly
    constructible and executable today, independently of ``Model``.

Explicitly out of scope for this slice (tracked as later M2/M3 work, not
faked here): the Hyper-Connections Sinkhorn split/merge that stitches the
attention layers into a residual stream, 384+1 MoE routing, sparse Engram
gather, vision/aligner token-budget arithmetic, and DSpark/MTP. ``Model``
is registered (``model_type == "deepseek_v41"``, not a ``deepseek_v4`` alias)
so config validation and packed-weight loading are exercisable now, but
``Model.__call__`` raises ``NotImplementedError`` naming the deferred pieces
rather than silently producing an unfaithful forward pass: the attention
stack above is real, and the glue around it is honestly absent.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import _BaseCache


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
    of 32 or 128 and requires *both* weight axes to divide that block
    size exactly, yielding only the official square (32, 32) or
    (128, 128) scale-grid tile -- convert.py never ceil-divides a wo_a
    axis, so an out-axis (row) partial tail is not a supported layout
    any more than an in-axis one; this function enforces the same
    exact-division constraint on both axes so an unexpected checkpoint
    layout fails closed instead of silently zero-padding a tail.
    """
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"wo_a expects 2-D weight/scale tensors, got weight.shape="
            f"{weight.shape} scale.shape={scale.shape}"
        )
    out_dim, in_dim = weight.shape
    n_out_blocks, n_in_blocks = scale.shape
    for block in _WO_A_BLOCK_SIZES:
        if out_dim % block or in_dim % block:
            continue
        if (n_out_blocks, n_in_blocks) == (out_dim // block, in_dim // block):
            return dequantize_fp8_block(weight, scale, dtype=mx.bfloat16)
    raise ValueError(
        f"wo_a scale shape {scale.shape} for weight shape {weight.shape} does not "
        f"match either official square block size in {_WO_A_BLOCK_SIZES} (both "
        "axes must divide the block size exactly; convert.py never accepts a "
        "partial tail on either axis)"
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

    The last (unpacked, logical reduction) axis must divide block_size
    exactly: fp4_act_quant/fp4_gemm never ceil-divide or pad this
    dimension, so an in_dim that is not an exact multiple of block_size
    is not a supported layout and fails closed rather than silently
    zero-padding a partial tail group onto a reduction dimension.
    """
    codes = unpack_fp4_e2m1(packed)
    lead = codes.shape[:-1]
    in_dim = codes.shape[-1]
    if in_dim % block_size:
        raise ValueError(
            f"in_dim {in_dim} is not evenly divisible by block_size={block_size}; "
            "fp4_act_quant/fp4_gemm require the logical reduction dimension to "
            "divide the block size exactly (no partial tail group is supported)"
        )
    n_blocks = in_dim // block_size
    if tuple(scale.shape) != (*lead, n_blocks):
        raise ValueError(
            f"expected one E8M0 scale per {block_size}-element group "
            f"({n_blocks} groups), got scale shape {scale.shape} "
            f"for unpacked shape {codes.shape}"
        )
    values = decode_fp4_e2m1_codes(codes)
    s = decode_e8m0_scale(scale)
    values = values.reshape(*lead, n_blocks, block_size)
    values = values * mx.expand_dims(s, -1)
    values = values.reshape(*lead, n_blocks * block_size)
    return values.astype(dtype)


# --------------------------------------------------------------------------- #
# Base-decode attention + cache architecture                                   #
#                                                                              #
# Exact port of inference/model.py at the pinned revision                      #
# deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277:    #
# precompute_freqs_cis / apply_rotary_emb / get_window_topk_idxs / Compressor  #
# / Indexer / select_candidate_blocks / Attention / SharedAttentionRuntime,    #
# plus inference/kernel.py sparse_attn and the act_quant / fp4_act_quant       #
# inplace=True fused quantize-dequantize round trips that every KV-shaped      #
# cache write goes through. Grounded in Plan 0051                              #
# artifacts/m1-architecture-trace.md Sections 2, 3, 7 and 12.                  #
#                                                                              #
# Deliberate, declared divergences from the reference script (see the module   #
# docstring for what is not implemented here at all):                          #
#                                                                              #
#   * The reference preallocates every context-scaling cache to               #
#     max_seq_len // compress_ratio slots up front. max_position_embeddings is #
#     1,048,576 at this revision, so this port grows the four physical owners  #
#     in fixed steps instead, under a strict append-only growth invariant.     #
#   * SharedAttentionRuntime is a process-global singleton in the reference.   #
#     Here it is owned by one cache list, so two concurrent sequences cannot   #
#     silently cross-contaminate, and every cross-layer read is bound to the   #
#     statically resolved producing layer rather than to whatever the last     #
#     writer happened to leave in a global slot.                               #
#   * sparse_attn is a TileLang CUDA/HIP kernel upstream with no Metal or MLX  #
#     equivalent anywhere (m1-architecture-trace.md Section 13). What follows  #
#     is a numerically equivalent gather-and-softmax implementation in plain   #
#     MLX ops, chunked over queries to bound the gather. It is correct, not    #
#     fast; a fused Metal kernel is a separate, later concern.                 #
# --------------------------------------------------------------------------- #

# Fixed at the pinned revision (inference/model.py module constants).
FP8_ACT_BLOCK_SIZE = 32
FP4_ACT_BLOCK_SIZE = 32
COMPRESS_KV_FP4_BLOCK_SIZE = 16

_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0

# Matches the TileLang kernel finite lower bound: a query row whose indices are
# all -1 must yield an all-zero output, not a NaN.
_SPARSE_ATTN_NEG_BOUND = -1e30

# Bounds the [b, chunk, topk, head_dim] gather inside sparse_attn.
SPARSE_ATTN_QUERY_CHUNK = 128

# Slots, not tokens: the four physical CED owners grow by this many compressed
# latents at a time rather than being preallocated to max_position_embeddings.
COMPRESS_CACHE_GROWTH_SLOTS = 256


def _binade_exponent(a: mx.array) -> mx.array:
    """floor(log2(a)) for a strictly positive normal float32, read straight off
    the IEEE-754 exponent field rather than via log2, so a value that is an
    exact power of two never lands one binade low on a rounding wobble."""
    bits = a.astype(mx.float32).view(mx.uint32)
    exponent = mx.bitwise_and(
        mx.right_shift(bits, mx.array(23, dtype=mx.uint32)),
        mx.array(0xFF, dtype=mx.uint32),
    )
    return exponent.astype(mx.float32) - 127.0


def _round_scale_pow2(amax: mx.array, dtype_max_inv: float) -> mx.array:
    """2 ** ceil(log2(amax * dtype_max_inv)).

    Bit-exact port of inference/kernel.py fast_round_scale (fast_log2_ceil then
    fast_pow2), which is the scale recipe used whenever scale_fmt is set.
    quantization_config.scale_fmt is ue8m0 at the pinned revision, so this is
    the live path for every KV-shaped cache write.
    """
    v = (amax.astype(mx.float32) * dtype_max_inv).astype(mx.float32)
    bits = v.view(mx.uint32)
    exponent = mx.bitwise_and(
        mx.right_shift(bits, mx.array(23, dtype=mx.uint32)),
        mx.array(0xFF, dtype=mx.uint32),
    )
    mantissa = mx.bitwise_and(bits, mx.array(0x7FFFFF, dtype=mx.uint32))
    k = exponent.astype(mx.int32) - 127 + (mantissa != 0).astype(mx.int32)
    return mx.power(mx.array(2.0, dtype=mx.float32), k.astype(mx.float32))


def _round_half_even(t: mx.array) -> mx.array:
    """Round to nearest, ties to even: the IEEE default that the hardware FP8
    and FP4 casts in the reference kernels use."""
    floor = mx.floor(t)
    frac = t - floor
    is_even = (floor - 2.0 * mx.floor(floor * 0.5)) == 0.0
    tie = mx.where(is_even, floor, floor + 1.0)
    return mx.where(frac > 0.5, floor + 1.0, mx.where(frac < 0.5, floor, tie))


def round_to_fp8_e4m3(x: mx.array) -> mx.array:
    """Round a float32 array into the float8_e4m3fn value space, returned as
    float32 (MLX exposes mx.from_fp8 for decode but no encode primitive).

    E4M3FN: 3 mantissa bits, exponent bias 7, no infinities, finite max 448,
    smallest subnormal 2**-9. Values are clamped to +/-448 first, matching the
    explicit T.clamp in act_quant_kernel, so the finite-max saturation of the
    reference is reproduced rather than producing a NaN.
    """
    x = x.astype(mx.float32)
    magnitude = mx.minimum(mx.abs(x), _FP8_E4M3_MAX)
    # Clamp the binade at the subnormal floor so subnormals share the 2**-6 ulp
    # grid, exactly as E4M3FN encodes them.
    exponent = mx.maximum(_binade_exponent(mx.maximum(magnitude, 2.0**-9)), -6.0)
    ulp = mx.power(mx.array(2.0, dtype=mx.float32), exponent - 3.0)
    quantized = mx.minimum(_round_half_even(magnitude / ulp) * ulp, _FP8_E4M3_MAX)
    quantized = mx.where(magnitude == 0.0, mx.zeros_like(magnitude), quantized)
    return mx.sign(x) * quantized


def round_to_fp4_e2m1(x: mx.array) -> mx.array:
    """Round a float32 array into the float4_e2m1fn value space (float32 out).

    E2M1 has exactly eight magnitudes: the first half of FP4_E2M1_TABLE. The
    cascade below is round to nearest with ties to even on the stored
    significand bit, which is what the hardware cast in fp4_quant_kernel does:
    0.25 -> 0.0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0, 3.5 -> 4.0,
    5.0 -> 4.0. Input is clamped to +/-6 first, matching the explicit T.clamp
    in the reference kernel.
    """
    x = x.astype(mx.float32)
    a = mx.minimum(mx.abs(x), _FP4_E2M1_MAX)
    q = mx.full(a.shape, 6.0, dtype=mx.float32)
    q = mx.where(a <= 5.0, mx.array(4.0, dtype=mx.float32), q)
    q = mx.where(a < 3.5, mx.array(3.0, dtype=mx.float32), q)
    q = mx.where(a <= 2.5, mx.array(2.0, dtype=mx.float32), q)
    q = mx.where(a < 1.75, mx.array(1.5, dtype=mx.float32), q)
    q = mx.where(a <= 1.25, mx.array(1.0, dtype=mx.float32), q)
    q = mx.where(a < 0.75, mx.array(0.5, dtype=mx.float32), q)
    q = mx.where(a <= 0.25, mx.array(0.0, dtype=mx.float32), q)
    return mx.sign(x) * q


def _blocked(x: mx.array, block_size: int, what: str):
    n = x.shape[-1]
    if block_size <= 0 or n % block_size:
        raise ValueError(
            f"{what}: last dimension {n} is not evenly divisible by "
            f"block_size={block_size}; the reference kernels assert "
            "N % block_size == 0 and never pad a partial tail group"
        )
    lead = tuple(x.shape[:-1])
    return x.astype(mx.float32).reshape(*lead, n // block_size, block_size), lead, n


def act_quant_roundtrip(x: mx.array, block_size: int = FP8_ACT_BLOCK_SIZE) -> mx.array:
    """Fused FP8(E4M3) quantize-then-dequantize: the inplace=True path of
    inference/kernel.py act_quant with scale_fmt=ue8m0 (round_scale=True).

    Per-block amax is floored at 1e-4, the scale is the power-of-two
    fast_round_scale of amax/448, values are clamped to +/-448, cast to E4M3
    and multiplied back by the scale. The result therefore carries the exact
    simulated precision loss of the reference while staying at the input dtype,
    which is what every window, compressed and index cache in the reference
    actually stores (m1-architecture-trace.md Section 7, storage-fidelity
    caveat).
    """
    out_dtype = x.dtype
    groups, lead, n = _blocked(x, block_size, "act_quant_roundtrip")
    amax = mx.maximum(mx.max(mx.abs(groups), axis=-1, keepdims=True), 1e-4)
    scale = _round_scale_pow2(amax, 1.0 / _FP8_E4M3_MAX)
    clamped = mx.clip(groups / scale, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    y = round_to_fp8_e4m3(clamped) * scale
    return y.reshape(*lead, n).astype(out_dtype)


def fp4_act_quant_roundtrip(
    x: mx.array,
    block_size: int = FP4_ACT_BLOCK_SIZE,
    e4m3_scale: bool = False,
) -> mx.array:
    """Fused FP4(E2M1) quantize-then-dequantize: the inplace=True path of
    inference/kernel.py fp4_act_quant.

    Two scale formats, both live at the pinned revision and both reproduced
    here. The indexer keys and queries use E8M0 power-of-two scales over groups
    of 32 (e4m3_scale=False, amax floored at 6*2**-126), while the compressed
    KV latent uses E4M3 scales over groups of 16 (e4m3_scale=True, amax floored
    at 6*2**-9 so an all-zero group still gets a nonzero scale, matching the
    training-time compressed KV).
    """
    out_dtype = x.dtype
    groups, lead, n = _blocked(x, block_size, "fp4_act_quant_roundtrip")
    amax = mx.max(mx.abs(groups), axis=-1, keepdims=True)
    if e4m3_scale:
        amax = mx.maximum(amax, _FP4_E2M1_MAX * (2.0**-9))
        scale = round_to_fp8_e4m3(amax / _FP4_E2M1_MAX)
    else:
        amax = mx.maximum(amax, _FP4_E2M1_MAX * (2.0**-126))
        scale = _round_scale_pow2(amax, 1.0 / _FP4_E2M1_MAX)
    clamped = mx.clip(groups / scale, -_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    y = round_to_fp4_e2m1(clamped) * scale
    return y.reshape(*lead, n).astype(out_dtype)


def yarn_rope_frequencies(
    rope_head_dim: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> mx.array:
    """The per-pair rotary frequencies of inference/model.py
    precompute_freqs_cis, without materializing the position table.

    The reference builds a [max_seq_len, rope_head_dim // 2] complex table at
    construction time. max_position_embeddings is 1,048,576 at this revision,
    so that table alone would be hundreds of megabytes per layer; positions are
    applied on demand here instead. The frequency vector itself, YaRN ramp
    included, is unchanged: dimensions whose wavelength already fits inside the
    training context keep their frequency, those far beyond it are divided by
    factor, and the beta_fast..beta_slow band is faded across with a linear
    ramp. original_seq_len == 0 disables YaRN, which is what a pure
    sliding-window (compress_ratio == 0) layer uses.
    """
    if rope_head_dim % 2:
        raise ValueError(f"rope_head_dim must be even, got {rope_head_dim}")
    freqs = 1.0 / mx.power(
        mx.array(float(base), dtype=mx.float32),
        mx.arange(0, rope_head_dim, 2, dtype=mx.float32) / rope_head_dim,
    )
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return (
                rope_head_dim
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), rope_head_dim - 1)
        ramp = mx.clip(
            (mx.arange(rope_head_dim // 2, dtype=mx.float32) - low)
            / max(high - low, 1e-3),
            0.0,
            1.0,
        )
        smooth = 1.0 - ramp
        freqs = freqs / factor * (1.0 - smooth) + freqs * smooth
    return freqs


def rope_cos_sin(freqs: mx.array, positions: mx.array):
    """cos/sin rows for the given absolute positions, [n, rope_head_dim // 2]."""
    angles = positions.astype(mx.float32).reshape(-1, 1) * freqs.reshape(1, -1)
    return mx.cos(angles), mx.sin(angles)


def apply_rope_tail(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    rope_head_dim: int,
    inverse: bool = False,
) -> mx.array:
    """Rotate only the last rope_head_dim components of x, taking adjacent
    element pairs as complex numbers.

    Functional port of apply_rotary_emb(x[..., -rd:], freqs_cis, inverse) from
    inference/model.py. inverse conjugates the rotation, which is how the
    attention output has the query rotation removed again so one shared rotated
    cache serves every query. Accepts [b, s, d] and [b, s, h, d]; the
    un-rotated head of the vector passes through untouched and the rotated tail
    is computed in float32 then cast back, matching the reference x.float() /
    y.copy_(x) round trip.
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"apply_rope_tail expects a 3-D or 4-D array, got {x.shape}")
    d = x.shape[-1]
    if rope_head_dim > d:
        raise ValueError(f"rope_head_dim={rope_head_dim} exceeds last dim {d}")
    tail = x[..., d - rope_head_dim :]
    pairs = tail.astype(mx.float32).reshape(*tail.shape[:-1], rope_head_dim // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    if x.ndim == 3:
        c, s = cos[None, :, :], sin[None, :, :]
    else:
        c, s = cos[None, :, None, :], sin[None, :, None, :]
    if inverse:
        s = -s
    rotated = mx.stack([even * c - odd * s, even * s + odd * c], axis=-1)
    rotated = rotated.reshape(tail.shape).astype(x.dtype)
    if rope_head_dim == d:
        return rotated
    return mx.concatenate([x[..., : d - rope_head_dim], rotated], axis=-1)


def window_topk_idxs(
    window_size: int, batch_size: int, seqlen: int, start_pos: int
) -> mx.array:
    """Which sliding-window slots each query attends to; -1 marks a slot that
    holds nothing. Port of inference/model.py get_window_topk_idxs.

    Prefill (start_pos == 0) indexes into the prefill chunk itself, one row per
    query with its own causal window. A decode step has a single query that
    sees the whole ring, listed oldest first. Order within a row is irrelevant
    to sparse_attn, which handles every slot independently.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if start_pos == 0:
        end = mx.arange(seqlen, dtype=mx.int32).reshape(seqlen, 1)
        idxs = mx.maximum(end - window_size + 1, 0) + mx.arange(
            min(seqlen, window_size), dtype=mx.int32
        )
        idxs = mx.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = mx.concatenate(
            [
                mx.arange(oldest, window_size, dtype=mx.int32),
                mx.arange(oldest, dtype=mx.int32),
            ]
        ).reshape(1, window_size)
        idxs = mx.where(idxs > start_pos, -1, idxs)
    return mx.broadcast_to(
        idxs[None], (batch_size, idxs.shape[0], idxs.shape[1])
    ).astype(mx.int32)


def sparse_attn(
    q: mx.array,
    kv: mx.array,
    attn_sink: mx.array,
    topk_idxs: mx.array,
    softmax_scale: float,
    query_chunk: int = SPARSE_ATTN_QUERY_CHUNK,
) -> mx.array:
    """Index-gathering sparse multi-head attention with a learned per-head
    attn_sink folded into the softmax denominator.

    Numerically equivalent MLX implementation of the sparse_attn TileLang
    kernel in inference/kernel.py. Shapes follow it exactly: q [b, m, h, d],
    kv [b, n, d] (one shared KV head, since num_key_value_heads is
    architecturally hardcoded to 1), topk_idxs [b, m, topk] with -1 as the
    unreachable sentinel, output [b, m, h, d].

    The kernel seeds its running maximum at a finite -1e30 rather than -inf so
    a row whose indices are all -1 gives exp(sink + 1e30) == inf in the
    denominator, and therefore an all-zero output instead of a NaN. That
    convention is reproduced here rather than approximated, because it is the
    training kernel contract for unreachable rows.

    The kernel streams KV blocks with an online softmax; this gathers the
    selected rows instead, in query chunks, which is memory-bounded and gives
    the same result because the softmax is over exactly the same finite set.
    """
    if q.ndim != 4 or kv.ndim != 3 or topk_idxs.ndim != 3:
        raise ValueError(
            "sparse_attn expects q[b, m, h, d], kv[b, n, d], topk_idxs[b, m, topk]; "
            f"got {q.shape}, {kv.shape}, {topk_idxs.shape}"
        )
    b, m, h, d = q.shape
    bk, n, dk = kv.shape
    if (bk, dk) != (b, d):
        raise ValueError(f"sparse_attn: q {q.shape} and kv {kv.shape} disagree")
    if tuple(topk_idxs.shape[:2]) != (b, m):
        raise ValueError(
            f"sparse_attn: topk_idxs {topk_idxs.shape} does not match q {q.shape}"
        )
    if tuple(attn_sink.shape) != (h,):
        raise ValueError(
            f"sparse_attn: attn_sink {attn_sink.shape} must be one bias per head ({h})"
        )
    if query_chunk <= 0:
        raise ValueError(f"query_chunk must be positive, got {query_chunk}")

    flat_kv = kv.astype(mx.float32).reshape(b * n, d)
    row_base = (mx.arange(b, dtype=mx.int32) * n).reshape(b, 1, 1)
    sink = attn_sink.astype(mx.float32).reshape(1, 1, h)
    neg_inf = mx.array(-float("inf"), dtype=mx.float32)

    outputs = []
    for start in range(0, m, query_chunk):
        stop = min(start + query_chunk, m)
        qc = q[:, start:stop].astype(mx.float32)
        idxs = topk_idxs[:, start:stop].astype(mx.int32)
        valid = idxs >= 0
        gathered = mx.take(
            flat_kv,
            (mx.where(valid, idxs, 0) + row_base).reshape(-1),
            axis=0,
        ).reshape(b, stop - start, idxs.shape[-1], d)
        scores = mx.einsum("bchd,bctd->bcht", qc, gathered) * softmax_scale
        scores = mx.where(valid[:, :, None, :], scores, neg_inf)
        row_max = mx.maximum(mx.max(scores, axis=-1), _SPARSE_ATTN_NEG_BOUND)
        weights = mx.exp(scores - row_max[..., None])
        denom = mx.sum(weights, axis=-1) + mx.exp(sink - row_max)
        outputs.append(
            mx.einsum("bcht,bctd->bchd", weights, gathered) / denom[..., None]
        )
    return mx.concatenate(outputs, axis=1).astype(q.dtype)


def select_candidate_blocks(
    logits: mx.array,
    compress_lens,
    topk_blocks: int,
    block_size: int,
) -> mx.array:
    """Level one of the two-level sparse selection: keep the topk_blocks
    highest-scoring block_size-position blocks per query.

    Port of inference/model.py select_candidate_blocks. logits is
    [..., n_positions] with positions the query cannot reach already at -inf,
    which is what makes a block score of -inf mean not reachable yet.
    compress_lens is a plain int during decode, or an array broadcasting
    against the leading dims of logits during prefill. The block holding the
    newest position of the query is pinned in with +inf: it is only partly
    filled and would otherwise be outscored by an older, full block. Returns a
    bool mask shaped like logits, so consumers just mask and never think about
    blocks again.
    """
    if block_size <= 0:
        raise ValueError(f"candidate_block_size must be positive, got {block_size}")
    if topk_blocks <= 0:
        raise ValueError(f"candidate_topk_blocks must be positive, got {topk_blocks}")
    neg_inf = mx.array(-float("inf"), dtype=mx.float32)
    logits = logits.astype(mx.float32)
    width = logits.shape[-1]
    pad = -width % block_size
    padded = logits
    if pad:
        pad_width = [(0, 0)] * (logits.ndim - 1) + [(0, pad)]
        padded = mx.pad(logits, pad_width, constant_values=neg_inf)
    scores = mx.max(padded.reshape(*logits.shape[:-1], -1, block_size), axis=-1)
    num_blocks = scores.shape[-1]
    if isinstance(compress_lens, int):
        last = mx.array((compress_lens - 1) // block_size if compress_lens > 0 else -1)
    else:
        lens = compress_lens.astype(mx.int32)
        last = mx.where(lens > 0, (lens - 1) // block_size, -1)
    scores = mx.where(
        mx.arange(num_blocks, dtype=mx.int32) == last,
        mx.array(float("inf"), dtype=mx.float32),
        scores,
    )
    k = min(topk_blocks, num_blocks)
    chosen = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
    chosen_scores = mx.take_along_axis(scores, chosen, axis=-1)
    # Fewer reachable blocks than topk_blocks means the leftover picks came
    # back -inf; drop them rather than admitting an unreachable block.
    keep = mx.zeros(scores.shape, dtype=mx.int8)
    keep = mx.put_along_axis(
        keep, chosen, (chosen_scores > neg_inf).astype(mx.int8), axis=-1
    )
    return mx.repeat(keep.astype(mx.bool_), block_size, axis=-1)[..., :width]


@dataclass(frozen=True)
class AttentionLayerPolicy:
    """Static per-layer CED / index-source / candidate roles.

    Resolved once, from config alone, by resolve_attention_layer_policies.
    Every cross-layer read is bound to the layer that produces it here, rather
    than to whatever the last writer happened to leave in the reference
    process-global slot.
    """

    layer_id: int
    compress_ratio: int
    is_kv_source: bool
    is_index_source: bool
    owns_index_keys: bool
    is_candidate_source: bool
    uses_candidates: bool
    compress_source_layer_id: Optional[int]
    index_key_source_layer_id: Optional[int]
    topk_source_layer_id: Optional[int]


def _validate_id_list(name: str, ids: List[int], num_layers: int) -> List[int]:
    if len(set(ids)) != len(ids):
        raise ValueError(f"text_config.{name} contains duplicate layer ids: {ids}")
    if list(ids) != sorted(ids):
        raise ValueError(
            f"text_config.{name} must be in ascending execution order, got {ids}: "
            "a source layer has to run before the layers that read it"
        )
    for layer_id in ids:
        if not 0 <= layer_id < num_layers:
            raise ValueError(
                f"text_config.{name} entry {layer_id} is outside the backbone "
                f"range [0, {num_layers})"
            )
    return list(ids)


def resolve_attention_layer_policies(
    config: TextConfig, num_layers: Optional[int] = None
) -> List[AttentionLayerPolicy]:
    """Resolve, and strictly validate, the CED / index / candidate sharing plan.

    Encodes the structure traced in m1-architecture-trace.md Sections 2 and 3:
    only kv_source_layer_ids layers own a compressed-KV buffer and an index-key
    buffer; every other layer with compress_ratio > 0 reads the most recent
    source at or before it; only index_source_layer_ids layers run an Indexer,
    and the ones that are not also KV sources read index keys an earlier owner
    published; candidate_source_layer_id produces the level-one block mask that
    strictly later index sources consume.

    Every malformed combination fails closed here, at construction, instead of
    index-erroring or silently mis-sharing a buffer deep inside a forward pass.
    """
    num_layers = config.num_hidden_layers if num_layers is None else num_layers
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if len(config.compress_ratios) < num_layers:
        raise ValueError(
            f"text_config.compress_ratios has {len(config.compress_ratios)} entries, "
            f"fewer than the {num_layers} backbone layers being built"
        )
    if config.sliding_window <= 0:
        raise ValueError(
            "text_config.sliding_window must be positive, got "
            f"{config.sliding_window}: every layer attends over a fixed "
            "sliding-window ring"
        )
    if config.num_key_value_heads != 1:
        raise ValueError(
            "text_config.num_key_value_heads must be 1, got "
            f"{config.num_key_value_heads}: wkv projects straight to one head_dim"
            " KV vector shared by every query head, and the whole window /"
            " compressed / index cache layout is built on that single shared head"
        )
    if config.num_attention_heads % config.o_groups:
        raise ValueError(
            f"num_attention_heads={config.num_attention_heads} is not divisible by "
            f"o_groups={config.o_groups}; wo_a is block-diagonal over o_groups"
        )
    if config.qk_rope_head_dim % 2 or config.qk_rope_head_dim > config.head_dim:
        raise ValueError(
            f"qk_rope_head_dim={config.qk_rope_head_dim} must be even and no larger "
            f"than head_dim={config.head_dim}"
        )
    if config.qk_rope_head_dim > config.index_head_dim:
        raise ValueError(
            f"qk_rope_head_dim={config.qk_rope_head_dim} exceeds "
            f"index_head_dim={config.index_head_dim}; the indexer rotates the same "
            "rope tail on its own keys and queries"
        )
    if config.head_dim % FP8_ACT_BLOCK_SIZE:
        raise ValueError(
            f"head_dim={config.head_dim} is not divisible by the FP8 activation "
            f"block size {FP8_ACT_BLOCK_SIZE} used for window-KV cache writes"
        )
    if config.head_dim % COMPRESS_KV_FP4_BLOCK_SIZE:
        raise ValueError(
            f"head_dim={config.head_dim} is not divisible by the compressed-KV FP4 "
            f"block size {COMPRESS_KV_FP4_BLOCK_SIZE}"
        )
    if config.index_head_dim % FP4_ACT_BLOCK_SIZE:
        raise ValueError(
            f"index_head_dim={config.index_head_dim} is not divisible by the indexer "
            f"FP4 block size {FP4_ACT_BLOCK_SIZE}"
        )

    kv_sources = _validate_id_list(
        "kv_source_layer_ids", config.kv_source_layer_ids, num_layers
    )
    index_sources = _validate_id_list(
        "index_source_layer_ids", config.index_source_layer_ids, num_layers
    )
    missing = [i for i in kv_sources if i not in index_sources]
    if missing:
        raise ValueError(
            f"kv_source_layer_ids {missing} are not in index_source_layer_ids: a KV "
            "source is the only layer that can derive index keys from its own latent, "
            "so it must also run an Indexer"
        )
    candidate_source = config.candidate_source_layer_id
    if candidate_source >= 0 and candidate_source not in index_sources:
        raise ValueError(
            f"candidate_source_layer_id={candidate_source} is not in "
            f"index_source_layer_ids {index_sources}: only an Indexer produces the "
            "scores that level-one block selection ranks"
        )

    policies: List[AttentionLayerPolicy] = []
    last_kv_source: Optional[int] = None
    last_index_source: Optional[int] = None
    for layer_id in range(num_layers):
        ratio = config.compress_ratios[layer_id]
        if ratio < 0:
            raise ValueError(
                f"text_config.compress_ratios[{layer_id}]={ratio} is negative"
            )
        is_kv_source = layer_id in kv_sources
        is_index_source = layer_id in index_sources
        if is_kv_source and ratio < 1:
            raise ValueError(
                f"layer {layer_id} is a kv_source but "
                f"compress_ratios[{layer_id}]={ratio}: a KV source must compress at "
                "ratio >= 1"
            )
        if is_kv_source:
            last_kv_source = layer_id

        compress_source = None
        index_key_source = None
        topk_source = None
        if ratio:
            if last_kv_source is None:
                raise ValueError(
                    f"layer {layer_id} has compress_ratio={ratio} but no kv_source "
                    f"layer runs at or before it (kv_source_layer_ids={kv_sources}): "
                    "it would read a compressed-KV buffer nobody has written"
                )
            source_ratio = config.compress_ratios[last_kv_source]
            if source_ratio != ratio:
                raise ValueError(
                    f"layer {layer_id} has compress_ratio={ratio} but reads the buffer "
                    f"published by kv_source layer {last_kv_source}, which compresses "
                    f"at ratio {source_ratio}: compressed slots are addressed by "
                    "position // ratio, so a consumer must share the ratio of its "
                    "source"
                )
            compress_source = last_kv_source
            if is_index_source:
                # Index keys are derived from the latent of a compressing layer,
                # so the most recent KV source is also the index-key owner.
                index_key_source = last_kv_source
                topk_source = layer_id
            else:
                if last_index_source is None:
                    raise ValueError(
                        f"layer {layer_id} has compress_ratio={ratio} but no "
                        "index_source layer runs at or before it to publish topk_idxs"
                    )
                topk_source = last_index_source
        elif is_index_source:
            raise ValueError(
                f"layer {layer_id} is an index_source but "
                f"compress_ratios[{layer_id}]=0: a pure sliding-window layer has no "
                "compressed positions to index"
            )

        if is_index_source:
            last_index_source = layer_id

        policies.append(
            AttentionLayerPolicy(
                layer_id=layer_id,
                compress_ratio=ratio,
                is_kv_source=is_kv_source,
                is_index_source=is_index_source,
                owns_index_keys=is_kv_source,
                is_candidate_source=is_index_source and layer_id == candidate_source,
                uses_candidates=is_index_source and 0 <= candidate_source < layer_id,
                compress_source_layer_id=compress_source,
                index_key_source_layer_id=index_key_source,
                topk_source_layer_id=topk_source,
            )
        )
    return policies


class PhysicalLatentCache:
    """One context-scaling buffer with exactly one owning layer.

    There are only eight of these for the whole 40-layer backbone at the pinned
    config: four compress_kv buffers and four index-key buffers, one pair per
    kv_source layer (m1-architecture-trace.md Section 2). The other 36 layers
    hold a reference to the same object rather than allocating their own, which
    is the roughly 10x KV reduction the memory model in Section 8 depends on.

    Growth is strictly append-only and contiguous: a write must start exactly
    where the previous one ended, so an out-of-order or skipped layer produces
    a loud error instead of a silently torn buffer.
    """

    def __init__(
        self,
        kind: str,
        owner_layer_id: int,
        dim: int,
        compress_ratio: int,
        growth_slots: int = COMPRESS_CACHE_GROWTH_SLOTS,
    ):
        if growth_slots <= 0:
            raise ValueError(f"growth_slots must be positive, got {growth_slots}")
        self.kind = kind
        self.owner_layer_id = owner_layer_id
        self.dim = dim
        self.compress_ratio = compress_ratio
        self.growth_slots = growth_slots
        self.buffer: Optional[mx.array] = None
        self.length = 0

    @property
    def capacity(self) -> int:
        return 0 if self.buffer is None else self.buffer.shape[1]

    @property
    def nbytes(self) -> int:
        return 0 if self.buffer is None else self.buffer.nbytes

    def write(self, values: mx.array, slot_start: int) -> None:
        if values.ndim != 3 or values.shape[-1] != self.dim:
            raise ValueError(
                f"{self.kind} owner (layer {self.owner_layer_id}) expects "
                f"[batch, slots, {self.dim}] writes, got {values.shape}"
            )
        batch, slots, _ = values.shape
        if slots == 0:
            return
        if slot_start != self.length:
            raise ValueError(
                f"{self.kind} owner (layer {self.owner_layer_id}) growth invariant "
                f"violated: write starts at slot {slot_start} but the buffer holds "
                f"{self.length} slots. Compressed slots are append-only and "
                "contiguous; a gap means a layer was skipped or ran out of order."
            )
        needed = slot_start + slots
        if self.buffer is None or self.buffer.shape[0] != batch:
            if self.length:
                raise ValueError(
                    f"{self.kind} owner (layer {self.owner_layer_id}) cannot change "
                    f"batch size from {self.buffer.shape[0]} to {batch} mid-sequence"
                )
            self.buffer = mx.zeros(
                (batch, self._grown(needed), self.dim), dtype=values.dtype
            )
        elif self.buffer.shape[1] < needed:
            grown = mx.zeros(
                (batch, self._grown(needed), self.dim), dtype=self.buffer.dtype
            )
            grown[:, : self.length] = self.buffer[:, : self.length]
            self.buffer = grown
        self.buffer[:, slot_start:needed] = values.astype(self.buffer.dtype)
        self.length = needed

    def _grown(self, needed: int) -> int:
        blocks = (needed + self.growth_slots - 1) // self.growth_slots
        return blocks * self.growth_slots

    def read(self, slots: int) -> mx.array:
        if slots < 0:
            raise ValueError(f"cannot read {slots} slots")
        if slots > self.length:
            raise ValueError(
                f"{self.kind} owner (layer {self.owner_layer_id}) holds {self.length} "
                f"slots but {slots} were requested: the owning layer has not written "
                "this step yet, so a consumer layer ran before its source."
            )
        if self.buffer is None:
            raise ValueError(
                f"{self.kind} owner (layer {self.owner_layer_id}) has never been "
                "written"
            )
        return self.buffer[:, :slots]


class CompressorPoolState:
    """The partial group a Compressor carries across decode steps.

    Mirrors the kv_state / score_state buffers of inference/model.py Compressor,
    but lives in the cache rather than the module, because it is per-sequence
    state and not a parameter. score_state starts at -inf so an unwritten slot
    contributes nothing to the pooling softmax.
    """

    def __init__(self, compress_ratio: int, head_dim: int):
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.kv: Optional[mx.array] = None
        self.score: Optional[mx.array] = None

    def ensure(self, batch: int) -> None:
        if self.kv is not None and self.kv.shape[0] == batch:
            return
        shape = (batch, self.compress_ratio, self.head_dim)
        self.kv = mx.zeros(shape, dtype=mx.float32)
        self.score = mx.full(shape, -float("inf"), dtype=mx.float32)

    @property
    def nbytes(self) -> int:
        if self.kv is None:
            return 0
        return self.kv.nbytes + self.score.nbytes


class DeepseekV41SharedAttentionRuntime:
    """What attention layers hand down the stack instead of recomputing.

    Port of inference/model.py SharedAttentionRuntime, scoped to one cache list
    instead of being a process global. It owns the four compress_kv and four
    index-key physical buffers, carries the per-step topk_idxs and candidate
    mask, and enforces that layers execute in ascending order over a single
    consistent step, which is the assumption the reference states but never
    checks.
    """

    def __init__(self, config: TextConfig, policies: List[AttentionLayerPolicy]):
        self.config = config
        self.policies = {p.layer_id: p for p in policies}
        if not self.policies:
            raise ValueError("at least one attention layer policy is required")
        self.compress_kv_owners: Dict[int, PhysicalLatentCache] = {
            p.layer_id: PhysicalLatentCache(
                "compress_kv", p.layer_id, config.head_dim, p.compress_ratio
            )
            for p in policies
            if p.is_kv_source
        }
        self.index_key_owners: Dict[int, PhysicalLatentCache] = {
            p.layer_id: PhysicalLatentCache(
                "index_k", p.layer_id, config.index_head_dim, p.compress_ratio
            )
            for p in policies
            if p.owns_index_keys
        }
        self._first_layer_id = min(self.policies)
        self._last_layer_id: Optional[int] = None
        self.step = 0
        self._step_start_pos = 0
        self._step_seqlen = 0
        self._topk_idxs: Optional[mx.array] = None
        self._topk_layer_id: Optional[int] = None
        self._candidates: Optional[mx.array] = None
        self._candidates_layer_id: Optional[int] = None

    def enter_layer(self, layer_id: int, start_pos: int, seqlen: int) -> None:
        if layer_id not in self.policies:
            raise ValueError(f"layer {layer_id} is not part of this cache")
        if layer_id == self._first_layer_id:
            self.step += 1
            self._step_start_pos = start_pos
            self._step_seqlen = seqlen
            # topk_idxs and the candidate mask are recomputed every step by
            # their source layers, so a stale one is never legitimate.
            self._topk_idxs = None
            self._topk_layer_id = None
            self._candidates = None
            self._candidates_layer_id = None
        else:
            if self._last_layer_id is None or layer_id <= self._last_layer_id:
                raise ValueError(
                    f"layer {layer_id} ran after layer {self._last_layer_id}: "
                    "cross-layer CED sharing requires strictly ascending layer "
                    "execution within one forward pass, because every source "
                    "publishes before its consumers read"
                )
            if (start_pos, seqlen) != (self._step_start_pos, self._step_seqlen):
                raise ValueError(
                    f"layer {layer_id} was called at (start_pos={start_pos}, "
                    f"seqlen={seqlen}) but this step began at "
                    f"(start_pos={self._step_start_pos}, "
                    f"seqlen={self._step_seqlen}): all layers must advance together"
                )
        self._last_layer_id = layer_id

    def publish_topk_idxs(self, layer_id: int, idxs: mx.array) -> None:
        self._topk_idxs = idxs
        self._topk_layer_id = layer_id

    def read_topk_idxs(self, layer_id: int) -> mx.array:
        expected = self.policies[layer_id].topk_source_layer_id
        if self._topk_idxs is None:
            raise ValueError(
                f"layer {layer_id} read shared topk_idxs before index-source layer "
                f"{expected} published them this step"
            )
        if self._topk_layer_id != expected:
            raise ValueError(
                f"layer {layer_id} expects topk_idxs from index-source layer "
                f"{expected} but layer {self._topk_layer_id} published them"
            )
        return self._topk_idxs

    def publish_candidates(self, layer_id: int, candidates: mx.array) -> None:
        self._candidates = candidates
        self._candidates_layer_id = layer_id

    def read_candidates(self, layer_id: int) -> mx.array:
        expected = self.config.candidate_source_layer_id
        if self._candidates is None:
            raise ValueError(
                f"layer {layer_id} consumes level-one candidate blocks but "
                f"candidate source layer {expected} has not published them this step"
            )
        if self._candidates_layer_id != expected:
            raise ValueError(
                f"layer {layer_id} expects candidate blocks from layer {expected} but "
                f"layer {self._candidates_layer_id} published them"
            )
        return self._candidates

    @property
    def nbytes(self) -> int:
        return sum(o.nbytes for o in self.compress_kv_owners.values()) + sum(
            o.nbytes for o in self.index_key_owners.values()
        )


class DeepseekV41AttentionCache(_BaseCache):
    """Per-layer cache handle: an owned 128-slot window ring plus references to
    the shared, cross-layer physical CED buffers.

    The window ring is the only KV-shaped state every layer owns outright
    (m1-architecture-trace.md Section 7): a fixed sliding_window-slot buffer
    that never scales with context. compress_kv_owner and index_key_owner are
    references to one of the four physical owners; for a consumer layer they
    are the very same object the source layer writes, which is what makes the
    per-rank KV bound in Section 8 hold.
    """

    def __init__(
        self,
        shared: DeepseekV41SharedAttentionRuntime,
        policy: AttentionLayerPolicy,
        window_size: int,
        head_dim: int,
    ):
        self.shared = shared
        self.policy = policy
        self.layer_id = policy.layer_id
        self.window_size = window_size
        self.head_dim = head_dim
        self.offset = 0
        self.window: Optional[mx.array] = None
        self.compress_kv_writer = (
            shared.compress_kv_owners[policy.layer_id] if policy.is_kv_source else None
        )
        self.compress_kv_owner = (
            shared.compress_kv_owners[policy.compress_source_layer_id]
            if policy.compress_source_layer_id is not None
            else None
        )
        self.index_key_writer = (
            shared.index_key_owners[policy.layer_id] if policy.owns_index_keys else None
        )
        self.index_key_owner = (
            shared.index_key_owners[policy.index_key_source_layer_id]
            if policy.index_key_source_layer_id is not None
            else None
        )
        self.pool_state = (
            CompressorPoolState(policy.compress_ratio, head_dim)
            if policy.is_kv_source and policy.compress_ratio > 1
            else None
        )

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def nbytes(self) -> int:
        # Only count the buffers this layer actually owns, so summing over the
        # cache list counts each of the four physical CED owners exactly once.
        total = 0 if self.window is None else self.window.nbytes
        if self.compress_kv_writer is not None:
            total += self.compress_kv_writer.nbytes
        if self.index_key_writer is not None:
            total += self.index_key_writer.nbytes
        if self.pool_state is not None:
            total += self.pool_state.nbytes
        return total

    @property
    def state(self):
        raise NotImplementedError(
            "deepseek_v41 attention caches cannot round-trip through the flat "
            "per-layer prompt-cache state protocol yet: four physical CED buffers "
            "are shared by reference across 40 layers, so saving them per layer "
            "would either duplicate them or silently unshare them on load. "
            "Prompt-cache save/load for this architecture is deferred."
        )

    @state.setter
    def state(self, v):
        raise NotImplementedError(
            "deepseek_v41 attention cache state cannot be restored; see the getter"
        )

    def enter(self, start_pos: int, seqlen: int) -> None:
        self.shared.enter_layer(self.layer_id, start_pos, seqlen)

    def update_window(self, kv: mx.array, start_pos: int) -> mx.array:
        """Write this step into the sliding-window ring and return the KV the
        queries of this step attend over.

        Port of the cache half of inference/model.py Attention._window_kv.
        Prefill (start_pos == 0) attends over the whole chunk directly and
        seeds the ring for decode; a decode step writes one slot and attends
        over the whole ring. The reference supports exactly these two shapes,
        so a chunked prefill at start_pos > 0 fails loud here rather than
        silently writing one ring slot and dropping the rest of the chunk.
        """
        if kv.ndim != 3 or kv.shape[-1] != self.head_dim:
            raise ValueError(
                f"layer {self.layer_id} window cache expects "
                f"[batch, seqlen, {self.head_dim}] KV, got {kv.shape}"
            )
        batch, seqlen, _ = kv.shape
        win = self.window_size
        if self.window is None or self.window.shape[0] != batch:
            if start_pos != 0:
                raise ValueError(
                    f"layer {self.layer_id} window ring was never seeded: decode at "
                    f"start_pos={start_pos} requires a preceding start_pos=0 prefill "
                    "call at the same batch size"
                )
            self.window = mx.zeros((batch, win, self.head_dim), dtype=kv.dtype)
        if start_pos == 0:
            if seqlen <= win:
                self.window[:, :seqlen] = kv
            else:
                cutoff = seqlen % win
                tail = kv[:, seqlen - win :]
                self.window[:, cutoff:win] = tail[:, : win - cutoff]
                if cutoff:
                    self.window[:, :cutoff] = tail[:, win - cutoff :]
            return kv
        if seqlen != 1:
            raise ValueError(
                f"layer {self.layer_id} received a {seqlen}-token chunk at "
                f"start_pos={start_pos}. The reference implementation has exactly two "
                "shapes, a single start_pos=0 prefill and single-token decode steps; "
                "chunked prefill would desynchronize the window ring, the compressor "
                "pooling groups and the compressed rope positions, so it is refused "
                "rather than approximated."
            )
        self.window[:, start_pos % win] = kv[:, 0]
        return self.window


def make_deepseek_v41_attention_caches(
    config: TextConfig, num_layers: Optional[int] = None
) -> List[DeepseekV41AttentionCache]:
    """Build one cache per backbone attention layer, sharing exactly one
    physical compressed-KV buffer and one physical index-key buffer per
    kv_source layer.

    At the pinned config this yields 40 window rings but only 4 compress_kv and
    4 index-key buffers, not 40 of each. The policy validation in
    resolve_attention_layer_policies runs first, so a malformed sharing plan is
    rejected here rather than at the first forward pass.
    """
    policies = resolve_attention_layer_policies(config, num_layers)
    shared = DeepseekV41SharedAttentionRuntime(config, policies)
    return [
        DeepseekV41AttentionCache(
            shared, policy, config.sliding_window, config.head_dim
        )
        for policy in policies
    ]


class DeepseekV41Compressor(nn.Module):
    """Pools compress_ratio consecutive tokens into one KV latent with a learned
    softmax gate. Port of inference/model.py Compressor.

    Returns the latent before RoPE, or None while a group is still filling up,
    so during decode it only yields every compress_ratio steps and holds the
    partial group in the cache pooling state. Pre-RoPE is deliberate: the
    indexer needs the unrotated form, so the attention layer rotates afterwards.

    Ratio 1 is a plain projection with no gate and no float32 promotion, which
    is why wgate exists only above ratio 1 (and why layer 20, the ratio-1 CED
    source, has no wgate tensor in the checkpoint).
    """

    def __init__(self, config: TextConfig, compress_ratio: int):
        super().__init__()
        if compress_ratio < 1:
            raise ValueError(f"compress_ratio must be >= 1, got {compress_ratio}")
        self.compress_ratio = compress_ratio
        self.head_dim = config.head_dim
        self.norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        if compress_ratio > 1:
            self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False)

    def __call__(self, x: mx.array, start_pos: int, state):
        batch, seqlen, _ = x.shape
        ratio = self.compress_ratio
        out_dtype = x.dtype
        if ratio == 1:
            return self.norm(self.wkv(x))
        if state is None:
            raise ValueError(
                "a ratio > 1 compressor needs pooling state carried in the cache"
            )
        state.ensure(batch)
        # The softmax pooling runs in float32, which is why the reference
        # promotes these two weights to float32 in the checkpoint.
        xf = x.astype(mx.float32)
        kv = self.wkv(xf).astype(mx.float32)
        score = self.wgate(xf).astype(mx.float32)
        if start_pos == 0:
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:
                # Trailing partial group waits in the state until decode
                # completes it.
                state.kv[:, :remainder] = kv[:, cutoff:]
                state.score[:, :remainder] = score[:, cutoff:]
                kv = kv[:, :cutoff]
                score = score[:, :cutoff]
            if seqlen < ratio:
                return None
            kv = kv.reshape(batch, cutoff // ratio, ratio, self.head_dim)
            score = score.reshape(batch, cutoff // ratio, ratio, self.head_dim)
            kv = mx.sum(kv * mx.softmax(score, axis=2), axis=2)
        else:
            if seqlen != 1:
                raise ValueError(
                    "the compressor decode path handles exactly one token per step, "
                    f"got {seqlen}"
                )
            slot = start_pos % ratio
            state.kv[:, slot] = kv[:, 0]
            state.score[:, slot] = score[:, 0]
            if (start_pos + 1) % ratio:
                return None
            kv = mx.sum(
                state.kv * mx.softmax(state.score, axis=1), axis=1, keepdims=True
            )
        return self.norm(kv.astype(out_dtype))


class DeepseekV41Indexer(nn.Module):
    """Keeps the index_topk best compressed positions per query.

    Port of inference/model.py Indexer: a small side attention with FP4
    quantized query heads scored against one shared key per compressed
    position, ReLU-rectified and combined by weights_proj. With a candidate
    source this is the second of two levels; select_candidate_blocks is the
    first.

    Only a layer that compresses its own KV can derive index keys from the
    latent, so wk/k_norm exist exactly on the owns_index_keys layers; the
    others read the keys an earlier owner wrote into the shared physical
    buffer.
    """

    def __init__(self, config: TextConfig, policy: AttentionLayerPolicy):
        super().__init__()
        self.layer_id = policy.layer_id
        self.compress_ratio = policy.compress_ratio
        self.owns_k = policy.owns_index_keys
        self.is_candidate_source = policy.is_candidate_source
        self.uses_candidates = policy.uses_candidates
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = config.index_head_dim**-0.5
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
        )
        self.weights_proj = nn.Linear(
            config.hidden_size, config.index_n_heads, bias=False
        )
        if self.owns_k:
            self.wk = nn.Linear(config.head_dim, config.index_head_dim, bias=False)
            self.k_norm = nn.RMSNorm(config.index_head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, qr, latent, start_pos, offset, rope, cache):
        batch, seqlen, _ = x.shape
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        compress_len = end_pos // ratio
        neg_inf = mx.array(-float("inf"), dtype=mx.float32)

        # latent is None while a group is still filling up, so there is nothing
        # new to publish on those steps; the physical buffer keeps serving the
        # keys written on earlier steps.
        if self.owns_k and latent is not None:
            cos, sin = rope(_latent_positions(start_pos, seqlen, ratio))
            k = self.k_norm(self.wk(latent))
            k = apply_rope_tail(k, cos, sin, rd)
            k = fp4_act_quant_roundtrip(k, FP4_ACT_BLOCK_SIZE, e4m3_scale=False)
            cache.index_key_writer.write(k, start_pos // ratio)
        index_k = cache.index_key_owner.read(compress_len).astype(mx.float32)

        cos, sin = rope(mx.arange(start_pos, end_pos, dtype=mx.int32))
        q = self.wq_b(qr).reshape(batch, seqlen, self.n_heads, self.index_head_dim)
        q = apply_rope_tail(q, cos, sin, rd)
        q = fp4_act_quant_roundtrip(q, FP4_ACT_BLOCK_SIZE, e4m3_scale=False)

        weights = self.weights_proj(x).astype(mx.float32) * (
            self.softmax_scale * self.n_heads**-0.5
        )
        scores = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), index_k)
        scores = mx.sum(mx.maximum(scores, 0.0) * weights[..., None], axis=2)

        # How many compressed positions each query can see: a block becomes
        # visible once the query has passed its last token. One query per decode
        # step, so there it is just a number.
        if start_pos == 0:
            compress_lens = (mx.arange(1, seqlen + 1, dtype=mx.int32) // ratio).reshape(
                seqlen, 1
            )
            reachable = mx.arange(compress_len, dtype=mx.int32) < compress_lens
            scores = mx.where(reachable, scores, neg_inf)
        else:
            compress_lens = compress_len

        if self.is_candidate_source:
            cache.shared.publish_candidates(
                self.layer_id,
                select_candidate_blocks(
                    scores,
                    compress_lens,
                    self.candidate_topk_blocks,
                    self.candidate_block_size,
                ),
            )
        elif self.uses_candidates:
            # Level two: score with our own weights, but only inside the
            # candidate blocks the source selected.
            scores = mx.where(cache.shared.read_candidates(self.layer_id), scores, neg_inf)

        # Top-k by score, re-sorted into position order; unreachable -> -1, the
        # rest shifted by the window-KV offset they are concatenated behind.
        topk = min(self.index_topk, compress_len)
        chosen = mx.argpartition(-scores, topk - 1, axis=-1)[..., :topk]
        idxs = mx.sort(chosen, axis=-1).astype(mx.int32)
        return mx.where(idxs < compress_lens, idxs + offset, -1).astype(mx.int32)


def _latent_positions(start_pos: int, seqlen: int, ratio: int) -> mx.array:
    """Absolute rope positions for freshly produced compressed latents.

    A latent stands for the first token of its group, so group j takes position
    j * ratio (inference/model.py Attention._compress_kv and Indexer.forward
    both slice freqs_cis exactly this way).
    """
    if start_pos == 0:
        return mx.arange(0, seqlen - seqlen % ratio, ratio, dtype=mx.int32)
    return mx.array([start_pos + 1 - ratio], dtype=mx.int32)


class DeepseekV41Attention(nn.Module):
    """One base-decode attention layer. Port of inference/model.py Attention.

    Every layer attends over a fixed 128-token sliding window. A layer with
    compress_ratio > 0 additionally attends over index_topk selected compressed
    positions, which is where the long context lives; a ratio-0 layer is purely
    local. The compressed positions come from a shared physical buffer that only
    four layers in the backbone actually own and write.

    Note there is a single KV head (wkv projects straight to head_dim), shared
    by all query heads, and that the query rotation is removed again from the
    attention output before the output projection, both as in the reference.
    """

    def __init__(self, config: TextConfig, policy: AttentionLayerPolicy):
        super().__init__()
        self.policy = policy
        self.layer_id = policy.layer_id
        self.compress_ratio = policy.compress_ratio
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.window_size = config.sliding_window
        self.softmax_scale = config.head_dim**-0.5

        self.attn_sink = mx.zeros((config.num_attention_heads,), dtype=mx.float32)
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        # Block diagonal over o_groups: each group sees only its own heads.
        self.wo_a = nn.Linear(
            config.num_attention_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank, config.hidden_size, bias=False
        )
        if policy.is_kv_source:
            self.compressor = DeepseekV41Compressor(config, policy.compress_ratio)
        if policy.is_index_source:
            self.indexer = DeepseekV41Indexer(config, policy)

        # Compressing layers stretch a separate, larger rope base over the long
        # context with YaRN; purely local layers see at most 128 positions and
        # so use the plain base with no scaling at all.
        scaling = config.rope_scaling or {}
        if policy.compress_ratio:
            self._rope_freqs = yarn_rope_frequencies(
                config.qk_rope_head_dim,
                int(scaling.get("original_max_position_embeddings", 0) or 0),
                config.compress_rope_theta,
                float(scaling.get("factor", 1.0) or 1.0),
                float(scaling.get("beta_fast", 32)),
                float(scaling.get("beta_slow", 1)),
            )
        else:
            self._rope_freqs = yarn_rope_frequencies(
                config.qk_rope_head_dim, 0, config.rope_theta, 1.0, 32.0, 1.0
            )

    def rope(self, positions: mx.array) -> Tuple[mx.array, mx.array]:
        return rope_cos_sin(self._rope_freqs, positions)

    def _window_kv(self, x, cos, sin, cache, start_pos):
        batch, seqlen, _ = x.shape
        kv = self.kv_norm(self.wkv(x))
        kv = apply_rope_tail(kv, cos, sin, self.rope_head_dim)
        kv = act_quant_roundtrip(kv, FP8_ACT_BLOCK_SIZE)
        window_kv = cache.update_window(kv, start_pos)
        idxs = window_topk_idxs(self.window_size, batch, seqlen, start_pos)
        return window_kv, idxs

    def _compress_topk_idxs(self, x, qr, latent, cache, start_pos, offset, compress_len):
        if not self.policy.is_index_source:
            return cache.shared.read_topk_idxs(self.layer_id)
        if compress_len == 0:
            idxs = mx.zeros((x.shape[0], x.shape[1], 0), dtype=mx.int32)
        else:
            idxs = self.indexer(x, qr, latent, start_pos, offset, self.rope, cache)
        cache.shared.publish_topk_idxs(self.layer_id, idxs)
        return idxs

    def _compress_kv(self, x, qr, cache, start_pos, offset):
        batch, seqlen, _ = x.shape
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.policy.is_kv_source:
            latent = self.compressor(x, start_pos, cache.pool_state)
        # The indexer runs before the latent is stored because it needs the
        # pre-rope, pre-FP4 form, and it publishes topk_idxs for the layers
        # further down that share this compressed KV.
        idxs = self._compress_topk_idxs(
            x, qr, latent, cache, start_pos, offset, compress_len
        )
        if latent is not None:
            cos, sin = self.rope(_latent_positions(start_pos, seqlen, ratio))
            latent = apply_rope_tail(latent, cos, sin, self.rope_head_dim)
            latent = fp4_act_quant_roundtrip(
                latent, COMPRESS_KV_FP4_BLOCK_SIZE, e4m3_scale=True
            )
            cache.compress_kv_writer.write(latent, start_pos // ratio)
        if compress_len == 0:
            compress_kv = mx.zeros((batch, 0, self.head_dim), dtype=x.dtype)
        else:
            compress_kv = cache.compress_kv_owner.read(compress_len)
        return compress_kv, idxs

    def __call__(self, x: mx.array, cache) -> mx.array:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, seqlen, hidden] input, got {x.shape}")
        if cache.layer_id != self.layer_id:
            raise ValueError(
                f"layer {self.layer_id} was handed the cache of layer "
                f"{cache.layer_id}"
            )
        batch, seqlen, _ = x.shape
        start_pos = cache.offset
        cache.enter(start_pos, seqlen)
        cos, sin = self.rope(
            mx.arange(start_pos, start_pos + seqlen, dtype=mx.int32)
        )

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(batch, seqlen, self.n_heads, self.head_dim)
        q = apply_rope_tail(q, cos, sin, self.rope_head_dim)

        kv, topk_idxs = self._window_kv(x, cos, sin, cache, start_pos)
        if self.compress_ratio:
            # Compressed indices are offset past the window KV they get
            # concatenated behind, so one sparse gather covers both halves.
            compress_kv, compress_idxs = self._compress_kv(
                x, qr, cache, start_pos, kv.shape[1]
            )
            kv = mx.concatenate([kv, compress_kv.astype(kv.dtype)], axis=1)
            topk_idxs = mx.concatenate([topk_idxs, compress_idxs], axis=-1)

        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        o = apply_rope_tail(o, cos, sin, self.rope_head_dim, inverse=True)
        wo_a = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)
        o = mx.einsum(
            "bsgd,grd->bsgr",
            o.reshape(batch, seqlen, self.n_groups, -1),
            wo_a.astype(o.dtype),
        )
        out = self.wo_b(o.reshape(batch, seqlen, -1))
        cache.offset = start_pos + seqlen
        return out


class DeepseekV41AttentionStack(nn.Module):
    """The base-decode attention layers of the backbone, in execution order.

    This is deliberately not a full model: it takes one already-normalized
    hidden state per layer and returns one attention output per layer, leaving
    the Hyper-Connections residual mixing, the MoE feed-forward, Engram, the
    vision tower and the MTP heads to later milestones. It exists so the
    attention and cache architecture is directly executable and testable before
    any of that lands.

    Layers must be run in ascending order, which the shared runtime enforces,
    because the CED sources publish compressed KV, index keys, topk_idxs and
    candidate blocks that later layers consume.
    """

    def __init__(self, config: TextConfig, num_layers: Optional[int] = None):
        super().__init__()
        self.config = config
        self._policies = resolve_attention_layer_policies(config, num_layers)
        self.layers = [DeepseekV41Attention(config, p) for p in self._policies]

    @property
    def policies(self) -> List[AttentionLayerPolicy]:
        return list(self._policies)

    def make_cache(self) -> List[DeepseekV41AttentionCache]:
        return make_deepseek_v41_attention_caches(self.config, len(self.layers))

    def __call__(self, hidden_states: List[mx.array], cache) -> List[mx.array]:
        if len(hidden_states) != len(self.layers):
            raise ValueError(
                f"expected one input per layer ({len(self.layers)}), got "
                f"{len(hidden_states)}"
            )
        if cache is None or len(cache) != len(self.layers):
            raise ValueError(
                f"expected one cache per layer ({len(self.layers)}), got "
                f"{0 if cache is None else len(cache)}"
            )
        return [
            layer(h, c) for layer, h, c in zip(self.layers, hidden_states, cache)
        ]

class Model(nn.Module):
    """Registration entry point for model_type == "deepseek_v41".

    This is an exact V4.1 architecture registration (not a deepseek_v4
    alias): ModelArgs fails closed on every V4.1-specific required field
    (see module docstring), and sanitize applies the real, faithful wo_a
    FP8->dense-bf16 exception at load time. The base-decode attention and
    cache architecture is implemented in this module and is directly
    usable via DeepseekV41AttentionStack, but the glue that would turn it
    into an end-to-end model (Hyper-Connections residual mixing, MoE
    routing, sparse Engram, vision/aligner, DSpark/MTP) is still out of
    scope, so this Model still holds no layers and __call__ raises rather
    than faking a result. Loading real checkpoint weights
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
            "The base-decode attention and cache architecture IS "
            "implemented and executable: build DeepseekV41AttentionStack "
            "directly against make_deepseek_v41_attention_caches. Still "
            "deferred: Hyper-Connections Sinkhorn split/merge, 384+1 MoE "
            "routing, sparse Engram row-sharded gather, vision/aligner "
            "token-budget arithmetic, and DSpark/MTP."
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Apply the mandatory wo_a FP8-at-rest -> dense-bf16 exception.

        Every other packed weight is passed through unchanged: the general
        dense/expert packed-weight loading and key remapping strategy is
        explicitly deferred to the base-decode-architecture slice (see
        class docstring), not faked here. Fails closed if a wo_a.weight is
        present without its paired .scale (or vice versa), and via
        dequantize_wo_a if the weight is not evenly tiled by one of the
        official square 32/128 block sizes on both axes.
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
