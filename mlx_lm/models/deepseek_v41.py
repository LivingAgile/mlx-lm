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

  * The exact Hyper-Connections residual stream (``hc_mult == 4`` parallel
    copies): ``hc_split_sinkhorn`` with the official asymmetric
    softmax/column/row normalization order, ``hc_pre``/``hc_post``, the
    identity pre-mix, and ``DeepseekV41HyperConnections`` which encodes the
    ordering of ``Block.forward`` in which a sublayer's own coefficients are
    consumed by the *next* sublayer, never by itself.
  * The exact MoE: ``sqrtsoftplus`` scoring, ``noaux_tc`` bias-steered
    selection with deterministic ties, ``norm_topk_prob`` normalization and
    ``routed_scaling_factor``, 384 routed experts with 6 active per token
    plus exactly one shared expert, over packed-at-rest FP4/FP8 experts
    (``DeepseekV41PackedLinear``) that are decoded only for the experts a
    token actually routes to, and only for the duration of that call
    (``DeepseekV41Gate``, ``DeepseekV41Expert``, ``DeepseekV41MoE``).
    ``routed_expert_partition`` resolves the per-rank expert split, and
    ``Model.shard`` retains globally numbered local experts and binds MLX's
    cross-rank all-sum before the replicated shared expert is added.

  * The exact sparse Engram n-gram path, which is the one component that
    cannot be ported by allocating its weights: the two official tables are
    384,006,168 x 256 and 384,016,682 x 256 packed FP8 rows, ~101 GB of
    payload. ``normalize_engram_token_text`` /
    ``build_engram_compressed_token_map`` reproduce the official
    NFKC/NFD/StripAccents/Lowercase/whitespace-fold normalizer chain that
    defines the compressed vocabulary every hash multiplier is derived from;
    ``EngramLayout`` builds the disjoint prime-sized bucket ranges, which
    tile the two official tables exactly; ``EngramNgramHasher`` produces the
    ``(max_ngram_size - 1) * n_heads == 24`` row ids per token per Engram
    layer over a shared, dead-token-aware history cache;
    ``SafetensorsEngramRowStore`` reads individual rows by byte range and
    ``BoundedEngramRowCache`` dedups and bounds them;
    ``dequantize_engram_rows`` applies the row-level FP8/E8M0 decode;
    ``DeepseekV41EngramEmbedding`` shards rows across ranks behind an
    explicitly injected all-reduce (absent one, ``world_size > 1`` fails
    loud); and ``DeepseekV41Engram`` applies the signed-sqrt-sigmoid gate.
    No surface in this module accepts or returns a dense table.
  * The exact vision tower, aligner, and image-grid / token-budget arithmetic
    (``DeepseekV41Vision``, ``plan_image_grid`` / ``num_image_tokens`` /
    ``image_token_types``, ``merge_image_embeddings``). Public tensor names
    match ``inference/convert.py``; ``public_checkpoint_name`` and
    ``require_public_weights`` pin the remaining gate-bias and ``wo_a`` names
    the later loader must accept.


The registered ``Model`` composes these pieces into the exact public checkpoint
tree (``embed``, ``layers.*``, ``norm``, ``head``, ``mtp.*``, and optional
vision leaves). Its load hook keeps the two giant Engram tables file-backed and
excludes only those four payloads from ordinary MLX loading; all other tensors
remain subject to strict name and shape validation. The tokenizer-derived
compressed map is bound after tokenizer loading and must match the configured
compressed vocabulary exactly.
"""

import json
import math
import re
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Regex as TokenizerRegex
from tokenizers import normalizers as tokenizer_normalizers

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
        self._rollback = deque(maxlen=2)

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return True

    def begin_update(self) -> None:
        self._rollback.append(
            (
                self.offset,
                None if self.window is None else mx.array(self.window),
                None if self.pool_state is None or self.pool_state.kv is None else mx.array(self.pool_state.kv),
                None if self.pool_state is None or self.pool_state.score is None else mx.array(self.pool_state.score),
                None if self.compress_kv_writer is None else self.compress_kv_writer.length,
                None if self.index_key_writer is None else self.index_key_writer.length,
            )
        )

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        target = self.offset - n
        snapshot = next((entry for entry in self._rollback if entry[0] == target), None)
        if snapshot is None:
            raise ValueError(
                f"deepseek_v41 cache can only trim the last two single-token updates; "
                f"cannot roll layer {self.layer_id} back from {self.offset} to {target}"
            )
        _, window, pool_kv, pool_score, compress_len, index_len = snapshot
        self.offset = target
        self.window = window
        if self.pool_state is not None:
            self.pool_state.kv = pool_kv
            self.pool_state.score = pool_score
        if self.compress_kv_writer is not None:
            self.compress_kv_writer.length = compress_len
        if self.index_key_writer is not None:
            self.index_key_writer.length = index_len
        while self._rollback and self._rollback[-1][0] >= target:
            self._rollback.pop()
        return n

    @classmethod
    def merge(cls, caches):
        if len(caches) != 1:
            raise ValueError(
                "deepseek_v41 attention caches currently support one active "
                f"sequence per batch, got {len(caches)}"
            )
        return caches[0]

    def extract(self, idx):
        if idx != 0:
            raise IndexError(
                "deepseek_v41 singleton attention cache only has batch index 0"
            )
        return self

    def filter(self, batch_indices):
        if list(batch_indices) != [0]:
            raise ValueError(
                "deepseek_v41 attention caches currently support retaining only "
                f"the singleton batch index, got {list(batch_indices)}"
            )

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
        arrays = []
        if self.window is not None:
            arrays.append(self.window)
        if self.compress_kv_writer is not None:
            if self.compress_kv_writer.buffer is not None:
                arrays.append(self.compress_kv_writer.buffer)
        if self.index_key_writer is not None:
            if self.index_key_writer.buffer is not None:
                arrays.append(self.index_key_writer.buffer)
        if self.pool_state is not None and self.pool_state.kv is not None:
            arrays.extend((self.pool_state.kv, self.pool_state.score))
        return tuple(arrays)

    @state.setter
    def state(self, v):
        raise NotImplementedError(
            "deepseek_v41 attention cache state cannot be restored; see the getter"
        )

    @property
    def meta_state(self):
        raise NotImplementedError(
            "deepseek_v41 prompt-cache persistence is unsupported because its "
            "physical CED buffers are shared across layer cache handles"
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
        self.wq_b = _make_fp8_linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
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
        self.wq_a = _make_fp8_linear(config.hidden_size, config.q_lora_rank)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = _make_fp8_linear(
            config.q_lora_rank,
            config.num_attention_heads * config.head_dim,
        )
        self.wkv = _make_fp8_linear(config.hidden_size, config.head_dim)
        self.kv_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        # Block diagonal over o_groups: each group sees only its own heads.
        self.wo_a = nn.Linear(
            config.num_attention_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
        )
        self.wo_b = _make_fp8_linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
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
        cache.begin_update()
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


# --------------------------------------------------------------------------- #
# Hyper-Connections (hc_mult parallel residual streams)                        #
#                                                                              #
# Exact port of inference/model.py Block.hc_mixes / hc_pre / hc_post /         #
# Block.forward and inference/kernel.py hc_split_sinkhorn at the pinned        #
# revision deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db39    #
# 60a90277, plus Transformer.forward hc_mult expansion and                     #
# make_identity_pre_mix. Grounded in Plan 0051                                 #
# artifacts/m1-architecture-trace.md Sections 9 and 12 item 4.                 #
#                                                                              #
# hc_split_sinkhorn is a TileLang CUDA/HIP kernel upstream with no Metal or    #
# MLX equivalent; what follows is the same arithmetic in plain MLX ops, in     #
# float32 throughout as the kernel is, including the exact (and deliberately   #
# asymmetric) normalization order: a row softmax plus eps, then one column     #
# normalization, then sinkhorn_iters - 1 further row/column passes. It is      #
# numerically equivalent, not fused.                                           #
# --------------------------------------------------------------------------- #


def hc_split_sinkhorn(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Split one projection of the residual stream into the three Hyper-Connection
    coefficient sets, exactly as inference/kernel.py hc_split_sinkhorn_kernel does.

    ``mixes`` is ``[..., (2 + hc_mult) * hc_mult]``, laid out as three
    contiguous segments in this order (the kernel indexes them literally):

      * ``[0, hc)``          -> ``pre``  = ``sigmoid(m * hc_scale[0] + hc_base) + eps``
      * ``[hc, 2*hc)``       -> ``post`` = ``2 * sigmoid(m * hc_scale[1] + hc_base)``
      * ``[2*hc, 2*hc+hc^2)`` -> ``comb`` logits, read row-major as ``[j * hc + k]``

    ``comb`` is then made (approximately) doubly stochastic by Sinkhorn:
    one row softmax with ``+ eps``, one column normalization, then
    ``sinkhorn_iters - 1`` further row-then-column passes -- so the very
    first row pass is a softmax rather than a plain sum normalization, and
    the sequence ends on a column pass. Reproducing that asymmetry matters:
    a symmetric loop converges to a different fixed point at finite
    iteration counts.

    Returns ``(pre, post, comb)`` with shapes ``[..., hc]``, ``[..., hc]``
    and ``[..., hc, hc]``, all float32.
    """
    if hc_mult < 1:
        raise ValueError(f"hc_mult must be positive, got {hc_mult}")
    if sinkhorn_iters < 1:
        raise ValueError(
            f"hc_sinkhorn_iters must be at least 1, got {sinkhorn_iters}: the "
            "kernel always runs one softmax-plus-column pass and then "
            "sinkhorn_iters - 1 more"
        )
    if not eps > 0:
        raise ValueError(
            f"hc_eps must be positive, got {eps}: it is both the pre-mix floor "
            "and the Sinkhorn division guard"
        )
    mix_hc = (2 + hc_mult) * hc_mult
    if mixes.ndim < 1 or mixes.shape[-1] != mix_hc:
        raise ValueError(
            f"mixes last dimension must be (2 + hc_mult) * hc_mult = {mix_hc} "
            f"for hc_mult={hc_mult}, got shape {mixes.shape}"
        )
    if tuple(hc_scale.shape) != (3,):
        raise ValueError(
            f"hc_scale must have exactly 3 entries (pre/post/comb), got shape "
            f"{hc_scale.shape}"
        )
    if tuple(hc_base.shape) != (mix_hc,):
        raise ValueError(
            f"hc_base must have one entry per mix channel ({mix_hc}), got shape "
            f"{hc_base.shape}"
        )

    m = mixes.astype(mx.float32)
    scale = hc_scale.astype(mx.float32)
    base = hc_base.astype(mx.float32)
    hc = hc_mult

    pre = mx.sigmoid(m[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * mx.sigmoid(m[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])

    comb = m[..., 2 * hc :] * scale[2] + base[2 * hc :]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)

    comb = mx.softmax(comb, axis=-1) + eps
    comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    return pre, post, comb


def expand_hyper_connection_stream(h: mx.array, hc_mult: int) -> mx.array:
    """Expand ``[b, s, d]`` token embeddings into the ``[b, s, hc, d]`` residual
    stream the backbone actually carries (Transformer.forward's
    ``h.unsqueeze(2).repeat(1, 1, hc_mult, 1)``): every copy starts identical.
    """
    if hc_mult < 1:
        raise ValueError(f"hc_mult must be positive, got {hc_mult}")
    if h.ndim != 3:
        raise ValueError(
            f"expected [batch, seqlen, dim] token embeddings, got shape {h.shape}"
        )
    return mx.repeat(mx.expand_dims(h, 2), hc_mult, axis=2)


def make_identity_pre_mix(batch: int, seqlen: int, hc_mult: int) -> mx.array:
    """The initial one-hot pre-mix (inference/model.py make_identity_pre_mix).

    The first block collapses the expanded stream by selecting copy 0 only,
    which -- because every copy starts identical -- makes the first sublayer
    input exactly the token embedding.
    """
    if hc_mult < 1:
        raise ValueError(f"hc_mult must be positive, got {hc_mult}")
    if batch < 1 or seqlen < 1:
        raise ValueError(f"batch and seqlen must be positive, got {batch}, {seqlen}")
    one_hot = (mx.arange(hc_mult, dtype=mx.int32) == 0).astype(mx.float32)
    return mx.broadcast_to(one_hot, (batch, seqlen, hc_mult))


def hc_pre(x: mx.array, pre_mix: mx.array) -> mx.array:
    """Collapse the hc copies into one sublayer input: ``[b,s,hc,d]`` x ``[b,s,hc]``
    -> ``[b,s,d]``, weighted-summed in float32 and cast back (Block.hc_pre).
    """
    if x.ndim < 3:
        raise ValueError(f"expected a [..., hc, dim] stream, got shape {x.shape}")
    if tuple(pre_mix.shape) != tuple(x.shape[:-1]):
        raise ValueError(
            f"pre_mix shape {pre_mix.shape} does not match the stream's "
            f"[..., hc] axes {tuple(x.shape[:-1])}"
        )
    y = mx.sum(mx.expand_dims(pre_mix.astype(mx.float32), -1) * x.astype(mx.float32), axis=-2)
    return y.astype(x.dtype)


def hc_post(
    x: mx.array, residual: mx.array, post_mix: mx.array, comb: mx.array
) -> mx.array:
    """Expand a sublayer output back to hc copies and mix the residual stream in
    through ``comb`` (Block.hc_post).

    ``x``: ``[b,s,d]``, ``residual``: ``[b,s,hc,d]``, ``post_mix``: ``[b,s,hc]``,
    ``comb``: ``[b,s,hc,hc]`` -> ``[b,s,hc,d]``. Note ``comb`` is summed over its
    *first* (source-copy) axis, so ``comb[.., j, k]`` sends residual copy ``j``
    into output copy ``k`` -- the transposed reading silently produces a
    plausible-looking but wrong stream, which is why this is spelled out.
    """
    if residual.ndim < 3:
        raise ValueError(f"expected a [..., hc, dim] residual, got {residual.shape}")
    hc = residual.shape[-2]
    if tuple(x.shape) != tuple(residual.shape[:-2]) + (residual.shape[-1],):
        raise ValueError(
            f"sublayer output shape {x.shape} does not match the residual "
            f"stream {residual.shape} with its hc axis collapsed"
        )
    if tuple(post_mix.shape) != tuple(residual.shape[:-1]):
        raise ValueError(
            f"post_mix shape {post_mix.shape} does not match [..., hc] "
            f"{tuple(residual.shape[:-1])}"
        )
    if tuple(comb.shape) != tuple(residual.shape[:-1]) + (hc,):
        raise ValueError(
            f"comb shape {comb.shape} does not match [..., hc, hc] "
            f"{tuple(residual.shape[:-1]) + (hc,)}"
        )
    broadcast = mx.expand_dims(post_mix.astype(mx.float32), -1) * mx.expand_dims(
        x.astype(mx.float32), -2
    )
    mixed = mx.sum(
        mx.expand_dims(comb.astype(mx.float32), -1)
        * mx.expand_dims(residual.astype(mx.float32), -2),
        axis=-3,
    )
    return (broadcast + mixed).astype(x.dtype)


class DeepseekV41HyperConnections(nn.Module):
    """The six per-block Hyper-Connection parameters and the residual-stream
    semantics built on them (inference/model.py Block).

    Parameter names are the checkpoint's own flat Block-level names
    (``hc_attn_fn`` / ``hc_ffn_fn`` ``[mix_hc, hc_mult * dim]``,
    ``hc_attn_base`` / ``hc_ffn_base`` ``[mix_hc]``, ``hc_attn_scale`` /
    ``hc_ffn_scale`` ``[3]``, all float32), so a Block that composes this
    module has to flatten one level of nesting when it maps checkpoint keys.

    The load-bearing ordering subtlety, spelled out because it is easy to get
    backwards: each sublayer's own ``hc_mixes`` produces the ``pre`` mix used
    by the *next* sublayer, never by itself. Attention consumes the pre-mix the
    previous block's FFN produced; the FFN consumes the one this block's
    attention produced; and the block hands its FFN's pre-mix onward. That is
    encoded once, in ``block_step``, rather than left to each caller.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        if config.hc_mult < 1:
            raise ValueError(
                f"text_config.hc_mult must be positive, got {config.hc_mult}"
            )
        if config.hc_sinkhorn_iters < 1:
            raise ValueError(
                "text_config.hc_sinkhorn_iters must be at least 1, got "
                f"{config.hc_sinkhorn_iters}"
            )
        if not config.hc_eps > 0:
            raise ValueError(
                f"text_config.hc_eps must be positive, got {config.hc_eps}"
            )
        if config.hidden_size < 1:
            raise ValueError(
                f"text_config.hidden_size must be positive, got {config.hidden_size}"
            )
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.dim
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_attn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_ffn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_attn_scale = mx.zeros((3,), dtype=mx.float32)
        self.hc_ffn_scale = mx.zeros((3,), dtype=mx.float32)

    def _mixes(self, x, hc_fn, hc_scale, hc_base):
        if x.ndim < 3 or x.shape[-2] != self.hc_mult or x.shape[-1] != self.dim:
            raise ValueError(
                f"expected a [..., hc_mult={self.hc_mult}, dim={self.dim}] residual "
                f"stream, got shape {x.shape}"
            )
        flat = x.reshape(*x.shape[:-2], self.hc_mult * self.dim).astype(mx.float32)
        # One RMS statistic per token over the whole flattened hc*d stream,
        # applied after the projection (Block.hc_mixes).
        rsqrt = mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + self.norm_eps)
        mixes = (flat @ hc_fn.astype(mx.float32).T) * rsqrt
        return hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )

    def attn_mixes(self, x: mx.array):
        return self._mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)

    def ffn_mixes(self, x: mx.array):
        return self._mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)

    def sublayer_step(self, x: mx.array, pre_mix: mx.array, sublayer, which: str):
        """Run one sublayer inside the hc residual stream.

        Returns ``(stream, next_pre_mix)``: the coefficients derived here are
        the ones the *next* sublayer collapses with, while this sublayer's
        input is collapsed with the ``pre_mix`` handed in.
        """
        if which == "attn":
            mixes = self.attn_mixes
        elif which == "ffn":
            mixes = self.ffn_mixes
        else:
            raise ValueError(
                f"which must be 'attn' or 'ffn', got {which!r}: a block has "
                "exactly those two Hyper-Connection coefficient sets"
            )
        residual = x
        next_pre_mix, post_mix, comb = mixes(x)
        out = sublayer(hc_pre(x, pre_mix))
        return hc_post(out, residual, post_mix, comb), next_pre_mix

    def block_step(self, x: mx.array, pre_mix: mx.array, attn, ffn):
        """One full block's residual-stream traversal (Block.forward), returning
        ``(stream, next_pre_mix)`` where ``next_pre_mix`` is the FFN's.
        """
        x, attn_pre = self.sublayer_step(x, pre_mix, attn, "attn")
        return self.sublayer_step(x, attn_pre, ffn, "ffn")


# --------------------------------------------------------------------------- #
# MoE: 384 routed + 1 shared expert, noaux_tc / sqrtsoftplus routing           #
#                                                                              #
# Exact port of inference/model.py Gate / Expert / MoE at the pinned revision  #
# deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277,    #
# consuming the packed FP4(E2M1)/E8M0 and FP8(E4M3)/E8M0 primitives above.     #
# Grounded in Plan 0051 artifacts/m1-architecture-trace.md Section 9.          #
#                                                                              #
# The pinned text_config routes 6 of 384 experts per token plus exactly one    #
# shared expert every token goes through, scores with sqrtsoftplus, selects    #
# with noaux_tc (a correction bias steers *selection* only; the routing        #
# weights come from the unbiased scores), normalizes the selected weights and  #
# then scales them by routed_scaling_factor = 1.5.                             #
#                                                                              #
# Storage: routed experts are FP4-packed with an E8M0 block scale and the      #
# shared expert is FP8-block-quantized. Both are dequantized transiently,      #
# inside the GEMM call that needs them, and never persistently promoted --     #
# the whole point of DeepseekV41PackedLinear. wo_a (elsewhere in this module)  #
# is the single declared exception to that rule.                               #
# --------------------------------------------------------------------------- #

# Weight-side block sizes: quantization_config.weight_block_size is [32, 32]
# (2-D tiled, FP8 path) and inference/kernel.py fp4_gemm quantizes FP4 weights
# 1x32 along the K/reduction axis with an E8M0 scale.
FP4_WEIGHT_BLOCK_SIZE = 32
FP8_WEIGHT_BLOCK_SIZE = 32

# inference/model.py ModelArgs.gate_temp. Neither the pinned HF config.json
# text_config nor inference/config.json carries this key, so the released model
# uses the reference default and the division below is an identity; it is kept
# explicit rather than dropped so a future config that does set it has an
# obvious home.
GATE_TEMP = 1.0

# Gate.forward: "not norm_eps, matches training".
ROUTING_NORM_EPS = 1e-20

SUPPORTED_SCORING_FUNCS = ("softmax", "sigmoid", "sqrtsoftplus")
NOAUX_TC = "noaux_tc"


def routing_scores(logits: mx.array, scoring_func: str) -> mx.array:
    """Map raw gate logits to routing scores in float32 (Gate.forward).

    The pinned config uses ``sqrtsoftplus``: ``sqrt(softplus(logits))``, which
    is unbounded above (unlike sigmoid) and not normalized across experts
    (unlike softmax), so the later top-k normalization is what makes the
    selected weights sum to one.
    """
    x = logits.astype(mx.float32)
    if scoring_func == "softmax":
        return mx.softmax(x, axis=-1)
    if scoring_func == "sigmoid":
        return mx.sigmoid(x)
    if scoring_func == "sqrtsoftplus":
        # log1p(exp(x)) via logaddexp, which is stable in both tails where a
        # literal log(1 + exp(x)) overflows or cancels.
        return mx.sqrt(mx.logaddexp(x, mx.zeros_like(x)))
    raise ValueError(
        f"unsupported scoring_func {scoring_func!r}; inference/model.py Gate "
        f"implements exactly {SUPPORTED_SCORING_FUNCS}"
    )


def deterministic_topk_indices(scores: mx.array, k: int) -> mx.array:
    """Top-k indices along the last axis, descending, ties broken by lowest index.

    ``torch.topk`` (which Gate.forward uses) is deterministic in both respects,
    so routing must be too: on the flat score surfaces this gate produces --
    an all-zero gate weight makes every expert score identical -- an
    order-unspecified partition would silently pick a different expert set per
    run and per backend. ``mx.argpartition`` gives no such guarantee, so the
    selection is built from ``max`` plus a lowest-matching-index ``min``, which
    is exact by construction and needs no assumption about sort stability.

    ``k`` is small here (6 routed experts per token in the pinned config, 3 for
    the DSpark gate), so the k sequential passes are cheap.
    """
    if scores.ndim < 1:
        raise ValueError(f"expected at least a 1-D score array, got {scores.shape}")
    n = scores.shape[-1]
    if not 1 <= k <= n:
        raise ValueError(
            f"top-k k={k} is out of range for {n} candidates along the last axis"
        )
    positions = mx.arange(n, dtype=mx.int32)
    remaining = scores.astype(mx.float32)
    neg_inf = mx.array(-float("inf"), dtype=mx.float32)
    chosen = []
    for _ in range(k):
        best = mx.max(remaining, axis=-1, keepdims=True)
        # Lowest index attaining the max; n is an unreachable sentinel that a
        # non-empty row can never select.
        idx = mx.min(
            mx.where(remaining == best, positions, mx.array(n, dtype=mx.int32)),
            axis=-1,
            keepdims=True,
        )
        chosen.append(idx)
        remaining = mx.where(positions == idx, neg_inf, remaining)
    return mx.concatenate(chosen, axis=-1).astype(mx.int32)


def noaux_tc_route(
    scores: mx.array,
    bias: mx.array,
    topk: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> Tuple[mx.array, mx.array]:
    """The noaux_tc selection rule (Gate.forward).

    The correction bias steers which experts are picked and nothing else: the
    returned weights are gathered from the *unbiased* scores. Getting this
    backwards -- scaling by the biased score -- is numerically plausible and
    architecturally wrong, so it is asserted directly in the tests.
    """
    if topk < 1:
        raise ValueError(f"num_experts_per_tok must be positive, got {topk}")
    indices = deterministic_topk_indices(scores + bias, topk)
    weights = mx.take_along_axis(scores.astype(mx.float32), indices, axis=-1)
    if norm_topk_prob and topk > 1:
        weights = weights / (
            mx.sum(weights, axis=-1, keepdims=True) + ROUTING_NORM_EPS
        )
    return weights * routed_scaling_factor, indices


def routed_expert_partition(
    n_routed_experts: int, world_size: int = 1, rank: int = 0
) -> Tuple[int, int]:
    """The half-open ``[start, end)`` routed-expert range one rank owns.

    Mirrors inference/model.py MoE.__init__: the routed experts are split into
    ``world_size`` equal contiguous blocks, so the count must divide exactly --
    an uneven split is refused rather than silently leaving some experts
    unowned (every token that routed to one would then contribute nothing and
    quietly degrade output instead of failing). 384 admits world sizes
    1/2/3/4/6/8/12/16/24/32/48/64/96/128/192/384; 5 and 7 do not.
    """
    if n_routed_experts < 1:
        raise ValueError(
            f"n_routed_experts must be positive, got {n_routed_experts}"
        )
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(
            f"rank {rank} is outside the range [0, world_size={world_size})"
        )
    if n_routed_experts % world_size:
        raise ValueError(
            f"number of experts ({n_routed_experts}) must be divisible by world "
            f"size ({world_size})"
        )
    n_local = n_routed_experts // world_size
    start = rank * n_local
    return start, start + n_local


def validate_moe_routing_config(
    config: TextConfig, n_routed_experts: int, n_activated_experts: int
) -> None:
    """Fail closed on every routing config the official Gate/MoE cannot express."""
    if config.topk_method != NOAUX_TC:
        raise ValueError(
            f"text_config.topk_method must be {NOAUX_TC!r}, got "
            f"{config.topk_method!r}: inference/model.py Gate implements only the "
            "bias-corrected selection rule, with no auxiliary-loss or "
            "group-limited variant"
        )
    if config.scoring_func not in SUPPORTED_SCORING_FUNCS:
        raise ValueError(
            f"text_config.scoring_func must be one of {SUPPORTED_SCORING_FUNCS}, "
            f"got {config.scoring_func!r}"
        )
    if config.n_shared_experts != 1:
        raise ValueError(
            "text_config.n_shared_experts must be exactly 1, got "
            f"{config.n_shared_experts}: inference/model.py MoE asserts this "
            "directly and builds a single, unrouted shared Expert per layer"
        )
    if n_routed_experts < 1:
        raise ValueError(
            f"n_routed_experts must be positive, got {n_routed_experts}"
        )
    if not 1 <= n_activated_experts <= n_routed_experts:
        raise ValueError(
            f"num_experts_per_tok={n_activated_experts} is outside "
            f"[1, n_routed_experts={n_routed_experts}]"
        )
    if config.hidden_size < 1 or config.moe_intermediate_size < 1:
        raise ValueError(
            f"hidden_size={config.hidden_size} and "
            f"moe_intermediate_size={config.moe_intermediate_size} must both be "
            "positive"
        )
    if config.swiglu_limit < 0:
        raise ValueError(
            f"text_config.swiglu_limit must be non-negative, got "
            f"{config.swiglu_limit} (0 disables the clamp)"
        )


class DeepseekV41PackedLinear(nn.Module):
    """A bias-free Linear whose weight stays packed at rest.

    ``quant="fp4"`` stores ``[out, in // 2]`` uint8 (two E2M1 codes per byte)
    plus an ``[out, in // 32]`` E8M0 scale; ``quant="fp8"`` stores ``[out, in]``
    uint8 E4M3 plus a 2-D-tiled ``[ceil(out/32), in // 32]`` E8M0 scale;
    ``quant=None`` is a plain dense weight.

    The packed bytes are decoded inside ``__call__`` and the decoded tensor is
    deliberately not retained anywhere: with 384 routed experts per layer,
    persistently unpacking them would multiply expert storage by 4x (FP4) and
    defeat the entire at-rest format. Only the experts a token actually routes
    to are ever decoded, and only for the duration of that call.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant: Optional[str] = "fp4",
        dtype=mx.bfloat16,
    ):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError(
                f"in_features={in_features} and out_features={out_features} must "
                "both be positive"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.quant = quant
        if quant == "fp4":
            if in_features % FP4_WEIGHT_BLOCK_SIZE:
                raise ValueError(
                    f"in_features={in_features} is not divisible by the FP4 weight "
                    f"block size {FP4_WEIGHT_BLOCK_SIZE}; fp4_gemm quantizes 1x32 "
                    "along the reduction axis and never pads a partial tail"
                )
            self.weight = mx.zeros(
                (out_features, in_features // 2), dtype=mx.uint8
            )
            self.scale = mx.zeros(
                (out_features, in_features // FP4_WEIGHT_BLOCK_SIZE), dtype=mx.uint8
            )
        elif quant == "fp8":
            if in_features % FP8_WEIGHT_BLOCK_SIZE:
                raise ValueError(
                    f"in_features={in_features} is not divisible by the FP8 weight "
                    f"block size {FP8_WEIGHT_BLOCK_SIZE}; only the out axis may "
                    "carry a partial tail block"
                )
            self.weight = mx.zeros((out_features, in_features), dtype=mx.uint8)
            self.scale = mx.zeros(
                (
                    -(-out_features // FP8_WEIGHT_BLOCK_SIZE),
                    in_features // FP8_WEIGHT_BLOCK_SIZE,
                ),
                dtype=mx.uint8,
            )
        elif quant is None:
            self.weight = mx.zeros((out_features, in_features), dtype=dtype)
        else:
            raise ValueError(
                f"unsupported quant {quant!r}; expected 'fp4', 'fp8' or None"
            )

    def __iter__(self):
        # Immediate public leaves so dict(module) inspects packed state, not
        # a transient decode. mlx.nn.Module is not a mapping on every runtime.
        return iter(self.parameters().items())

    def dequantized(self) -> mx.array:
        """Decode the packed weight to a dense float32 tensor.

        Intentionally a method and not a cached property: the result must stay
        transient (see the class docstring).
        """
        if self.quant == "fp4":
            return dequantize_fp4_block(
                self.weight, self.scale, FP4_WEIGHT_BLOCK_SIZE, mx.float32
            )
        if self.quant == "fp8":
            return dequantize_fp8_block(
                self.weight, self.scale, FP8_WEIGHT_BLOCK_SIZE, mx.float32
            )
        return self.weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected a trailing dimension of {self.in_features}, got shape "
                f"{x.shape}"
            )
        return x.astype(mx.float32) @ self.dequantized().T


def _make_fp8_linear(in_features: int, out_features: int) -> DeepseekV41PackedLinear:
    quant = "fp8" if in_features % FP8_WEIGHT_BLOCK_SIZE == 0 else None
    return DeepseekV41PackedLinear(in_features, out_features, quant=quant)


class DeepseekV41Expert(nn.Module):
    """One SwiGLU FFN expert (inference/model.py Expert).

    The clamps come straight from training, where they keep fp8/fp4
    activations in range: the up branch is clamped on both sides, the gate
    branch only from above. The optional ``weights`` argument applies the
    routing weight *before* the down projection, not after -- which matters
    because w2 is quantized, so scaling after it would quantize a differently
    scaled activation.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        quant: Optional[str] = "fp4",
        swiglu_limit: float = 0.0,
        dtype=mx.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        self.swiglu_limit = float(swiglu_limit)
        self.w1 = DeepseekV41PackedLinear(dim, inter_dim, quant, dtype)
        self.w2 = DeepseekV41PackedLinear(inter_dim, dim, quant, dtype)
        self.w3 = DeepseekV41PackedLinear(dim, inter_dim, quant, dtype)

    def __call__(self, x: mx.array, weights: Optional[mx.array] = None) -> mx.array:
        dtype = x.dtype
        gate = self.w1(x).astype(mx.float32)
        up = self.w3(x).astype(mx.float32)
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        h = nn.silu(gate) * up
        if weights is not None:
            h = weights.astype(mx.float32) * h
        return self.w2(h.astype(dtype))


class DeepseekV41Gate(nn.Module):
    """MoE gating (inference/model.py Gate).

    ``weight`` is dense float32 (the reference builds it outside its fp8
    ``set_dtype`` scope and casts to float for the GEMM), ``bias`` is the
    routing correction bias, and ``bias_vl`` -- present only when the vision
    tower is enabled -- replaces it for tokens inside an image span.
    """

    def __init__(
        self,
        config: TextConfig,
        n_routed_experts: Optional[int] = None,
        n_activated_experts: Optional[int] = None,
        vision_enabled: bool = False,
    ):
        super().__init__()
        n_routed = (
            config.n_routed_experts if n_routed_experts is None else n_routed_experts
        )
        topk = (
            config.num_experts_per_tok
            if n_activated_experts is None
            else n_activated_experts
        )
        validate_moe_routing_config(config, n_routed, topk)
        self.dim = config.hidden_size
        self.n_routed_experts = n_routed
        self.topk = topk
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = bool(config.norm_topk_prob)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.gate_temp = GATE_TEMP
        self.vision_enabled = bool(vision_enabled)
        self.weight = mx.zeros((n_routed, config.hidden_size), dtype=mx.float32)
        self.bias = mx.zeros((n_routed,), dtype=mx.float32)
        if self.vision_enabled:
            self.bias_vl = mx.zeros((n_routed,), dtype=mx.float32)

    def __call__(
        self, x: mx.array, image_mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"expected a trailing dimension of {self.dim}, got shape {x.shape}"
            )
        logits = (x.astype(mx.float32) @ self.weight.astype(mx.float32).T)
        if self.gate_temp != 1.0:
            logits = logits / self.gate_temp
        scores = routing_scores(logits, self.scoring_func)
        bias = self.bias.astype(mx.float32)
        if image_mask is not None:
            if not self.vision_enabled:
                raise ValueError(
                    "image_mask was supplied but this gate has no bias_vl: "
                    "inference/model.py only allocates the VL routing bias when "
                    "the vision tower is enabled, so a VL mask against a "
                    "text-only gate is a wiring error, not a no-op"
                )
            if tuple(image_mask.shape) != tuple(x.shape[:-1]):
                raise ValueError(
                    f"image_mask shape {image_mask.shape} does not match the token "
                    f"axes {tuple(x.shape[:-1])}"
                )
            bias = mx.where(
                mx.expand_dims(image_mask, -1),
                self.bias_vl.astype(mx.float32),
                bias,
            )
        return noaux_tc_route(
            scores,
            bias,
            self.topk,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )


class DeepseekV41MoE(nn.Module):
    """Top-k routed experts plus the one shared expert every token goes through
    (inference/model.py MoE).

    Routed experts are split across ranks, so only this rank's contiguous
    ``[experts_start_idx, experts_end_idx)`` block is constructed at all -- a
    rank never allocates, let alone unpacks, an expert another rank owns.
    Combining the per-rank partial sums uses the injected cross-rank all-reduce
    before the replicated shared expert is added. A sharded instance without a
    collective still fails loud rather than returning an incomplete sum.
    """

    def __init__(
        self,
        config: TextConfig,
        n_routed_experts: Optional[int] = None,
        n_activated_experts: Optional[int] = None,
        expert_quant: Optional[str] = "fp4",
        shared_expert_quant: Optional[str] = "fp8",
        world_size: int = 1,
        rank: int = 0,
        all_reduce: Optional[Callable[[mx.array], mx.array]] = None,
        vision_enabled: bool = False,
        dtype=mx.bfloat16,
    ):
        super().__init__()
        n_routed = (
            config.n_routed_experts if n_routed_experts is None else n_routed_experts
        )
        topk = (
            config.num_experts_per_tok
            if n_activated_experts is None
            else n_activated_experts
        )
        self.gate = DeepseekV41Gate(
            config,
            n_routed_experts=n_routed,
            n_activated_experts=topk,
            vision_enabled=vision_enabled,
        )
        self.dim = config.hidden_size
        self.inter_dim = config.moe_intermediate_size
        self.n_routed_experts = n_routed
        self.topk = topk
        self.world_size = world_size
        self.rank = rank
        self.all_reduce = all_reduce
        start, end = routed_expert_partition(n_routed, world_size, rank)
        self.experts_start_idx = start
        self.experts_end_idx = end
        self.experts = [None] * n_routed
        for expert_id in range(start, end):
            self.experts[expert_id] = DeepseekV41Expert(
                self.dim,
                self.inter_dim,
                expert_quant,
                config.swiglu_limit,
                dtype,
            )
        self.shared_experts = DeepseekV41Expert(
            self.dim,
            self.inter_dim,
            shared_expert_quant,
            config.swiglu_limit,
            dtype,
        )

    @property
    def n_local_experts(self) -> int:
        return self.experts_end_idx - self.experts_start_idx

    @property
    def local_expert_ids(self) -> List[int]:
        """The global routed-expert ids this rank owns."""
        return list(range(self.experts_start_idx, self.experts_end_idx))

    def expert(self, expert_id: int) -> DeepseekV41Expert:
        """Look an expert up by its *global* id, refusing one another rank owns."""
        if not self.experts_start_idx <= expert_id < self.experts_end_idx:
            raise KeyError(
                f"expert {expert_id} is not local to rank {self.rank}, which owns "
                f"[{self.experts_start_idx}, {self.experts_end_idx})"
            )
        return self.experts[expert_id]

    def shard(self, group) -> None:
        """Retain this rank's globally numbered experts and bind their sum."""
        world_size = group.size()
        rank = group.rank()
        start, end = routed_expert_partition(
            self.n_routed_experts, world_size, rank
        )
        if self.world_size != 1 and (
            self.world_size != world_size or self.rank != rank
        ):
            raise ValueError(
                f"MoE is already sharded as rank {self.rank}/{self.world_size}; "
                f"cannot reshard it as rank {rank}/{world_size}"
            )
        for expert_id in range(self.n_routed_experts):
            if not start <= expert_id < end:
                self.experts[expert_id] = None
        self.world_size = world_size
        self.rank = rank
        self.experts_start_idx = start
        self.experts_end_idx = end
        self.all_reduce = (
            None
            if world_size == 1
            else lambda value: mx.distributed.all_sum(value, group=group)
        )

    def __call__(
        self, x: mx.array, image_mask: Optional[mx.array] = None
    ) -> mx.array:
        if self.world_size > 1 and self.all_reduce is None:
            raise RuntimeError(
                "DeepseekV41MoE is sharded but has no all_reduce; rank-local "
                "routed outputs must be summed before the shared expert is added"
            )
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"expected a trailing dimension of {self.dim}, got shape {x.shape}"
            )
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        mask = None if image_mask is None else image_mask.reshape(-1)
        weights, indices = self.gate(flat, mask)

        # One host sync per call, exactly as the reference does via
        # bincount(...).tolist(): the routed set is data-dependent, and a dense
        # all-expert formulation would have to unpack all 384 experts per layer.
        buckets: Dict[int, List[Tuple[int, int]]] = {}
        for token, row in enumerate(indices.tolist()):
            for slot, expert_id in enumerate(row):
                buckets.setdefault(expert_id, []).append(
                    (token, token * self.topk + slot)
                )

        y = mx.zeros(flat.shape, dtype=mx.float32)
        flat_weights = weights.reshape(-1)
        for expert_id in sorted(buckets):
            if not self.experts_start_idx <= expert_id < self.experts_end_idx:
                continue
            pairs = buckets[expert_id]
            # A token selects any given expert at most once (top-k indices are
            # distinct), so these row indices are unique and the read-modify-write
            # below cannot drop a contribution.
            tokens = mx.array([t for t, _ in pairs], dtype=mx.int32)
            slots = mx.array([s for _, s in pairs], dtype=mx.int32)
            out = self.experts[expert_id](
                mx.take(flat, tokens, axis=0),
                mx.take(flat_weights, slots).reshape(-1, 1),
            )
            y[tokens] = y[tokens] + out.astype(mx.float32)

        if self.world_size > 1:
            y = self.all_reduce(y)
            if y is None:
                raise RuntimeError(
                    "the injected MoE all_reduce returned None; it must return "
                    "the summed routed output"
                )
        y = y + self.shared_experts(flat).astype(mx.float32)
        return y.astype(x.dtype).reshape(shape)


# --------------------------------------------------------------------------- #
# DSpark / MTP isolated forward_spec path                                      #
# --------------------------------------------------------------------------- #


def get_dspark_topk_idxs(
    window_size: int, batch: int, block_size: int, start_pos: int
) -> mx.array:
    """Indices used by the official DSpark attention over main and draft KV."""
    if start_pos <= 0:
        raise ValueError(f"DSpark decode requires start_pos > 0, got {start_pos}")
    if window_size < 1 or batch < 1 or block_size < 1:
        raise ValueError(
            "window_size, batch and block_size must be positive, got "
            f"{window_size}, {batch}, {block_size}"
        )
    main = mx.arange(min(window_size, start_pos + 1), dtype=mx.int32)
    draft = window_size + mx.arange(block_size, dtype=mx.int32)
    row = mx.concatenate([main, draft])
    return mx.broadcast_to(row.reshape(1, 1, -1), (batch, block_size, row.size))


class DeepseekV41DSparkEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = mx.zeros((vocab_size, dim), dtype=mx.bfloat16)

    def __call__(self, token_ids: mx.array) -> mx.array:
        ids = np.asarray(token_ids)
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise ValueError(f"token id is outside [0, {self.vocab_size})")
        return mx.take(self.weight, token_ids.astype(mx.int32), axis=0)


class DeepseekV41DSparkHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = mx.zeros((vocab_size, dim), dtype=mx.float32)

    def __call__(self, x: mx.array, full_logits: bool = False) -> mx.array:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"head expected a trailing dimension of {self.dim}, got {x.shape}"
            )
        if not full_logits:
            x = x[:, -1]
        return x.astype(mx.float32) @ self.weight.T


class DeepseekV41DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.embed = DeepseekV41DSparkEmbedding(vocab_size, markov_rank)
        self.head = DeepseekV41DSparkHead(vocab_size, markov_rank)

    def __call__(self, token_ids: mx.array) -> Tuple[mx.array, mx.array]:
        embed = self.embed(token_ids)
        return self.head(embed, full_logits=True), embed


class DeepseekV41DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = DeepseekV41PackedLinear(
            input_dim, 1, quant=None, dtype=mx.float32
        )

    def __call__(self, hidden: mx.array, markov_embed: mx.array) -> mx.array:
        if tuple(hidden.shape[:-1]) != tuple(markov_embed.shape[:-1]):
            raise ValueError(
                f"hidden axes {hidden.shape[:-1]} do not match Markov axes "
                f"{markov_embed.shape[:-1]}"
            )
        return self.proj(
            mx.concatenate(
                [hidden.astype(mx.float32), markov_embed.astype(mx.float32)], axis=-1
            )
        ).squeeze(-1)


class DeepseekV41DSparkAttentionCache:
    """One DSpark stage's main-model sliding-window KV state."""

    def __init__(self, window_size: int, head_dim: int):
        self.window_size = window_size
        self.head_dim = head_dim
        self.window: Optional[mx.array] = None
        self.offset = 0

    def update_main(self, kv: mx.array, start_pos: int) -> mx.array:
        batch, seqlen, dim = kv.shape
        if dim != self.head_dim:
            raise ValueError(f"DSpark cache expected head_dim={self.head_dim}, got {dim}")
        if self.window is None or self.window.shape[0] != batch:
            if start_pos != 0:
                raise ValueError("DSpark decode requires a preceding prefill")
            self.window = mx.zeros(
                (batch, self.window_size, self.head_dim), dtype=kv.dtype
            )
        if start_pos == 0:
            if seqlen <= self.window_size:
                self.window[:, :seqlen] = kv
            else:
                cutoff = seqlen % self.window_size
                tail = kv[:, -self.window_size :]
                self.window[:, cutoff:] = tail[:, : self.window_size - cutoff]
                if cutoff:
                    self.window[:, :cutoff] = tail[:, self.window_size - cutoff :]
            self.offset = seqlen
            return kv
        if start_pos != self.offset or seqlen != 1:
            raise ValueError(
                f"DSpark cache expected one main token at position {self.offset}, "
                f"got start_pos={start_pos}, seqlen={seqlen}"
            )
        self.window[:, start_pos % self.window_size] = kv[:, 0]
        self.offset = start_pos + 1
        return self.window


class DeepseekV41DSparkAttention(DeepseekV41Attention):
    """DSpark attention: draft queries plus main-model and draft KV."""

    def __init__(self, config: TextConfig, layer_id: int):
        if config.compress_ratios[layer_id] != 0:
            raise ValueError(
                f"DSpark layer {layer_id} must have compress_ratio=0, got "
                f"{config.compress_ratios[layer_id]}"
            )
        policy = AttentionLayerPolicy(
            layer_id, 0, False, False, False, False, False, None, None, None
        )
        super().__init__(config, policy)

    def __call__(
        self,
        x: mx.array,
        main_x: mx.array,
        start_pos: int,
        cache: DeepseekV41DSparkAttentionCache,
    ) -> mx.array:
        if main_x.ndim != 3 or main_x.shape[-1] != self.wkv.weight.shape[1]:
            raise ValueError(f"malformed DSpark main hidden state {main_x.shape}")
        main_positions = mx.arange(
            start_pos, start_pos + main_x.shape[1], dtype=mx.int32
        )
        main_cos, main_sin = self.rope(main_positions)
        main_kv = self.kv_norm(self.wkv(main_x))
        main_kv = apply_rope_tail(
            main_kv, main_cos, main_sin, self.rope_head_dim
        )
        main_kv = act_quant_roundtrip(main_kv, FP8_ACT_BLOCK_SIZE)
        window_kv = cache.update_main(main_kv, start_pos)
        if start_pos == 0:
            return x

        if x.ndim != 3:
            raise ValueError(f"DSpark draft stream must be [batch, block, dim], got {x.shape}")
        batch, block_size, _ = x.shape
        draft_positions = mx.arange(
            start_pos + main_x.shape[1],
            start_pos + main_x.shape[1] + block_size,
            dtype=mx.int32,
        )
        cos, sin = self.rope(draft_positions)
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(batch, block_size, self.n_heads, self.head_dim)
        q = apply_rope_tail(q, cos, sin, self.rope_head_dim)
        draft_kv = self.kv_norm(self.wkv(x))
        draft_kv = apply_rope_tail(draft_kv, cos, sin, self.rope_head_dim)
        draft_kv = act_quant_roundtrip(draft_kv, FP8_ACT_BLOCK_SIZE)
        kv = mx.concatenate([window_kv, draft_kv], axis=1)
        idxs = get_dspark_topk_idxs(
            self.window_size, batch, block_size, start_pos
        )
        out = sparse_attn(q, kv, self.attn_sink, idxs, self.softmax_scale)
        out = apply_rope_tail(out, cos, sin, self.rope_head_dim, inverse=True)
        wo_a = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)
        out = mx.einsum(
            "bsgd,grd->bsgr",
            out.reshape(batch, block_size, self.n_groups, -1),
            wo_a.astype(out.dtype),
        )
        return self.wo_b(out.reshape(batch, block_size, -1))


def _sample_dspark(logits: mx.array, temperature: float) -> mx.array:
    if temperature == 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    return mx.random.categorical(logits / max(temperature, 1e-5)).astype(mx.int32)


class DeepseekV41DSparkBlock(DeepseekV41HyperConnections):
    """One official DSpark stage stored under the ``mtp.*`` namespace."""

    def __init__(
        self,
        config: TextConfig,
        layer_id: int,
        temperature: float = 1.0,
        vision_enabled: bool = False,
    ):
        super().__init__(config)
        self.layer_id = layer_id
        self.stage_id = layer_id - config.num_hidden_layers
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id
        self.temperature = float(temperature)
        self.attn = DeepseekV41DSparkAttention(config, layer_id)
        self.ffn = DeepseekV41MoE(
            config,
            n_routed_experts=config.dspark_n_routed_experts,
            n_activated_experts=config.dspark_num_experts_per_tok,
            vision_enabled=vision_enabled,
        )
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cache = DeepseekV41DSparkAttentionCache(
            config.sliding_window, config.head_dim
        )
        if self.stage_id == 0:
            self.main_proj = DeepseekV41PackedLinear(
                config.hidden_size * len(config.dspark_target_layer_ids),
                config.hidden_size,
                quant="fp8",
            )
            self.main_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.stage_id == config.num_nextn_predict_layers - 1:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.markov_head = DeepseekV41DSparkMarkovHead(
                config.vocab_size, config.dspark_markov_rank
            )
            self.confidence_head = DeepseekV41DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank
            )

    def __call__(self, x, start_pos, pre_mix, main_x):
        if start_pos == 0:
            self.attn(x, main_x, start_pos, self.cache)
            return x, pre_mix
        x, attn_pre = self.sublayer_step(
            x,
            pre_mix,
            lambda value: self.attn(
                self.attn_norm(value), main_x, start_pos, self.cache
            ),
            "attn",
        )
        return self.sublayer_step(
            x, attn_pre, lambda value: self.ffn(self.ffn_norm(value)), "ffn"
        )

    def forward_embed(self, main_hidden, input_ids, embed):
        if self.stage_id != 0:
            raise ValueError("forward_embed belongs to DSpark stage 0")
        if input_ids.ndim != 1:
            raise ValueError(f"DSpark input_ids must be [batch], got {input_ids.shape}")
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_ids = mx.full(
            (input_ids.shape[0], self.block_size),
            self.noise_token_id,
            dtype=mx.int32,
        )
        draft_ids[:, 0] = input_ids.astype(mx.int32)
        x = expand_hyper_connection_stream(embed(draft_ids), self.hc_mult)
        return x, main_x

    def forward_head(self, x, pre_mix, input_ids, head):
        if not hasattr(self, "markov_head"):
            raise ValueError("forward_head belongs to the final DSpark stage")
        hidden = hc_pre(x, pre_mix)
        logits = head(self.norm(hidden), full_logits=True)
        output_ids = mx.zeros(
            (input_ids.shape[0], self.block_size + 1), dtype=mx.int32
        )
        output_ids[:, 0] = input_ids.astype(mx.int32)
        markov_embeds = []
        for index in range(self.block_size):
            logits_bias, markov_embed = self.markov_head(output_ids[:, index])
            logits[:, index] = logits[:, index] + logits_bias
            markov_embeds.append(markov_embed)
            output_ids[:, index + 1] = _sample_dspark(
                logits[:, index], self.temperature
            )
        markov_embed = mx.stack(markov_embeds, axis=1)
        confidence = self.confidence_head(hidden, markov_embed)
        return output_ids, logits, confidence


class DeepseekV41DSpark(nn.Module):
    """Isolated official ``forward_spec`` path; no speculative driver."""

    def __init__(
        self,
        config: TextConfig,
        temperature: float = 1.0,
        vision_enabled: bool = False,
    ):
        super().__init__()
        if config.num_nextn_predict_layers < 1 or config.dspark_block_size < 1:
            raise ValueError("DSpark requires positive MTP layer and block counts")
        if len(config.dspark_target_layer_ids) < 1:
            raise ValueError("DSpark requires at least one target backbone layer")
        if len(set(config.dspark_target_layer_ids)) != len(config.dspark_target_layer_ids):
            raise ValueError("DSpark target layer ids must be unique")
        if any(
            layer_id < 0 or layer_id >= config.num_hidden_layers
            for layer_id in config.dspark_target_layer_ids
        ):
            raise ValueError("DSpark target layer id is outside the backbone")
        mtp_ratios = config.compress_ratios[config.num_hidden_layers :]
        if mtp_ratios != [0] * config.num_nextn_predict_layers:
            raise ValueError(f"DSpark MTP compress ratios must all be zero, got {mtp_ratios}")
        if not 0 <= config.dspark_noise_token_id < config.vocab_size:
            raise ValueError("DSpark noise token id is outside the vocabulary")
        self.config = config
        self.hc_mult = config.hc_mult
        self.embed = DeepseekV41DSparkEmbedding(config.vocab_size, config.hidden_size)
        self.head = DeepseekV41DSparkHead(config.vocab_size, config.hidden_size)
        self.mtp = [
            DeepseekV41DSparkBlock(
                config,
                config.num_hidden_layers + stage_id,
                temperature,
                vision_enabled,
            )
            for stage_id in range(config.num_nextn_predict_layers)
        ]

    def forward_spec(self, input_ids, main_hidden, start_pos: int = 0):
        expected_width = self.config.hidden_size * len(
            self.config.dspark_target_layer_ids
        )
        if main_hidden.ndim != 3 or main_hidden.shape[-1] != expected_width:
            raise ValueError(
                f"main_hidden must be [batch, seqlen, {expected_width}], got "
                f"{main_hidden.shape}"
            )
        x, main_x = self.mtp[0].forward_embed(main_hidden, input_ids, self.embed)
        pre_mix = make_identity_pre_mix(x.shape[0], x.shape[1], self.hc_mult)
        for layer in self.mtp:
            x, pre_mix = layer(x, start_pos, pre_mix, main_x)
        if start_pos == 0:
            return None
        return self.mtp[-1].forward_head(x, pre_mix, input_ids, self.head)


# --------------------------------------------------------------------------- #
# Engram: sparse n-gram hash addressing                                        #
#                                                                              #
# Faithful to inference/engram.py (build_compressed_token_map,                 #
# find_next_prime, compute_hash_multipliers, EngramLayout, NgramHashState) and #
# inference/model.py (ParallelEngramEmbedding, Engram) at the pinned revision  #
# deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277.    #
#                                                                              #
# The load-bearing property of this whole section is that NOTHING here ever    #
# materializes an Engram table. The two official tables are 384,006,168 x 256  #
# and 384,016,682 x 256 float8_e4m3fn rows plus their E8M0 block scales --     #
# ~91.6 and ~98.3 GiB respectively, ~101 GB of raw safetensors payload across  #
# the shards that hold them. Rows are therefore addressed individually against #
# a file-backed row store (EngramRowStore) through a bounded LRU              #
# (BoundedEngramRowCache), and no surface in this module exposes, returns or   #
# accepts a dense [num_embeddings, head_dim] tensor. Peak memory of a lookup   #
# is a function of the number of *unique* rows requested plus the cache bound, #
# never of the table size.                                                     #
# --------------------------------------------------------------------------- #

# ParallelEngramEmbedding uses the global fp8_block_size, which the pinned
# config.json fixes at 32 (quantization_config.weight_block_size == [32, 32]).
# Unlike a 2-D-tiled Linear weight, an Engram scale row is 1-D: one E8M0 code
# per 32 contiguous columns of that row, i.e. head_dim // 32 == 8 codes/row.
ENGRAM_FP8_BLOCK_SIZE = 32

# Engram.clamp_value: the floor applied to |dot| before the signed sqrt, so the
# sqrt never sees exactly zero (its derivative is unbounded there).
ENGRAM_GATE_CLAMP = 1e-6

# compute_hash_multipliers seeds one numpy PCG64 stream per Engram layer as
# np.random.default_rng(10007 * layer_id), so two layers never hash alike.
ENGRAM_RNG_LAYER_SEED_STRIDE = 10007

# NgramHashState.DEAD: the compressed-id sentinel written into the history
# cache for a token that may not take part in an n-gram (an image span).
# Look-back stops at a DEAD token, so an n-gram never spans one.
ENGRAM_DEAD_TOKEN = -1

# build_compressed_token_map normalizer chain, in the exact order the official
# tokenizers.normalizers.Sequence applies it. Kept as data so the contract is
# inspectable and testable, not just implied by the code below.
ENGRAM_NORMALIZER_SEQUENCE = (
    "NFKC",
    "NFD",
    "StripAccents",
    "Lowercase",
    "Replace(Regex(r'[ \\t\\r\\n]+'), ' ')",
    "Replace(Regex(r'^ $'), sentinel)",
    "Strip",
    "Replace(sentinel, ' ')",
)

# A Unicode private-use character. A token that is exactly one space is swapped
# to this before Strip() so it survives instead of collapsing to the empty
# string and merging with unrelated tokens; it is swapped back afterwards.
ENGRAM_SPACE_SENTINEL = "\ue000"

# The Unicode replacement character. A token whose decoded form contains it is
# a partial UTF-8 byte token: there is nothing to normalize, so it is keyed by
# its raw (id_to_token) form instead.
_UNICODE_REPLACEMENT_CHAR = "\ufffd"

_ENGRAM_NORMALIZER = tokenizer_normalizers.Sequence(
    [
        tokenizer_normalizers.NFKC(),
        tokenizer_normalizers.NFD(),
        tokenizer_normalizers.StripAccents(),
        tokenizer_normalizers.Lowercase(),
        tokenizer_normalizers.Replace(TokenizerRegex(r"[ \t\r\n]+"), " "),
        tokenizer_normalizers.Replace(TokenizerRegex(r"^ $"), ENGRAM_SPACE_SENTINEL),
        tokenizer_normalizers.Strip(),
        tokenizer_normalizers.Replace(ENGRAM_SPACE_SENTINEL, " "),
    ]
)


def normalize_engram_token_text(text: str) -> str:
    """Apply the exact official Engram normalizer chain to one token text.

    This is the half of the compressed-vocabulary contract that model code
    owns: given the text a tokenizer decoded for a single id, produce the key
    that decides which ids collapse together. Every hash multiplier is derived
    from the resulting compressed vocab size, so a divergence here does not
    degrade quality gracefully -- it silently rehashes the entire table.

    Faithful to ENGRAM_NORMALIZER_SEQUENCE: NFKC, then NFD, then drop every
    non-spacing mark (StripAccents), then lowercase, then fold every run of
    space/tab/CR/LF to a single space, then protect a lone space with
    ENGRAM_SPACE_SENTINEL, then trim Unicode whitespace, then restore the
    sentinel. So " The", "the" and "THE" all produce the same key.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected a str token text, got {type(text).__name__}")
    return _ENGRAM_NORMALIZER.normalize_str(text)


def engram_compressed_token_key(decoded_text: str, raw_token: Optional[str]) -> str:
    """The compressed-vocabulary key for one token id.

    A decoded text containing U+FFFD is a partial UTF-8 byte token and is keyed
    by its raw form; otherwise the normalized text is used, falling back to the
    unnormalized text when normalization empties it (so a whitespace-only token
    keeps an identity of its own instead of merging into one bucket).
    """
    if _UNICODE_REPLACEMENT_CHAR in decoded_text:
        if raw_token is None:
            raise ValueError(
                "a token whose decoded text contains U+FFFD must supply its raw "
                "id_to_token form: it is a partial UTF-8 byte token and is keyed "
                "by that raw form, not by normalized text"
            )
        return raw_token
    normalized = normalize_engram_token_text(decoded_text)
    return normalized if normalized else decoded_text


def build_engram_compressed_token_map(
    decoded_texts: Sequence[str], raw_tokens: Sequence[Optional[str]]
) -> Tuple[List[int], int]:
    """Map every raw token id onto the smaller Engram id space.

    Returns ``(lookup, compressed_vocab_size)`` where ``lookup[token_id]`` is the
    compressed id. Compressed ids are handed out in first-seen token-id order,
    exactly as the official dict insertion does, so the mapping is fully
    deterministic given the tokenizer.

    Takes already-decoded text rather than a tokenizer so the normalization
    contract -- the part this repository owns -- is testable without a
    tokenizer backend. See build_engram_compressed_token_map_from_tokenizer for
    the driver that reads those two lists off a real fast tokenizer.
    """
    decoded = list(decoded_texts)
    raw = list(raw_tokens)
    if len(decoded) != len(raw):
        raise ValueError(
            f"decoded_texts ({len(decoded)}) and raw_tokens ({len(raw)}) must "
            "describe the same vocabulary, one entry per token id"
        )
    if not decoded:
        raise ValueError("an empty vocabulary has no Engram compressed map")
    key_to_new: Dict[str, int] = {}
    lookup = [0] * len(decoded)
    for token_id, (text, raw_token) in enumerate(zip(decoded, raw)):
        key = engram_compressed_token_key(text, raw_token)
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def build_engram_compressed_token_map_from_tokenizer(
    tokenizer,
    expected_size: Optional[int] = None,
    fallback_token_id: Optional[int] = None,
) -> Tuple[List[int], int]:
    """build_engram_compressed_token_map driven by a real fast tokenizer.

    Uses the raw Rust backend directly, matching what training decoded with
    (no clean_up_tokenization_spaces, no skip_special_tokens).
    """
    tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError(
            "the Engram compressed vocabulary is built from the raw fast-tokenizer "
            "backend (tokenizer.backend_tokenizer); a slow tokenizer decodes "
            "differently and would produce a different compressed vocab size, "
            "which silently rehashes both Engram tables"
        )
    size = len(tokenizer)
    decoded, raw = [], []
    for token_id in range(size):
        decoded.append(backend.decode([token_id], skip_special_tokens=False))
        raw.append(backend.id_to_token(token_id))
    lookup, compressed_size = build_engram_compressed_token_map(decoded, raw)
    if expected_size is None or compressed_size == expected_size:
        return lookup, compressed_size

    if compressed_size < expected_size or fallback_token_id is None:
        raise ValueError(
            f"tokenizer produces {compressed_size} compressed Engram tokens, "
            f"but the checkpoint config requires {expected_size}"
        )
    if not 0 <= fallback_token_id < len(lookup):
        raise ValueError(
            f"Engram fallback token id {fallback_token_id} is outside tokenizer "
            f"vocabulary size {len(lookup)}"
        )

    overflow_ids = [
        token_id
        for token_id, compressed_id in enumerate(lookup)
        if compressed_id >= expected_size
    ]
    first_overflow = overflow_ids[0] if overflow_ids else len(lookup)
    placeholder = re.compile(r"<\|place_holder_mm_span_\d{4}\|>")
    multimodal_controls = {
        "<｜rl_image_pad｜>",
        "<｜rl_image_start｜>",
        "<｜deepseek_image｜>",
        "<｜/polygon｜>",
        "<｜polygon｜>",
        "<｜/point｜>",
        "<｜point｜>",
        "<｜/box｜>",
        "<｜box｜>",
        "<｜/ref｜>",
        "<｜ref｜>",
    }
    if overflow_ids != list(range(first_overflow, len(lookup))) or any(
        raw[token_id] != decoded[token_id]
        or (
            placeholder.fullmatch(raw[token_id] or "") is None
            and raw[token_id] not in multimodal_controls
        )
        for token_id in overflow_ids
    ):
        raise ValueError(
            f"tokenizer produces {compressed_size} compressed Engram tokens, "
            f"but the checkpoint config requires {expected_size}; only a contiguous "
            "suffix of multimodal placeholder tokens may extend the tokenizer after "
            "Engram training"
        )

    fallback_id = lookup[fallback_token_id]
    for token_id in overflow_ids:
        lookup[token_id] = fallback_id
    return lookup, expected_size


def _is_prime(candidate: int) -> bool:
    """Deterministic trial-division primality test.

    The official layout draws primes just above engram_vocab_size (16,000,000),
    so the loop below runs to ~4,000 per candidate: exact, dependency-free and
    far cheaper than taking a sympy dependency for this one call.
    """
    if candidate < 2:
        return False
    if candidate < 4:
        return True
    if candidate % 2 == 0:
        return False
    factor = 3
    while factor * factor <= candidate:
        if candidate % factor == 0:
            return False
        factor += 2
    return True


def find_next_prime(start: int, seen_primes: Iterable[int]) -> int:
    """The smallest prime strictly above start that has not been handed out yet.

    Faithful to inference/engram.py find_next_prime. Drawing in strictly
    increasing order and never reusing a prime is what keeps every
    (n-gram size, head) bucket range disjoint inside one table.
    """
    seen = (
        seen_primes if isinstance(seen_primes, (set, frozenset)) else set(seen_primes)
    )
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def compute_engram_hash_multipliers(
    layer_ids: Sequence[int], max_ngram_size: int, compressed_vocab_size: int
) -> np.ndarray:
    """One odd int64 multiplier per (Engram layer, look-back).

    Faithful to inference/engram.py compute_hash_multipliers: a separate numpy
    PCG64 stream seeded ENGRAM_RNG_LAYER_SEED_STRIDE * layer_id per layer,
    values drawn in [0, bound) and mapped to 2 * v + 1 so every multiplier is
    odd. bound is chosen so compressed_id * multiplier cannot overflow int64 --
    the running hash is an XOR of such products, so a wrap would not merely be
    inexact, it would alias unrelated n-grams onto the same row.

    Derived from the *compressed* vocab size, not the raw one, which is why
    build_engram_compressed_token_map must reproduce the official normalizer
    chain exactly.
    """
    if max_ngram_size < 1:
        raise ValueError(f"engram_max_ngram_size must be >= 1, got {max_ngram_size}")
    if compressed_vocab_size < 1:
        raise ValueError(
            f"engram_compressed_vocab_size must be >= 1, got {compressed_vocab_size}"
        )
    max_long = int(np.iinfo(np.int64).max)
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(ENGRAM_RNG_LAYER_SEED_STRIDE * int(layer_id))
        values = generator.integers(
            low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64
        )
        rows.append(values * 2 + 1)
    return np.stack(rows).astype(np.int64)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables (inference/engram.py EngramLayout).

    A position is hashed as max_ngram_size - 1 n-grams (2-gram up to
    max_ngram_size-gram), each split over n_heads heads. Every
    (n-gram size, head) pair owns its own prime-sized bucket range inside that
    layer table; primes are drawn in strictly increasing order and never
    reused, so the ranges are disjoint and their concatenation tiles the table.

    For the pinned official config this tiling is exact, not approximate: the
    24 primes of Engram layer 1 sum to 384,006,168 and the 24 primes of layer
    14 sum to 384,016,682 -- precisely the two engram_num_embeddings counts.
    """

    max_ngram_size: int
    layer_ids: Tuple[int, ...]
    num_embeddings: Tuple[int, ...]
    primes: Tuple[Tuple[Tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @property
    def n_hash_cols(self) -> int:
        """Hash ids per token per Engram layer: 24 for the official config."""
        return (self.max_ngram_size - 1) * self.n_heads

    def flat_primes(self, layer_hash_index: int) -> Tuple[int, ...]:
        """The bucket moduli of one layer, n-gram-size major and head minor."""
        return tuple(
            p for per_ngram in self.primes[layer_hash_index] for p in per_ngram
        )

    def bucket_offsets(self) -> np.ndarray:
        """[n_engram_layers, n_hash_cols] base row of each bucket range."""
        return np.array(
            [
                np.cumsum([0, *self.flat_primes(i)[:-1]])
                for i in range(len(self.layer_ids))
            ],
            dtype=np.int64,
        )

    def bucket_span(self, layer_hash_index: int) -> int:
        """Rows the bucket ranges of one layer address, i.e. one past the max id."""
        return int(sum(self.flat_primes(layer_hash_index)))

    def prime_array(self) -> np.ndarray:
        """[n_engram_layers, max_ngram_size - 1, n_heads] bucket moduli."""
        return np.array(self.primes, dtype=np.int64)

    @classmethod
    def from_config(cls, config: TextConfig) -> Optional["EngramLayout"]:
        """Build the layout for a config, or None when Engram is disabled.

        Fails closed on every malformed layout the official EngramLayout would
        accept and then mis-address. The load-bearing one is a bucket span
        wider than the table it indexes: that produces row ids past the end of
        the shard, which ParallelEngramEmbedding masks silently to zero rather
        than reporting, so it would degrade output instead of failing.
        """
        layer_ids = tuple(int(i) for i in config.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size = int(config.engram_max_ngram_size)
        n_heads = int(config.engram_n_heads)
        head_dim = int(config.engram_head_dim)
        num_embeddings = tuple(int(n) for n in config.engram_num_embeddings)
        vocab_size = int(config.engram_vocab_size)

        if max_ngram_size < 2:
            raise ValueError(
                "engram_max_ngram_size must be >= 2 when engram_layer_ids is "
                f"non-empty, got {max_ngram_size}: a layer that hashes no n-gram "
                "produces no hash ids at all"
            )
        if n_heads < 1:
            raise ValueError(f"engram_n_heads must be >= 1, got {n_heads}")
        if head_dim < 1 or head_dim % ENGRAM_FP8_BLOCK_SIZE:
            raise ValueError(
                "engram_head_dim must be a positive multiple of the E8M0 block "
                f"size {ENGRAM_FP8_BLOCK_SIZE}, got {head_dim}: a row carries "
                "exactly head_dim // block scale codes and a partial block has "
                "no scale to apply"
            )
        if vocab_size < 2:
            raise ValueError(
                f"engram_vocab_size must be >= 2, got {vocab_size}: it is the "
                "value each (n-gram size, head) bucket range starts searching "
                "for its prime modulus from"
            )
        if len(num_embeddings) != len(layer_ids):
            raise ValueError(
                f"engram_num_embeddings has {len(num_embeddings)} entries but "
                f"engram_layer_ids has {len(layer_ids)}: exactly one table row "
                "count per Engram layer is required"
            )
        if any(a >= b for a, b in zip(layer_ids, layer_ids[1:])):
            raise ValueError(
                f"engram_layer_ids must be strictly increasing, got {layer_ids}"
            )
        n_layers = int(config.num_hidden_layers)
        if any(not 0 <= i < n_layers for i in layer_ids):
            raise ValueError(
                f"engram_layer_ids {layer_ids} must all lie in "
                f"[0, num_hidden_layers={n_layers})"
            )

        primes: List[Tuple[Tuple[int, ...], ...]] = []
        seen = set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))

        layout = cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=num_embeddings,
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=head_dim,
        )
        for index, layer_id in enumerate(layer_ids):
            rows = num_embeddings[index]
            if rows < 1:
                raise ValueError(
                    f"engram_num_embeddings[{index}] (layer {layer_id}) must be "
                    f"positive, got {rows}"
                )
            span = layout.bucket_span(index)
            if span > rows:
                raise ValueError(
                    f"Engram layer {layer_id} bucket ranges span {span} rows but "
                    f"engram_num_embeddings[{index}] declares only {rows}: the "
                    "hash would address past the end of the table"
                )
        return layout


def validate_engram_config(config: TextConfig) -> Optional[EngramLayout]:
    """Fail closed on every Engram config the official modules cannot express."""
    layout = EngramLayout.from_config(config)
    if layout is None:
        return None
    if int(config.engram_compressed_vocab_size) < 1:
        raise ValueError(
            "engram_compressed_vocab_size must be positive: every hash "
            "multiplier is derived from it, so a wrong value silently rehashes "
            "both tables instead of failing"
        )
    if not 0 <= int(config.engram_pad_token_id) < int(config.vocab_size):
        raise ValueError(
            f"engram_pad_token_id={config.engram_pad_token_id} is outside "
            f"[0, vocab_size={config.vocab_size}): it fills the n-gram slots of "
            "positions with no usable history"
        )
    if int(config.hc_mult) < 1:
        raise ValueError(
            f"hc_mult must be >= 1, got {config.hc_mult}: Engram writes into the "
            "hc_mult-expanded residual stream"
        )
    return layout


class EngramNgramHasher:
    """Maps each position to the hash ids of the n-grams ending there.

    Faithful to inference/engram.py NgramHashState. Raw token ids go through
    the compressed map, then each position is hashed together with the
    max_ngram_size - 1 tokens before it. Look-back stops at the start of the
    sequence and at any dead token (an image span, cached as ENGRAM_DEAD_TOKEN),
    so an n-gram never spans one; blocked slots are filled with the compressed
    pad id. The XOR is accumulated one look-back at a time, so the running
    value after step i is the hash of the (i+1)-gram, and each lands in its own
    prime-sized bucket range.

    The history cache is [max_batch_size, max_seq_len] int64 and is carried
    across the prefill/decode split. It is an always-real, non-quantizable
    8 bytes/token/batch fixed cost, and is shared once by *both* Engram layers
    rather than duplicated per layer -- the per-layer difference lives entirely
    in the multipliers and the bucket primes.

    Deliberately host-side (numpy int64) rather than device-side. The ids this
    produces are not activations: they are row addresses that must reach the
    host anyway to drive the file-backed gather in BoundedEngramRowCache, and
    int64 multiply/XOR has to be exact or unrelated n-grams alias onto one row.
    """

    def __init__(
        self,
        layout: EngramLayout,
        token_map: Sequence[int],
        pad_token_id: int,
        compressed_vocab_size: int,
        max_batch_size: int = 1,
        max_seq_len: int = 4096,
    ):
        if max_batch_size < 1 or max_seq_len < 1:
            raise ValueError(
                f"max_batch_size={max_batch_size} and max_seq_len={max_seq_len} "
                "must both be positive"
            )
        token_map_arr = np.asarray(token_map, dtype=np.int64)
        if token_map_arr.ndim != 1 or token_map_arr.size < 1:
            raise ValueError(
                "token_map must be a 1-D lookup with one compressed id per raw "
                f"token id, got shape {token_map_arr.shape}"
            )
        if int(token_map_arr.min()) < 0:
            raise ValueError("token_map contains a negative compressed id")
        observed = int(token_map_arr.max()) + 1
        if observed > compressed_vocab_size:
            raise ValueError(
                f"token_map addresses {observed} compressed ids but "
                f"engram_compressed_vocab_size is {compressed_vocab_size}: every "
                "hash multiplier is derived from that size, so a mismatch "
                "rehashes the whole table"
            )
        if not 0 <= pad_token_id < token_map_arr.size:
            raise ValueError(
                f"engram_pad_token_id={pad_token_id} is outside the token_map "
                f"range [0, {token_map_arr.size})"
            )

        self.layout = layout
        self.vocab_size = int(token_map_arr.size)
        self.compressed_vocab_size = int(compressed_vocab_size)
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.token_map = token_map_arr
        self.pad_id = int(token_map_arr[pad_token_id])
        self.primes = layout.prime_array()
        self.offsets = layout.bucket_offsets()
        self.multipliers = compute_engram_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, self.compressed_vocab_size
        )
        # Initialized to DEAD, not to zero. The reference uses torch.empty and
        # relies on start_pos advancing monotonically from 0 so a slot is
        # always written before it is read; seeding DEAD instead means a
        # misuse blocks look-back rather than inventing a phantom token-0
        # n-gram, which would be indistinguishable from a real one.
        self.cache = np.full(
            (self.max_batch_size, self.max_seq_len), ENGRAM_DEAD_TOKEN, dtype=np.int64
        )

    def reset(self) -> None:
        """Drop the compressed-id history, e.g. between unrelated sequences.

        Every slot returns to DEAD, so the next sequence look-back stops at
        its own start instead of reaching into the previous one.
        """
        self.cache[...] = ENGRAM_DEAD_TOKEN

    def nbytes(self) -> int:
        """Bytes held by the shared history cache (8 per token per batch row)."""
        return int(self.cache.nbytes)

    def hash_ids(
        self,
        input_ids,
        start_pos: int = 0,
        token_mask=None,
    ) -> np.ndarray:
        """Host-side [B, L, n_engram_layers, n_hash_cols] int64 row ids.

        token_mask is [B, L] and False for tokens that take no part in an
        n-gram (image spans), matching Transformer.forward, which passes
        ~image_mask.
        """
        ids = np.asarray(input_ids)
        if ids.ndim != 2:
            raise ValueError(
                f"input_ids must be [batch, seqlen], got shape {ids.shape}"
            )
        ids = ids.astype(np.int64, copy=False)
        batch, seqlen = ids.shape
        if batch > self.max_batch_size:
            raise ValueError(
                f"batch {batch} exceeds max_batch_size={self.max_batch_size}: the "
                "shared n-gram history cache is preallocated and never grows"
            )
        if start_pos < 0 or start_pos + seqlen > self.max_seq_len:
            raise ValueError(
                f"positions [{start_pos}, {start_pos + seqlen}) fall outside the "
                f"history cache range [0, max_seq_len={self.max_seq_len})"
            )
        if seqlen and (int(ids.min()) < 0 or int(ids.max()) >= self.vocab_size):
            raise ValueError(
                f"input_ids must lie in [0, vocab_size={self.vocab_size}); got "
                f"[{int(ids.min())}, {int(ids.max())}]"
            )

        compressed = self.token_map[ids]
        if token_mask is not None:
            mask = np.asarray(token_mask).astype(bool, copy=False)
            if mask.shape != ids.shape:
                raise ValueError(
                    f"token_mask shape {mask.shape} must match input_ids shape "
                    f"{ids.shape}"
                )
            compressed = np.where(mask, compressed, ENGRAM_DEAD_TOKEN)
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = np.broadcast_to(
            np.arange(start_pos, start_pos + seqlen, dtype=np.int64), (batch, seqlen)
        )
        history = self.cache[:batch]
        blocked = np.zeros_like(positions, dtype=bool)
        tokens = []
        for shift in range(self.layout.max_ngram_size):
            source = np.take_along_axis(
                history, np.clip(positions - shift, 0, None), axis=1
            )
            blocked = blocked | (positions < shift) | (source == ENGRAM_DEAD_TOKEN)
            tokens.append(np.where(blocked, self.pad_id, source))
        stacked = np.stack(tokens, axis=-1)

        products = stacked[:, :, None, :] * self.multipliers
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self.layout.max_ngram_size):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self.primes[:, i - 1])
        return np.concatenate(hashes, axis=-1) + self.offsets

    def __call__(self, input_ids, start_pos: int = 0, token_mask=None) -> mx.array:
        """hash_ids as an mx.array, matching the reference return type."""
        return mx.array(self.hash_ids(input_ids, start_pos, token_mask))


def dequantize_engram_rows(
    weight, scale, block_size: int = ENGRAM_FP8_BLOCK_SIZE, dtype=mx.bfloat16
) -> mx.array:
    """Row-level FP8(E4M3) x E8M0 dequantization for gathered Engram rows.

    Faithful to inference/model.py ParallelEngramEmbedding.forward:
    values.float().unflatten(-1, (-1, block_size)) * scales.float().unsqueeze(-1),
    flattened back and cast to bfloat16.

    Deliberately *not* dequantize_fp8_block. A packed Linear weight carries a
    2-D (out_block, in_block) scale grid; an Engram scale is per row and 1-D
    along the row, one E8M0 code per block_size contiguous columns
    (head_dim // block_size == 8 codes for the official 256-wide rows). Reusing
    the 2-D tiling here would mis-partition the scales.

    Takes only the rows actually gathered. There is no call shape that decodes
    a whole table: weight is [n_rows, dim], and n_rows comes from the caller.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    weight_arr = np.asarray(weight)
    scale_arr = np.asarray(scale)
    if weight_arr.ndim != 2 or scale_arr.ndim != 2:
        raise ValueError(
            "dequantize_engram_rows expects 2-D [n_rows, dim] weight and "
            f"[n_rows, dim // block] scale, got weight.shape={weight_arr.shape} "
            f"scale.shape={scale_arr.shape}"
        )
    n_rows, dim = weight_arr.shape
    if scale_arr.shape[0] != n_rows:
        raise ValueError(f"weight has {n_rows} rows but scale has {scale_arr.shape[0]}")
    if dim % block_size:
        raise ValueError(
            f"row width {dim} is not divisible by block_size={block_size}; the "
            "Engram scale grid never carries a partial trailing block"
        )
    if scale_arr.shape[1] != dim // block_size:
        raise ValueError(
            f"scale shape {scale_arr.shape} does not match the expected "
            f"{(n_rows, dim // block_size)} row-scale grid for a [{n_rows}, {dim}] "
            f"row block at block_size={block_size}"
        )
    if n_rows == 0:
        return mx.zeros((0, dim), dtype=dtype)
    values = mx.from_fp8(mx.array(weight_arr.astype(np.uint8)), dtype=mx.float32)
    scales = decode_e8m0_scale(mx.array(scale_arr.astype(np.uint8)))
    values = values.reshape(n_rows, dim // block_size, block_size)
    values = values * scales[:, :, None]
    return values.reshape(n_rows, dim).astype(dtype)


class EngramRowStore:
    """Read-only, row-addressed view of one packed Engram table shard.

    The whole point of this interface is what it does *not* offer: there is no
    way to ask it for the table. read_rows takes explicit row ids and returns
    exactly that many packed rows, so peak memory is bounded by the request,
    never by the ~91.6-98.3 GiB the two official tables occupy.
    """

    num_rows: int = 0
    dim: int = 0
    block_size: int = ENGRAM_FP8_BLOCK_SIZE

    @property
    def scale_dim(self) -> int:
        return self.dim // self.block_size

    @property
    def row_nbytes(self) -> int:
        """Packed bytes per row: dim E4M3 codes plus dim // block E8M0 codes."""
        return self.dim + self.scale_dim

    def read_rows(self, row_ids) -> Tuple[np.ndarray, np.ndarray]:
        """Return ([n, dim] uint8 weight, [n, dim // block] uint8 scale)."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# safetensors dtype tags. The official tables are float8_e4m3fn rows with
# float8_e8m0fnu scales; U8 is also accepted because both are byte-identical
# containers and not every writer can emit the fp8 tags. Anything else is
# refused rather than reinterpreted.
ENGRAM_WEIGHT_SAFETENSORS_DTYPES = ("F8_E4M3", "U8")
ENGRAM_SCALE_SAFETENSORS_DTYPES = ("F8_E8M0", "U8")
_SAFETENSORS_HEADER_LIMIT = 100_000_000


class SafetensorsEngramRowStore(EngramRowStore):
    """A file-backed EngramRowStore that reads individual rows by byte range.

    Parses the safetensors header itself and then seeks to the exact bytes of
    each requested row. It deliberately does not go through mx.load or any
    other whole-tensor loader: loading either official Engram tensor that way
    would materialize ~91.6-98.3 GiB. This is the same per-row access pattern
    the Plan 0051 M1 layout probe validated at the storage-format level with a
    byte-range HTTP GET, applied locally to a real file.

    Reading is O(rows requested), so a tiny fixture and a 384-million-row shard
    exercise the identical code path.
    """

    def __init__(
        self,
        path: str,
        weight_key: str = "weight",
        scale_key: str = "scale",
        block_size: int = ENGRAM_FP8_BLOCK_SIZE,
        row_start: int = 0,
        num_rows: Optional[int] = None,
    ):
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.path = str(path)
        self.weight_key = weight_key
        self.scale_key = scale_key
        self.block_size = int(block_size)
        self.rows_read = 0
        self.bytes_read = 0
        self._handle = None

        with open(self.path, "rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                raise ValueError(
                    f"{self.path} is too short to be a safetensors file (no 8-byte "
                    "header length)"
                )
            header_len = int.from_bytes(raw_len, "little", signed=False)
            if header_len < 2 or header_len > _SAFETENSORS_HEADER_LIMIT:
                raise ValueError(
                    f"{self.path} declares an implausible safetensors header "
                    f"length of {header_len} bytes"
                )
            raw_header = handle.read(header_len)
            if len(raw_header) != header_len:
                raise ValueError(f"{self.path} has a truncated safetensors header")
            try:
                header = json.loads(raw_header.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{self.path} has an unreadable safetensors header: {exc}"
                ) from exc
            handle.seek(0, 2)
            file_size = handle.tell()

        if not isinstance(header, dict):
            raise ValueError(f"{self.path} safetensors header is not a JSON object")
        self._data_start = 8 + header_len

        weight_shape, weight_begin = self._entry(
            header, self.weight_key, ENGRAM_WEIGHT_SAFETENSORS_DTYPES, file_size
        )
        scale_shape, scale_begin = self._entry(
            header, self.scale_key, ENGRAM_SCALE_SAFETENSORS_DTYPES, file_size
        )
        self.full_num_rows, self.dim = weight_shape
        if self.full_num_rows < 1 or self.dim < 1:
            raise ValueError(
                f"{self.path}:{self.weight_key} has an empty shape {weight_shape}"
            )
        if self.dim % self.block_size:
            raise ValueError(
                f"{self.path}:{self.weight_key} row width {self.dim} is not "
                f"divisible by block_size={self.block_size}"
            )
        expected_scale = (self.full_num_rows, self.dim // self.block_size)
        if tuple(scale_shape) != expected_scale:
            raise ValueError(
                f"{self.path}:{self.scale_key} has shape {tuple(scale_shape)} but "
                f"the [{self.full_num_rows}, {self.dim}] weight at "
                f"block_size={self.block_size} requires {expected_scale}"
            )
        self.row_start = int(row_start)
        self.num_rows = self.full_num_rows if num_rows is None else int(num_rows)
        if self.row_start < 0 or self.row_start >= self.full_num_rows:
            raise ValueError(
                f"row_start {self.row_start} is outside the {self.full_num_rows}-row table"
            )
        if self.num_rows < 1:
            raise ValueError(f"num_rows must be positive, got {self.num_rows}")
        self._weight_begin = self._data_start + weight_begin
        self._scale_begin = self._data_start + scale_begin

    def _entry(self, header, key, allowed_dtypes, file_size):
        if key not in header:
            named = sorted(k for k in header if k != "__metadata__")
            raise ValueError(
                f"{self.path} has no tensor named {key!r}; found " f"{named}"
            )
        entry = header[key]
        dtype = entry.get("dtype")
        if dtype not in allowed_dtypes:
            raise ValueError(
                f"{self.path}:{key} has dtype {dtype!r}; the Engram row store "
                f"accepts only {allowed_dtypes} (one byte per element)"
            )
        shape = entry.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(
                f"{self.path}:{key} must be a 2-D tensor, got shape {shape!r}"
            )
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(
                f"{self.path}:{key} has malformed data_offsets {offsets!r}"
            )
        begin, end = int(offsets[0]), int(offsets[1])
        expected = int(shape[0]) * int(shape[1])
        if begin < 0 or end - begin != expected:
            raise ValueError(
                f"{self.path}:{key} spans {end - begin} bytes but shape {shape} at "
                f"one byte per element needs {expected}"
            )
        if self._data_start + end > file_size:
            raise ValueError(
                f"{self.path}:{key} ends at byte {self._data_start + end} but the "
                f"file is only {file_size} bytes long"
            )
        return (int(shape[0]), int(shape[1])), begin

    def _file(self):
        if self._handle is None or self._handle.closed:
            self._handle = open(self.path, "rb")
        return self._handle

    def read_rows(self, row_ids) -> Tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
        count = int(ids.size)
        weight = np.empty((count, self.dim), dtype=np.uint8)
        scale = np.empty((count, self.scale_dim), dtype=np.uint8)
        if count == 0:
            return weight, scale
        if int(ids.min()) < 0 or int(ids.max()) >= self.num_rows:
            raise IndexError(
                f"row ids [{int(ids.min())}, {int(ids.max())}] fall outside "
                f"[0, num_rows={self.num_rows}) of {self.path}"
            )
        physical_ids = ids + self.row_start
        if int(physical_ids.max()) >= self.full_num_rows:
            raise IndexError(
                f"local row {int(ids.max())} maps past the physical "
                f"{self.full_num_rows}-row table"
            )
        handle = self._file()
        for position, row_id in enumerate(physical_ids.tolist()):
            handle.seek(self._weight_begin + row_id * self.dim)
            chunk = handle.read(self.dim)
            if len(chunk) != self.dim:
                raise ValueError(
                    f"{self.path}:{self.weight_key} row {row_id} is truncated"
                )
            weight[position] = np.frombuffer(chunk, dtype=np.uint8)
            handle.seek(self._scale_begin + row_id * self.scale_dim)
            chunk = handle.read(self.scale_dim)
            if len(chunk) != self.scale_dim:
                raise ValueError(
                    f"{self.path}:{self.scale_key} row {row_id} is truncated"
                )
            scale[position] = np.frombuffer(chunk, dtype=np.uint8)
        self.rows_read += count
        self.bytes_read += count * self.row_nbytes
        return weight, scale

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._handle = None


class EngramCacheStats:
    """Observable cost of an Engram gather: what was asked for vs what was read."""

    __slots__ = (
        "calls",
        "requested_rows",
        "unique_rows",
        "hits",
        "misses",
        "rows_fetched",
        "evictions",
    )

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.requested_rows = 0
        self.unique_rows = 0
        self.hits = 0
        self.misses = 0
        self.rows_fetched = 0
        self.evictions = 0

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)}" for name in self.__slots__)
        return f"EngramCacheStats({fields})"


class BoundedEngramRowCache:
    """A bounded LRU of packed Engram rows in front of an EngramRowStore.

    Three properties matter and are all directly observable through stats and
    nbytes():

      * **Dedup.** A gather of B x L x 24 hash ids contains heavy repetition
        (the same n-gram recurs, and blocked look-backs all collapse onto the
        pad n-gram). Only distinct row ids are ever read from the store and
        only distinct rows are ever dequantized; the request order is restored
        with an index take afterwards.
      * **Bounded residency.** At most max_rows packed rows are retained.
        Residency is what the bound applies to, so cache memory is
        max_rows * (dim + dim // block) bytes regardless of table size or of
        how many rows the caller has asked for over time.
      * **Packed at rest.** Rows are cached in their packed FP8 + E8M0 form and
        dequantized per lookup, exactly as ParallelEngramEmbedding does. Caching
        decoded rows would nearly double residency for no fidelity gain.

    A single gather whose distinct row count exceeds max_rows is served
    correctly rather than refused: the call materializes the rows it was asked
    for, and residency afterwards is still bounded by max_rows.
    """

    def __init__(self, store: EngramRowStore, max_rows: int = 8192):
        if max_rows < 1:
            raise ValueError(
                f"max_rows must be at least 1, got {max_rows}: a zero-row cache "
                "cannot hold the row it just fetched"
            )
        if store.dim < 1 or store.num_rows < 1:
            raise ValueError(
                f"store must expose a positive num_rows/dim, got "
                f"num_rows={store.num_rows} dim={store.dim}"
            )
        self.store = store
        self.max_rows = int(max_rows)
        self.stats = EngramCacheStats()
        self._entries = OrderedDict()

    @property
    def dim(self) -> int:
        return self.store.dim

    @property
    def block_size(self) -> int:
        return self.store.block_size

    @property
    def resident_rows(self) -> int:
        return len(self._entries)

    def nbytes(self) -> int:
        """Packed bytes currently resident. Grows with the bound, not the table."""
        return self.resident_rows * self.store.row_nbytes

    def clear(self) -> None:
        self._entries.clear()

    def gather_packed(
        self, row_ids
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (unique_ids, packed weight, packed scale, inverse index).

        weight is [n_unique, dim] and inverse maps each requested position onto
        its row in that block, so the caller can restore request order without
        ever holding a duplicate copy of a row.
        """
        ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
        self.stats.calls += 1
        self.stats.requested_rows += int(ids.size)
        if ids.size == 0:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, self.dim), dtype=np.uint8),
                np.empty((0, self.store.scale_dim), dtype=np.uint8),
                np.empty((0,), dtype=np.int64),
            )

        # First-seen order, not sorted order: the LRU recency this produces is
        # the order the caller actually asked in.
        unique_ids, first_index, inverse = np.unique(
            ids, return_index=True, return_inverse=True
        )
        order = np.argsort(first_index)
        unique_ids = unique_ids[order]
        remap = np.empty(order.size, dtype=np.int64)
        remap[order] = np.arange(order.size, dtype=np.int64)
        inverse = remap[inverse.reshape(-1)]
        self.stats.unique_rows += int(unique_ids.size)

        missing = [int(i) for i in unique_ids.tolist() if i not in self._entries]
        self.stats.hits += int(unique_ids.size) - len(missing)
        self.stats.misses += len(missing)
        if missing:
            fetched_w, fetched_s = self.store.read_rows(missing)
            self.stats.rows_fetched += len(missing)
            for position, row_id in enumerate(missing):
                self._entries[row_id] = (fetched_w[position], fetched_s[position])

        weight = np.empty((unique_ids.size, self.dim), dtype=np.uint8)
        scale = np.empty((unique_ids.size, self.store.scale_dim), dtype=np.uint8)
        for position, row_id in enumerate(unique_ids.tolist()):
            entry = self._entries[int(row_id)]
            weight[position] = entry[0]
            scale[position] = entry[1]
            self._entries.move_to_end(int(row_id))

        while len(self._entries) > self.max_rows:
            self._entries.popitem(last=False)
            self.stats.evictions += 1
        return unique_ids, weight, scale, inverse

    def gather_rows(self, row_ids, dtype=mx.bfloat16) -> mx.array:
        """Dequantized [n_requested, dim] rows, in request order."""
        _, weight, scale, inverse = self.gather_packed(row_ids)
        rows = dequantize_engram_rows(weight, scale, self.block_size, dtype)
        if rows.shape[0] == 0:
            return rows
        return mx.take(rows, mx.array(inverse.astype(np.int32)), axis=0)


class DeepseekV41EngramEmbedding(nn.Module):
    """The row-sharded n-gram hash table (inference/model.py ParallelEngramEmbedding).

    The table stays packed FP8 on disk and rows are dequantized on lookup. This
    module therefore holds **no parameters at all**: there is no dense
    [num_embeddings, dim] array anywhere in it, and
    ``tree_flatten(self.parameters())`` is empty by construction. All state is
    the injected BoundedEngramRowCache and its file-backed store.

    Sharding matches the reference exactly: each of world_size ranks owns
    ``ceil(num_embeddings / world_size)`` contiguous rows, ids outside the
    local shard are zero-masked, and the per-rank partial results are summed
    with an all-reduce. That reducer is an explicit constructor argument rather
    than an implicit global: at world_size == 1 none is needed and none is
    used, and at world_size > 1 a missing one fails loud instead of silently
    returning this rank's fragment of the row as if it were the whole row.

    Shard-relative masking is the reference behaviour and is kept. A *globally*
    out-of-table id is a different thing -- it can only come from a malformed
    layout, and the reference would mask it to zero indistinguishably from a
    legitimate remote row -- so that case is refused here instead.
    """

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        cache: BoundedEngramRowCache,
        rank: int = 0,
        world_size: int = 1,
        all_reduce: Optional[Callable[[mx.array], mx.array]] = None,
        block_size: int = ENGRAM_FP8_BLOCK_SIZE,
    ):
        super().__init__()
        if num_embeddings < 1:
            raise ValueError(f"num_embeddings must be positive, got {num_embeddings}")
        if dim < 1 or dim % block_size:
            raise ValueError(
                f"dim must be a positive multiple of block_size={block_size}, got "
                f"{dim}"
            )
        if world_size < 1:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(
                f"rank {rank} is outside the range [0, world_size={world_size})"
            )
        if cache.dim != dim:
            raise ValueError(
                f"the row store serves {cache.dim}-wide rows but this embedding "
                f"expects {dim}"
            )
        if cache.block_size != block_size:
            raise ValueError(
                f"the row store uses block_size={cache.block_size} but this "
                f"embedding expects {block_size}"
            )

        self.num_embeddings = int(num_embeddings)
        self.dim = int(dim)
        self.block_size = int(block_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.part_num_embeddings = -(-self.num_embeddings // self.world_size)
        self.vocab_start_idx = self.rank * self.part_num_embeddings
        self.vocab_end_idx = self.vocab_start_idx + self.part_num_embeddings
        if cache.store.num_rows != self.part_num_embeddings:
            raise ValueError(
                f"the row store holds {cache.store.num_rows} rows but rank "
                f"{self.rank} of {self.world_size} owns "
                f"{self.part_num_embeddings} rows of the {self.num_embeddings}-row "
                "table; every rank allocates ceil(rows / world_size), padding "
                "included"
            )
        self.cache = cache
        self.all_reduce = all_reduce

    def nbytes(self) -> int:
        """Resident bytes. A function of the cache bound, never of the table."""
        return self.cache.nbytes()

    def __call__(self, indices, dtype=mx.bfloat16) -> mx.array:
        ids = np.asarray(indices)
        ids = ids.astype(np.int64, copy=False)
        shape = tuple(ids.shape)
        if ids.size == 0:
            return mx.zeros(shape + (self.dim,), dtype=dtype)
        low, high = int(ids.min()), int(ids.max())
        if low < 0 or high >= self.num_embeddings:
            raise IndexError(
                f"Engram row ids [{low}, {high}] fall outside "
                f"[0, num_embeddings={self.num_embeddings}); a hash id past the "
                "table can only come from a malformed EngramLayout, and masking "
                "it to zero would be indistinguishable from a remote-shard row"
            )
        mask = (ids < self.vocab_start_idx) | (ids >= self.vocab_end_idx)
        local = np.where(mask, 0, ids - self.vocab_start_idx)
        rows = self.cache.gather_rows(local.reshape(-1), dtype=dtype)
        rows = rows.reshape(shape + (self.dim,))
        if bool(mask.any()):
            rows = mx.where(mx.array(mask)[..., None], mx.zeros_like(rows), rows)
        if self.world_size > 1:
            if self.all_reduce is None:
                raise RuntimeError(
                    f"DeepseekV41EngramEmbedding is sharded over "
                    f"{self.world_size} ranks but no all_reduce was injected. "
                    "Each rank holds a disjoint row range and zero-masks every "
                    "id outside it, so without the cross-rank sum this rank "
                    "would return zeros for most lookups instead of the row. "
                    "Pass all_reduce=..., or run at world_size == 1."
                )
            rows = self.all_reduce(rows)
            if rows is None:
                raise RuntimeError(
                    "the injected all_reduce returned None; it must return the "
                    "summed array (an in-place reducer should return its argument)"
                )
        return rows


def engram_signed_sqrt_sigmoid_gate(
    stream: mx.array,
    key: mx.array,
    weight: mx.array,
    eps: float,
    clamp_value: float = ENGRAM_GATE_CLAMP,
) -> mx.array:
    """The Engram injection gate (inference/model.py Engram.forward).

    Engram does not add its lookup into the residual stream: it gates it by how
    well the looked-up key matches the stream. Three details are load-bearing
    and easy to get wrong:

      * the RMS normalization is per (token, hc copy) over dim, **not** jointly
        over the hc copies, and it is applied as a product of two rsqrt terms
        rather than by normalizing either tensor;
      * the dot product is additionally scaled by dim ** -0.5;
      * a **signed sqrt** is taken before the sigmoid, matching the training
        kernel, with |dot| floored at clamp_value first.

    stream and key are [..., hc_mult, dim]; weight is the q_weight * k_weight
    product, [hc_mult, dim]. Returns the [..., hc_mult] gate.
    """
    if stream.shape != key.shape:
        raise ValueError(
            f"stream shape {stream.shape} and key shape {key.shape} must match"
        )
    if stream.ndim < 2 or weight.shape != stream.shape[-2:]:
        raise ValueError(
            f"weight shape {weight.shape} must be the trailing (hc_mult, dim) of "
            f"the stream shape {stream.shape}"
        )
    if clamp_value <= 0:
        raise ValueError(f"clamp_value must be positive, got {clamp_value}")
    dim = stream.shape[-1]
    h = stream.astype(mx.float32)
    k = key.astype(mx.float32)
    w = weight.astype(mx.float32)
    rstd = mx.rsqrt(mx.mean(h * h, axis=-1) + eps) * mx.rsqrt(
        mx.mean(k * k, axis=-1) + eps
    )
    dot = mx.sum(h * w * k, axis=-1) * rstd * dim**-0.5
    magnitude = mx.sqrt(mx.maximum(mx.abs(dot), clamp_value))
    # copysign(magnitude, dot). dot is a real sum, so the only value whose sign
    # this misses is a negative zero, where the clamp has already flattened the
    # result to sigmoid(+/-sqrt(clamp_value)).
    signed = mx.where(dot < 0, -magnitude, magnitude)
    return mx.sigmoid(signed)


class DeepseekV41Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream (inference/model.py Engram).

    The n_hash_cols hash ids fetch that many rows; wkv turns the concatenated
    rows into one key per hc copy plus a single shared value; the key gates the
    value into the hc_mult-expanded stream through
    engram_signed_sqrt_sigmoid_gate. token_mask is [B, L] and False shuts the
    gate completely, so image-span positions pass through untouched.

    The embedding is injected rather than constructed here: it is file-backed
    and rank-specific, and nothing about this module should imply that a table
    can be allocated.
    """

    def __init__(
        self,
        config: TextConfig,
        layer_id: int,
        layout: EngramLayout,
        embedding: DeepseekV41EngramEmbedding,
    ):
        super().__init__()
        if layer_id not in layout.layer_ids:
            raise ValueError(
                f"layer {layer_id} is not an Engram layer; engram_layer_ids is "
                f"{layout.layer_ids}"
            )
        self.layer_id = int(layer_id)
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        expected_rows = layout.num_embeddings[self.layer_hash_index]
        if embedding.num_embeddings != expected_rows:
            raise ValueError(
                f"Engram layer {layer_id} indexes a {expected_rows}-row table but "
                f"the injected embedding declares {embedding.num_embeddings}"
            )
        if embedding.dim != layout.head_dim:
            raise ValueError(
                f"Engram rows are {layout.head_dim} wide but the injected "
                f"embedding serves {embedding.dim}"
            )
        self.dim = int(config.hidden_size)
        self.hc_mult = int(config.hc_mult)
        self.eps = float(config.rms_norm_eps)
        self.clamp_value = ENGRAM_GATE_CLAMP
        self.n_hash_cols = layout.n_hash_cols
        self.head_dim = layout.head_dim
        self.embed = embedding
        self.wkv = _make_fp8_linear(
            self.n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
        )
        self.q_weight = mx.ones((self.hc_mult, self.dim))
        self.k_weight = mx.ones((self.hc_mult, self.dim))

    def __call__(self, x: mx.array, hash_ids, token_mask=None) -> mx.array:
        """x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]."""
        if x.ndim != 4 or x.shape[-2:] != (self.hc_mult, self.dim):
            raise ValueError(
                f"expected a [batch, seqlen, hc_mult={self.hc_mult}, "
                f"dim={self.dim}] stream, got shape {x.shape}"
            )
        ids = np.asarray(hash_ids).astype(np.int64, copy=False)
        if ids.ndim != 3 or ids.shape[:2] != tuple(x.shape[:2]):
            raise ValueError(
                f"hash_ids shape {ids.shape} must be [batch, seqlen, "
                f"n_hash_cols] matching the stream batch/seqlen {tuple(x.shape[:2])}"
            )
        if ids.shape[-1] != self.n_hash_cols:
            raise ValueError(
                f"hash_ids carries {ids.shape[-1]} columns but this layer hashes "
                f"{self.n_hash_cols} (max_ngram_size - 1) * n_heads per token"
            )
        batch, seqlen = ids.shape[0], ids.shape[1]
        rows = self.embed(ids)
        kv = self.wkv(rows.reshape(batch, seqlen, self.n_hash_cols * self.head_dim))
        key = kv[..., : self.hc_mult * self.dim]
        value = kv[..., self.hc_mult * self.dim :]
        key = key.astype(mx.float32).reshape(batch, seqlen, self.hc_mult, self.dim)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        h = x.astype(mx.float32)
        gate = engram_signed_sqrt_sigmoid_gate(
            h, key, weight, self.eps, self.clamp_value
        )
        if token_mask is not None:
            mask = np.asarray(token_mask).astype(bool, copy=False)
            if mask.shape != (batch, seqlen):
                raise ValueError(
                    f"token_mask shape {mask.shape} must be [batch, seqlen] "
                    f"{(batch, seqlen)}"
                )
            gate = mx.where(mx.array(mask)[..., None], gate, mx.zeros_like(gate))
        out = h + gate[..., None] * value.astype(mx.float32)[..., None, :]
        return out.astype(x.dtype)


# --------------------------------------------------------------------------- #
# Vision tower, aligner, and exact image-grid / token-budget arithmetic       #
#                                                                              #
# Exact port of inference/vision.py and inference/image_processor.py at the    #
# pinned revision. Public tensor names match inference/convert.py.             #
# --------------------------------------------------------------------------- #

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
VISION_RMS_NORM_EPS = 1e-6
_IMAGE_TOKEN_TYPES = {IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END}
def vision_enabled(config) -> bool:
    vision = config.vision_config if hasattr(config, "vision_config") else config
    return int(vision.num_hidden_layers) > 0


def public_checkpoint_name(source_name: str) -> str:
    """Map a HuggingFace / official export name to the public inference name.

    Faithful to inference/convert.py (strip model., self_attn->attn,
    mlp->ffn except under vision., weight_scale_inv->scale,
    e_score_correction_bias->bias). Sharding is not applied here.
    """
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("checkpoint tensor names must be non-empty strings")
    name = source_name
    if name.startswith("model."):
        name = name[len("model.") :]
    name = name.replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    return name


def vision_public_weight_names(vision: VisionConfig, dim: int) -> List[str]:
    """Official public names for the vision tower, aligner, and delimiters."""
    if int(vision.num_hidden_layers) <= 0:
        return []
    if dim <= 0:
        raise ValueError(f"aligner output dim must be positive, got {dim}")
    names = [
        "vision.patch_embed.proj.weight",
        "vision.patch_embed.proj.bias",
        "vision.norm.weight",
        "aligner.w1.weight",
        "aligner.w1.bias",
        "aligner.w2.weight",
        "aligner.w2.bias",
        "image_start",
        "image_end",
        "image_newline",
    ]
    for i in range(int(vision.num_hidden_layers)):
        prefix = f"vision.blocks.{i}"
        names.extend(
            [
                f"{prefix}.norm1.weight",
                f"{prefix}.attn.wqkv.weight",
                f"{prefix}.attn.wqkv.bias",
                f"{prefix}.attn.wo.weight",
                f"{prefix}.attn.wo.bias",
                f"{prefix}.norm2.weight",
                f"{prefix}.mlp.w1.weight",
                f"{prefix}.mlp.w2.weight",
            ]
        )
    return names


def remaining_gate_bias_and_wo_a_names(
    n_layers: int, vision_on: bool
) -> List[str]:
    """Public names the later loader must accept for gate-bias and wo_a."""
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    names: List[str] = []
    for i in range(n_layers):
        names.append(f"layers.{i}.attn.wo_a.weight")
        names.append(f"layers.{i}.ffn.gate.bias")
        if vision_on:
            names.append(f"layers.{i}.ffn.gate.bias_vl")
    return names


def require_public_weights(
    weights: Dict[str, Any], names: Sequence[str]
) -> None:
    if not isinstance(weights, dict):
        raise ValueError("weights must be a dict of public tensor names")
    missing = [name for name in names if name not in weights]
    if missing:
        shown = ", ".join(missing[:8])
        extra = " ..." if len(missing) > 8 else ""
        raise ValueError(
            f"missing required tensors ({len(missing)}): {shown}{extra}"
        )

def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    if n_llm_h < 1 or n_llm_w < 1:
        raise ValueError(
            f"image token grid must be at least 1x1, got {n_llm_h}x{n_llm_w}"
        )
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(
    best_height: int, best_width: int, patch_size: int, downsample_ratio: int
) -> Tuple[int, int]:
    if patch_size < 1 or downsample_ratio < 1:
        raise ValueError(
            f"patch_size and downsample_ratio must be positive, got "
            f"{patch_size}, {downsample_ratio}"
        )
    if best_height < patch_size or best_width < patch_size:
        raise ValueError(
            f"resized image {best_width}x{best_height} is smaller than "
            f"patch_size={patch_size}"
        )
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(
    height, width, patch_size, downsample_ratio, max_n_token
):
    """Largest aspect-preserving pixel size whose token grid fits max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(
        math.floor(max_w_float) * cell / width,
        math.floor(max_h_float) * cell / height,
    )
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def safe_resize(
    height,
    width,
    best_height,
    best_width,
    patch_size,
    downsample_ratio,
    max_n_token,
):
    n_llm_h, n_llm_w = llm_grid(
        best_height, best_width, patch_size, downsample_ratio
    )
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token
        )
        n_llm_h, n_llm_w = llm_grid(
            best_height, best_width, patch_size, downsample_ratio
        )
        n_tokens = num_image_tokens(n_llm_h, n_llm_w)
        if n_tokens > max_n_token:
            raise ValueError(
                f"image still costs {n_tokens} tokens after shrink, "
                f"budget is {max_n_token}"
            )
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, vision: VisionConfig):
    """Resize plan for an image; a pure function of size and vision config."""
    p = int(vision.patch_size)
    down = int(vision.downsample_ratio)
    max_n = int(vision.max_image_tokens)
    min_pixels = int(vision.min_pixels)
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    if p < 1 or down < 1 or max_n < 4:
        raise ValueError(
            f"malformed vision grid config: patch_size={p}, "
            f"downsample_ratio={down}, max_image_tokens={max_n}"
        )
    if min_pixels < 0:
        raise ValueError(f"min_pixels must be non-negative, got {min_pixels}")
    max_wh = vision.max_wh_ratio
    if max_wh is not None:
        if max_wh <= 0:
            raise ValueError(f"max_wh_ratio must be positive, got {max_wh}")
        if width > height * max_wh:
            width = height * max_wh
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(
        height, width, best_height, best_width, p, down, max_n
    )


def image_token_types(n_llm_h: int, n_llm_w: int) -> mx.array:
    if n_llm_h < 1 or n_llm_w < 1:
        raise ValueError(
            f"image token grid must be at least 1x1, got {n_llm_h}x{n_llm_w}"
        )
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return mx.array(types, dtype=mx.int32)

@dataclass
class ImageInput:
    start: int
    patches: Any
    n_vit_h: int
    n_vit_w: int
    types: Any


def _as_numpy(x):
    return np.asarray(x)


def _erf(x: mx.array) -> mx.array:
    x32 = x.astype(mx.float32)
    ax = mx.abs(x32)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    p = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592
    return mx.sign(x32) * (1.0 - p * t * mx.exp(-ax * ax))


def _gelu(x: mx.array) -> mx.array:
    x32 = x.astype(mx.float32)
    return 0.5 * x32 * (1.0 + _erf(x32 * (0.5 ** 0.5)))


def _silu(x: mx.array) -> mx.array:
    x32 = x.astype(mx.float32)
    return x32 * (1.0 / (1.0 + mx.exp(-x32)))


def _transpose(x: mx.array, axes: Tuple[int, ...]) -> mx.array:
    return mx.transpose(x, axes)


class _VisionLinear(nn.Module):
    """Dense linear matching torch.nn.Linear used by the official ViT/Aligner."""

    def __init__(self, in_dims: int, out_dims: int, bias: bool = True):
        super().__init__()
        if in_dims <= 0 or out_dims <= 0:
            raise ValueError(
                f"linear dimensions must be positive, got {in_dims}->{out_dims}"
            )
        self.weight = mx.zeros((out_dims, in_dims))
        self._has_bias = bool(bias)
        if self._has_bias:
            self.bias = mx.zeros((out_dims,))

    def __call__(self, x: mx.array) -> mx.array:
        y = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        if self._has_bias:
            y = y + self.bias.astype(mx.float32)
        return y


class _VisionRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = VISION_RMS_NORM_EPS):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"RMSNorm dim must be positive, got {dim}")
        self.eps = float(eps)
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        x32 = x32 * mx.rsqrt(
            mx.mean(x32 * x32, axis=-1, keepdims=True) + self.eps
        )
        return self.weight.astype(mx.float32) * x32


def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    if n_h < 1 or n_w < 1 or dim < 2 or dim % 2 != 0:
        raise ValueError(
            f"malformed 2D RoPE request: n_h={n_h}, n_w={n_w}, dim={dim}"
        )
    idx = mx.arange(0, dim, 2).astype(mx.float32)
    inv_freq = theta ** (-idx / dim)
    hpos = mx.broadcast_to(
        mx.arange(n_h).astype(mx.float32).reshape(n_h, 1), (n_h, n_w)
    )
    wpos = mx.broadcast_to(
        mx.arange(n_w).astype(mx.float32).reshape(1, n_w), (n_h, n_w)
    )
    freqs = mx.stack([hpos, wpos], axis=-1).reshape(n_h * n_w, 2, 1) * inv_freq
    freqs = freqs.reshape(n_h * n_w, dim)
    cos = mx.cos(freqs).reshape(n_h * n_w, 1, dim)
    sin = mx.sin(freqs).reshape(n_h * n_w, 1, dim)
    return cos, sin


def apply_vision_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"vision rotary expects even head_dim, got {x.shape[-1]}"
        )
    half = x.shape[-1] // 2
    x32 = x.astype(mx.float32)
    x1 = x32[..., :half]
    x2 = x32[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

class DeepseekV41PatchEmbed(nn.Module):
    def __init__(self, vision: VisionConfig):
        super().__init__()
        self.patch_size = int(vision.patch_size)
        in_dim = 3 * self.patch_size ** 2
        self.proj = _VisionLinear(in_dim, int(vision.hidden_size), bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x_np = _as_numpy(x)
        if x_np.ndim == 4:
            if x_np.shape[1] != 3 or x_np.shape[2] != self.patch_size or x_np.shape[3] != self.patch_size:
                raise ValueError(
                    f"patches must be [N, 3, {self.patch_size}, {self.patch_size}], got {x_np.shape}"
                )
            x = mx.array(x_np.reshape(x_np.shape[0], -1))
        elif x_np.ndim == 2:
            if x_np.shape[-1] != 3 * self.patch_size ** 2:
                raise ValueError(
                    f"flat patches must have width {3 * self.patch_size ** 2}, got {x_np.shape}"
                )
            x = mx.array(x_np)
        else:
            raise ValueError(f"unsupported patch rank {x_np.ndim}, shape {x_np.shape}")
        return self.proj(x)


class DeepseekV41VisionAttention(nn.Module):
    def __init__(self, vision: VisionConfig):
        super().__init__()
        dim = int(vision.hidden_size)
        self.n_heads = int(vision.num_attention_heads)
        if dim % self.n_heads != 0:
            raise ValueError(
                f"vision hidden_size {dim} is not divisible by n_heads {self.n_heads}"
            )
        self.head_dim = dim // self.n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"vision head_dim must be even for 2D RoPE, got {self.head_dim}")
        self.wqkv = _VisionLinear(dim, 3 * dim, bias=True)
        self.wo = _VisionLinear(dim, dim, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        n = x.shape[0]
        qkv = self.wqkv(x)
        width = qkv.shape[-1] // 3
        q = qkv[:, :width].reshape(n, self.n_heads, self.head_dim)
        k = qkv[:, width : 2 * width].reshape(n, self.n_heads, self.head_dim)
        v = qkv[:, 2 * width :].reshape(n, self.n_heads, self.head_dim)
        q = apply_vision_rotary(q, cos, sin)
        k = apply_vision_rotary(k, cos, sin)
        qh = _transpose(q, (1, 0, 2))
        kh = _transpose(k, (1, 0, 2))
        vh = _transpose(v.astype(mx.float32), (1, 0, 2))
        scale = self.head_dim ** -0.5
        attn = mx.softmax(qh @ _transpose(kh, (0, 2, 1)) * scale, axis=-1)
        o = _transpose(attn @ vh, (1, 0, 2)).reshape(n, -1)
        return self.wo(o)


class DeepseekV41VisionMLP(nn.Module):
    def __init__(self, vision: VisionConfig):
        super().__init__()
        dim = int(vision.hidden_size)
        inter = int(vision.intermediate_size)
        self.w1 = _VisionLinear(dim, 2 * inter, bias=False)
        self.w2 = _VisionLinear(inter, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.w1(x)
        gate, up = h[:, : h.shape[-1] // 2], h[:, h.shape[-1] // 2 :]
        return self.w2(_silu(gate) * up)


class DeepseekV41VisionBlock(nn.Module):
    def __init__(self, vision: VisionConfig):
        super().__init__()
        dim = int(vision.hidden_size)
        self.norm1 = _VisionRMSNorm(dim)
        self.attn = DeepseekV41VisionAttention(vision)
        self.norm2 = _VisionRMSNorm(dim)
        self.mlp = DeepseekV41VisionMLP(vision)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))

class DeepseekV41ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, vision: VisionConfig):
        super().__init__()
        n_layers = int(vision.num_hidden_layers)
        n_heads = int(vision.num_attention_heads)
        dim = int(vision.hidden_size)
        if n_layers <= 0:
            raise ValueError("ViT requires num_hidden_layers > 0")
        if dim % n_heads != 0:
            raise ValueError(
                f"vision hidden_size {dim} is not divisible by n_heads {n_heads}"
            )
        self.rope_dim = dim // n_heads // 2
        self.rope_theta = float(vision.rope_theta)
        self.patch_embed = DeepseekV41PatchEmbed(vision)
        self.blocks = [DeepseekV41VisionBlock(vision) for _ in range(n_layers)]
        self.norm = _VisionRMSNorm(dim)

    def __call__(self, patches: mx.array, n_h: int, n_w: int) -> mx.array:
        if n_h < 1 or n_w < 1:
            raise ValueError(f"malformed ViT grid {n_h}x{n_w}")
        x = self.patch_embed(patches)
        if int(x.shape[0]) != n_h * n_w:
            raise ValueError(
                f"patch count {x.shape[0]} does not match ViT grid {n_h}x{n_w}"
            )
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV41Aligner(nn.Module):
    def __init__(self, vision: VisionConfig, dim: int):
        super().__init__()
        self.downsample_ratio = int(vision.downsample_ratio)
        if self.downsample_ratio < 1:
            raise ValueError(
                f"downsample_ratio must be positive, got {self.downsample_ratio}"
            )
        in_dim = int(vision.hidden_size) * self.downsample_ratio ** 2
        if dim <= 0:
            raise ValueError(f"aligner output dim must be positive, got {dim}")
        self.w1 = _VisionLinear(in_dim, dim, bias=True)
        self.w2 = _VisionLinear(dim, dim, bias=True)

    def __call__(self, x: mx.array, n_h: int, n_w: int) -> mx.array:
        r = self.downsample_ratio
        if n_h < 1 or n_w < 1:
            raise ValueError(f"malformed aligner grid {n_h}x{n_w}")
        if int(x.shape[0]) != n_h * n_w:
            raise ValueError(
                f"aligner tokens {x.shape[0]} do not match ViT grid {n_h}x{n_w}"
            )
        c = int(x.shape[-1])
        grid = _transpose(x.reshape(n_h, n_w, c), (2, 0, 1))
        pad_w = (-n_w) % r
        pad_h = (-n_h) % r
        if pad_h:
            grid = mx.concatenate(
                [grid, mx.zeros((c, pad_h, grid.shape[-1]), dtype=mx.float32)],
                axis=1,
            )
        if pad_w:
            grid = mx.concatenate(
                [grid, mx.zeros((c, grid.shape[1], pad_w), dtype=mx.float32)],
                axis=2,
            )
        c, h, w = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])
        oh, ow = h // r, w // r
        tiles = _transpose(grid.reshape(c, oh, r, ow, r), (1, 3, 0, 2, 4))
        flat = tiles.reshape(oh * ow, c * r * r)
        return self.w2(_gelu(self.w1(flat)))

class DeepseekV41Vision(nn.Module):
    """Official vision tower + aligner + learned image-span delimiters.

    Parameter names match the public checkpoint: vision.*, aligner.*, and
    image_start / image_end / image_newline with no .weight suffix.
    """

    def __init__(self, vision: VisionConfig, dim: int):
        super().__init__()
        if not vision_enabled(vision):
            raise ValueError(
                "vision tower is disabled (num_hidden_layers == 0); "
                "constructing DeepseekV41Vision for a text-only config is a wiring error"
            )
        if dim <= 0:
            raise ValueError(f"model hidden size must be positive, got {dim}")
        self.vision_config = vision
        self.dim = int(dim)
        self.vision = DeepseekV41ViT(vision)
        self.aligner = DeepseekV41Aligner(vision, self.dim)
        self.image_start = mx.zeros((self.dim,))
        self.image_end = mx.zeros((self.dim,))
        self.image_newline = mx.zeros((self.dim,))

    def encode_image(self, patches, n_vit_h: int, n_vit_w: int) -> mx.array:
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images, h: mx.array) -> mx.array:
        """Overwrite each image span in h with ViT/aligner features.

        IMAGE slots take aligner rows in row-major order; span delimiters take
        the learned embeddings. Text-only (images is None or empty) is identity.
        h is the pre-Hyper-Connection stream [batch, seq, dim].
        """
        if images is None:
            return h
        h_np = np.array(_as_numpy(h), copy=True)
        if h_np.ndim != 3:
            raise ValueError(
                f"merge_image_embeddings expects [batch, seq, dim], got {h_np.shape}"
            )
        if h_np.shape[-1] != self.dim:
            raise ValueError(
                f"stream width {h_np.shape[-1]} does not match aligner dim {self.dim}"
            )
        batch, seqlen, _ = h_np.shape
        samples = images
        if samples and not isinstance(samples[0], (list, tuple)):
            samples = [samples]
        if len(samples) != batch:
            raise ValueError(
                f"got {len(samples)} image batches for stream batch {batch}"
            )
        start_emb = np.asarray(self.image_start, dtype=np.float32)
        end_emb = np.asarray(self.image_end, dtype=np.float32)
        nl_emb = np.asarray(self.image_newline, dtype=np.float32)
        for i, sample in enumerate(samples):
            for img in sample or ():
                types = np.asarray(img.types).reshape(-1)
                if types.size == 0:
                    raise ValueError("image span types must be non-empty")
                unknown = set(int(t) for t in types.tolist()) - _IMAGE_TOKEN_TYPES
                if unknown:
                    raise ValueError(f"unsupported image token types {sorted(unknown)}")
                n_llm_h = math.ceil(int(img.n_vit_h) / self.aligner.downsample_ratio)
                n_llm_w = math.ceil(int(img.n_vit_w) / self.aligner.downsample_ratio)
                expected_types = np.asarray(image_token_types(n_llm_h, n_llm_w))
                if not np.array_equal(types, expected_types):
                    raise ValueError(
                        f"image span types do not match canonical {n_llm_h}x{n_llm_w} grid"
                    )
                start = int(img.start)
                end = start + int(types.size)
                if start < 0 or end > seqlen:
                    raise ValueError(
                        f"image span [{start}:{end}] overruns sequence length {seqlen}"
                    )
                n_image = int(np.sum(types == IMAGE))
                embeds = np.asarray(
                    self.encode_image(img.patches, int(img.n_vit_h), int(img.n_vit_w))
                )
                if embeds.ndim != 2 or embeds.shape[-1] != self.dim:
                    raise ValueError(
                        f"aligner rows must be [n, {self.dim}], got {embeds.shape}"
                    )
                if embeds.shape[0] != n_image:
                    raise ValueError(
                        f"aligner produced {embeds.shape[0]} rows but the span has "
                        f"{n_image} IMAGE slots"
                    )
                span = h_np[i, start:end]
                span[types == IMAGE_START] = start_emb
                span[types == IMAGE_END] = end_emb
                span[types == IMAGE_NEW_LINE] = nl_emb
                span[types == IMAGE] = embeds.astype(span.dtype, copy=False)
                h_np[i, start:end] = span
        return mx.array(h_np)


class DeepseekV41Block(DeepseekV41HyperConnections):
    """One exact backbone block over the Hyper-Connection residual stream."""

    def __init__(
        self,
        config: TextConfig,
        policy: AttentionLayerPolicy,
        vision_on: bool,
    ):
        super().__init__(config)
        self.layer_id = policy.layer_id
        self.attn = DeepseekV41Attention(config, policy)
        self.ffn = DeepseekV41MoE(config, vision_enabled=vision_on)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.engram: Optional[DeepseekV41Engram] = None

    def inject_engram(self, x, hash_ids, token_mask=None):
        if self.engram is None:
            return x
        if hash_ids is None:
            raise ValueError(
                f"Engram layer {self.layer_id} requires hash ids from the bound "
                "EngramNgramHasher"
            )
        return self.engram(x, hash_ids, token_mask)

    def __call__(self, x, pre_mix, cache, image_mask=None):
        x, attn_pre = self.sublayer_step(
            x,
            pre_mix,
            lambda value: self.attn(self.attn_norm(value), cache),
            "attn",
        )
        return self.sublayer_step(
            x,
            attn_pre,
            lambda value: self.ffn(self.ffn_norm(value), image_mask),
            "ffn",
        )


class DeepseekV41Transformer(nn.Module):
    """Embed, optional vision/Engram, backbone blocks, collapse, and logits."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        config = args.text_config
        self.config = config
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.engram_layout = validate_engram_config(config)
        self.engram_hash: Optional[EngramNgramHasher] = None
        self.embed = DeepseekV41DSparkEmbedding(config.vocab_size, config.hidden_size)
        policies = resolve_attention_layer_policies(config)
        vision_on = vision_enabled(args)
        self.layers = [
            DeepseekV41Block(config, policy, vision_on) for policy in policies
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = DeepseekV41DSparkHead(config.vocab_size, config.hidden_size)
        self.mtp = [
            DeepseekV41DSparkBlock(
                config,
                config.num_hidden_layers + stage_id,
                vision_enabled=vision_on,
            )
            for stage_id in range(config.num_nextn_predict_layers)
        ]
        if vision_on:
            vision_runtime = DeepseekV41Vision(args.vision_config, config.hidden_size)
            self.vision = vision_runtime.vision
            self.aligner = vision_runtime.aligner
            self.image_start = vision_runtime.image_start
            self.image_end = vision_runtime.image_end
            self.image_newline = vision_runtime.image_newline

    def make_cache(self) -> List[DeepseekV41AttentionCache]:
        return make_deepseek_v41_attention_caches(
            self.config, len(self.layers)
        )

    def bind_engram_modules(self, modules: Dict[int, DeepseekV41Engram]) -> None:
        if self.engram_layout is None:
            raise ValueError("cannot bind Engram to a config with no Engram layers")
        expected = set(self.engram_layout.layer_ids)
        if set(modules) != expected:
            raise ValueError(
                f"Engram modules must cover exactly layers {sorted(expected)}, "
                f"got {sorted(modules)}"
            )
        for layer_id, module in modules.items():
            if module.layer_id != layer_id:
                raise ValueError(
                    f"Engram module key {layer_id} carries layer {module.layer_id}"
                )
            self.layers[layer_id].engram = module

    def bind_engram_hasher(self, hasher: EngramNgramHasher) -> None:
        if self.engram_layout is None or hasher.layout != self.engram_layout:
            raise ValueError("Engram hasher layout does not match the model config")
        self.engram_hash = hasher

    def bind_engram(
        self,
        hasher: EngramNgramHasher,
        modules: Dict[int, DeepseekV41Engram],
    ) -> None:
        self.bind_engram_modules(modules)
        self.bind_engram_hasher(hasher)

    def encode_image(self, patches, n_vit_h: int, n_vit_w: int) -> mx.array:
        if not hasattr(self, "vision"):
            raise ValueError("image input was supplied to a text-only model")
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images, hidden: mx.array) -> mx.array:
        if not hasattr(self, "vision"):
            if images is None or all(not sample for sample in images):
                return hidden
            raise ValueError("image input was supplied to a text-only model")
        return DeepseekV41Vision.merge_image_embeddings(self, images, hidden)

    def forward_main(self, input_ids, cache=None, images=None, token_types=None):
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seqlen], got {input_ids.shape}")
        if cache is None:
            cache = self.make_cache()
        if len(cache) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} layer caches, got {len(cache)}"
            )
        start_pos = cache[0].offset
        if any(layer_cache.offset != start_pos for layer_cache in cache):
            raise ValueError("all backbone layer caches must have the same offset")
        image_mask = None
        if token_types is not None:
            if tuple(token_types.shape) != tuple(input_ids.shape):
                raise ValueError(
                    f"token_types shape {token_types.shape} must match input_ids "
                    f"shape {input_ids.shape}"
                )
            image_mask = token_types >= 0
        engram_mask = None if image_mask is None else ~image_mask
        hashes = None
        if self.engram_layout is not None:
            if self.engram_hash is None:
                raise RuntimeError(
                    "Engram is configured but not bound to file-backed row stores"
                )
            hashes = self.engram_hash(input_ids, start_pos, engram_mask)

        hidden = self.embed(input_ids)
        if images is not None:
            if start_pos != 0:
                raise ValueError("image spans must be prefilled at cache offset zero")
            hidden = self.merge_image_embeddings(images, hidden)
        hidden = expand_hyper_connection_stream(hidden, self.hc_mult)
        pre_mix = make_identity_pre_mix(
            hidden.shape[0], hidden.shape[1], self.hc_mult
        )
        main_hiddens = []
        for layer_id, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            layer_hashes = None
            if layer.engram is not None:
                layer_hashes = hashes[:, :, layer.engram.layer_hash_index, :]
            hidden = layer.inject_engram(hidden, layer_hashes, engram_mask)
            if layer_id in self.target_layer_ids:
                main_hiddens.append(mx.mean(hidden, axis=2))
            hidden, pre_mix = layer(hidden, pre_mix, layer_cache, image_mask)

        hidden = hc_pre(hidden, pre_mix)
        logits = self.head(self.norm(hidden), full_logits=True)
        main_hidden = (
            mx.concatenate(main_hiddens, axis=-1) if main_hiddens else None
        )
        return logits, main_hidden

    def __call__(self, input_ids, cache=None, images=None, token_types=None):
        logits, _ = self.forward_main(input_ids, cache, images, token_types)
        return logits

    def forward_spec(self, input_ids, main_hidden, start_pos: int = 0):
        if not self.mtp:
            raise ValueError("DSpark is disabled for this model")
        x, main_x = self.mtp[0].forward_embed(main_hidden, input_ids, self.embed)
        pre_mix = make_identity_pre_mix(x.shape[0], x.shape[1], self.hc_mult)
        for layer in self.mtp:
            x, pre_mix = layer(x, start_pos, pre_mix, main_x)
        if start_pos == 0:
            return None
        return self.mtp[-1].forward_head(x, pre_mix, input_ids, self.head)


class Model(nn.Module):
    """Registered exact V4.1 model with public checkpoint parameter names."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        transformer = DeepseekV41Transformer(args)
        self.embed = transformer.embed
        self.layers = transformer.layers
        self.norm = transformer.norm
        self.head = transformer.head
        self.mtp = transformer.mtp
        self._runtime = transformer
        if hasattr(transformer, "vision"):
            self.vision = transformer.vision
            self.aligner = transformer.aligner
            self.image_start = transformer.image_start
            self.image_end = transformer.image_end
            self.image_newline = transformer.image_newline

    def _sync_runtime(self):
        self._runtime.embed = self.embed
        self._runtime.layers = self.layers
        self._runtime.norm = self.norm
        self._runtime.head = self.head
        self._runtime.mtp = self.mtp
        for name in ("vision", "aligner", "image_start", "image_end", "image_newline"):
            if hasattr(self, name):
                setattr(self._runtime, name, getattr(self, name))

    def __call__(self, inputs, cache=None, images=None, token_types=None):
        self._sync_runtime()
        return self._runtime(inputs, cache, images, token_types)

    def forward_main(self, inputs, cache=None, images=None, token_types=None):
        self._sync_runtime()
        return self._runtime.forward_main(inputs, cache, images, token_types)

    def forward_spec(self, input_ids, main_hidden, start_pos: int = 0):
        self._sync_runtime()
        return self._runtime.forward_spec(input_ids, main_hidden, start_pos)

    def make_cache(self):
        return self._runtime.make_cache()

    def shard(self, group=None):
        group = group or mx.distributed.init()
        self._sync_runtime()
        self._shard_world_size = group.size()
        self._shard_rank = group.rank()
        self._shard_group = group
        for layer in self.layers:
            layer.ffn.shard(group)
        for layer in self.mtp:
            layer.ffn.shard(group)

    def prepare_sharded_load(self, group):
        self.shard(group)

    def bind_engram(self, hasher, modules):
        self._runtime.bind_engram(hasher, modules)

    def bind_tokenizer(self, tokenizer, max_batch_size: int = 1):
        layout = self._runtime.engram_layout
        if layout is None:
            return
        expected = self.args.text_config.engram_compressed_vocab_size
        token_map, compressed_size = (
            build_engram_compressed_token_map_from_tokenizer(
                tokenizer,
                expected_size=expected,
                fallback_token_id=self.args.text_config.engram_pad_token_id,
            )
        )
        if compressed_size != expected:
            raise ValueError(
                f"tokenizer produces {compressed_size} compressed Engram tokens, "
                f"but the checkpoint config requires {expected}"
            )
        hasher = EngramNgramHasher(
            layout,
            token_map,
            self.args.text_config.engram_pad_token_id,
            compressed_size,
            max_batch_size=max_batch_size,
            max_seq_len=self.args.text_config.max_position_embeddings,
        )
        self._runtime.bind_engram_hasher(hasher)

    def prepare_file_backed_weights(self, model_path, weight_files):
        """Bind official Engram tables by path and exclude their payloads from MLX."""
        layout = self._runtime.engram_layout
        world_size = getattr(self, "_shard_world_size", 1)
        rank = getattr(self, "_shard_rank", 0)
        group = getattr(self, "_shard_group", None)
        if layout is None and world_size == 1:
            return {}
        weight_files = [Path(path).resolve() for path in weight_files]
        tensor_files = {}
        for path in weight_files:
            with open(path, "rb") as handle:
                raw_len = handle.read(8)
                if len(raw_len) != 8:
                    raise ValueError(f"{path} has no safetensors header length")
                header_len = int.from_bytes(raw_len, "little", signed=False)
                if header_len < 2 or header_len > _SAFETENSORS_HEADER_LIMIT:
                    raise ValueError(
                        f"{path} declares an implausible safetensors header length"
                    )
                header = json.loads(handle.read(header_len).decode("utf-8"))
            for name in header:
                if name != "__metadata__":
                    if name in tensor_files:
                        raise ValueError(f"duplicate checkpoint tensor {name!r}")
                    tensor_files[name] = path

        excluded_by_file = {}
        if world_size > 1:
            for name, path in tensor_files.items():
                match = re.match(
                    r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.", name
                )
                if match is None:
                    continue
                blocks = self.layers if match.group(1) == "layers" else self.mtp
                block_id = int(match.group(2))
                expert_id = int(match.group(3))
                if block_id >= len(blocks):
                    raise ValueError(f"checkpoint tensor {name!r} names no model block")
                experts = blocks[block_id].ffn.experts
                if expert_id >= len(experts):
                    raise ValueError(f"checkpoint tensor {name!r} names no model expert")
                if experts[expert_id] is None:
                    excluded_by_file.setdefault(str(path), set()).add(name)
        if layout is None:
            return excluded_by_file
        modules = {}
        for layer_id, num_embeddings in zip(
            layout.layer_ids, layout.num_embeddings
        ):
            prefix = f"layers.{layer_id}.engram.embed"
            weight_key, scale_key = f"{prefix}.weight", f"{prefix}.scale"
            missing = [key for key in (weight_key, scale_key) if key not in tensor_files]
            if missing:
                raise ValueError(
                    f"checkpoint is missing file-backed Engram tensors {missing}"
                )
            path = tensor_files[weight_key]
            if tensor_files[scale_key] != path:
                raise ValueError(
                    f"{weight_key} and {scale_key} must share one safetensors file"
                )
            store = SafetensorsEngramRowStore(
                str(path),
                weight_key=weight_key,
                scale_key=scale_key,
                row_start=rank * (-(-num_embeddings // world_size)),
                num_rows=-(-num_embeddings // world_size),
            )
            if store.full_num_rows != num_embeddings or store.dim != layout.head_dim:
                store.close()
                raise ValueError(
                    f"{weight_key} has shape {(store.full_num_rows, store.dim)} but "
                    f"the config requires {(num_embeddings, layout.head_dim)}"
                )
            cache = BoundedEngramRowCache(store)
            embedding = DeepseekV41EngramEmbedding(
                num_embeddings,
                layout.head_dim,
                cache,
                rank=rank,
                world_size=world_size,
                all_reduce=(
                    None
                    if world_size == 1
                    else lambda value: mx.distributed.all_sum(value, group=group)
                ),
            )
            modules[layer_id] = DeepseekV41Engram(
                self.args.text_config, layer_id, layout, embedding
            )
            excluded_by_file.setdefault(str(path), set()).update(
                (weight_key, scale_key)
            )
        self._runtime.bind_engram_modules(modules)
        self._sync_runtime()
        return excluded_by_file

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Apply the mandatory wo_a FP8-at-rest -> dense-bf16 exception.

        Every other packed weight is passed through unchanged. Fails closed if a wo_a.weight is
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
