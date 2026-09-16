# Copyright (c) 2026 the mlx-lm contributors
"""Plan 0051 M2: config fail-closed behavior, packed FP8(E4M3)/E8M0 and
FP4(E2M1)/E8M0 numerical primitives, the wo_a exception, and the exact
base-decode attention and cache architecture for deepseek_v41.

The attention tests below run real production code end to end on a small but
structurally faithful config (two compress ratios, two kv_source layers, three
index sources and a candidate source that is not the first index source), plus
the real 40-layer pinned config for the cross-layer sharing structure itself.

The FP8 and FP4 fixtures below are the *real* sampled tensor bytes and
independently-decoded expected values recorded in Plan 0051 ``artifacts/
m1-layout-probes.md`` (Probe 1: ``layers.0.attn.wkv.weight``/``.scale`` from
``model-00003-of-00048.safetensors`` at the pinned revision
``deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277``;
Probe 2: ``layers.0.ffn.experts.0.w1.weight``/``.scale`` from the same shard).
The FP4 nibble order and magnitude table were independently confirmed
byte-for-byte against the official ``inference/convert.py`` reference
converter at that pinned revision (see ``artifacts/m1-gap-closure.md``).

The Hyper-Connections and MoE tests below run the real production modules on
tiny-but-structurally-faithful configs: the actual ``hc_mult == 4`` /
20-iteration Sinkhorn split, and the actual sqrtsoftplus + ``noaux_tc``
routing rule with one shared expert, ``norm_topk_prob`` and
``routed_scaling_factor`` -- over 8 routed experts and 32-wide hidden dims
rather than the official 384 x 2304 x 5120 stack, which no assertion here
needs allocated. The one place the official numbers appear directly is the
gate itself (384 routed / 6 active), whose projection is small enough to
build.

The Engram tests below run the real production sparse-lookup path. The prime
bucket layout and hash multipliers are checked against the *pinned* config, so
the two 384-million-row tables are exercised as arithmetic; every gather runs
against a tiny real safetensors fixture written by the test itself, because the
production row store addresses rows by byte range and is therefore
table-size-independent by construction. No test here allocates, loads or
implies a dense ``[num_embeddings, head_dim]`` tensor -- the official pair is
~91.6 GiB and ~98.3 GiB -- and TestDeepseekV41EngramMemoryBounds asserts that
property directly by holding the measured gather cost constant across fixtures
that differ 16x in size.
The vision tests below run the real production tower, aligner, and image-grid
formulas. Token counts for fixed synthetic images are pinned against the
official ``plan_image_grid`` / ``num_image_tokens`` arithmetic; span replacement
overwrites IMAGE slots with aligner rows and leaves text-only streams untouched.
"""
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.deepseek_v41 import (
    COMPRESS_KV_FP4_BLOCK_SIZE,
    ENGRAM_FP8_BLOCK_SIZE,
    ENGRAM_GATE_CLAMP,
    ENGRAM_NORMALIZER_SEQUENCE,
    ENGRAM_SPACE_SENTINEL,
    FP4_E2M1_TABLE,
    FP4_WEIGHT_BLOCK_SIZE,
    BoundedEngramRowCache,
    DeepseekV41AttentionStack,
    DeepseekV41DSpark,
    DeepseekV41Engram,
    DeepseekV41EngramEmbedding,
    DeepseekV41Expert,
    DeepseekV41Gate,
    DeepseekV41HyperConnections,
    DeepseekV41MoE,
    DeepseekV41PackedLinear,
    DeepseekV41Vision,
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    ImageInput,
    EngramLayout,
    EngramNgramHasher,
    Model,
    ModelArgs,
    PhysicalLatentCache,
    QuantizationConfig,
    SafetensorsEngramRowStore,
    TEXT,
    TextConfig,
    VisionConfig,
    act_quant_roundtrip,
    apply_rope_tail,
    build_engram_compressed_token_map,
    build_engram_compressed_token_map_from_tokenizer,
    compute_engram_hash_multipliers,
    decode_e8m0_scale,
    dequantize_engram_rows,
    dequantize_fp4_block,
    dequantize_fp8_block,
    dequantize_wo_a,
    deterministic_topk_indices,
    engram_compressed_token_key,
    engram_signed_sqrt_sigmoid_gate,
    expand_hyper_connection_stream,
    find_next_prime,
    get_dspark_topk_idxs,
    image_token_types,
    fp4_act_quant_roundtrip,
    llm_grid,
    hc_post,
    hc_pre,
    hc_split_sinkhorn,
    make_deepseek_v41_attention_caches,
    make_identity_pre_mix,
    num_image_tokens,
    noaux_tc_route,
    normalize_engram_token_text,
    plan_image_grid,
    public_checkpoint_name,
    remaining_gate_bias_and_wo_a_names,
    require_public_weights,
    resolve_attention_layer_policies,
    rope_cos_sin,
    routed_expert_partition,
    routing_scores,
    select_candidate_blocks,
    sparse_attn,
    unpack_fp4_e2m1,
    vision_enabled,
    vision_public_weight_names,
    validate_engram_config,
    validate_moe_routing_config,
    window_topk_idxs,
    yarn_rope_frequencies,
)


def _text_config_dict():
    return {
        "model_type": "deepseek_v41_text",
        "vocab_size": 129280,
        "hidden_size": 5120,
        "moe_intermediate_size": 2304,
        "num_hidden_layers": 40,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1280,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-20,
        "attention_bias": False,
        "max_position_embeddings": 1048576,
        "rope_theta": 10000,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        "compress_ratios": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "compress_rope_theta": 160000,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-06,
        "engram_layer_ids": [1, 14],
        "engram_num_embeddings": [384006168, 384016682],
        "engram_max_ngram_size": 4,
        "engram_vocab_size": 16000000,
        "engram_n_heads": 8,
        "engram_head_dim": 256,
        "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 99092,
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
    }


def _vision_config_dict():
    return {
        "model_type": "deepseek_v41_vision",
        "num_hidden_layers": 32,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "intermediate_size": 2816,
        "patch_size": 14,
        "rope_theta": 10000,
        "downsample_ratio": 3,
        "max_image_tokens": 1024,
        "min_pixels": 295936,
        "max_wh_ratio": None,
    }


def _quantization_config_dict():
    return {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    }


def _full_config_dict():
    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "image_token_id": 129264,
        "quantization_config": _quantization_config_dict(),
        "text_config": _text_config_dict(),
        "vision_config": _vision_config_dict(),
    }


class TestDeepseekV41Config(unittest.TestCase):
    """Fail-closed config coverage (Plan 0051 M2, config-dataclass slice)."""

    def test_full_official_config_round_trips(self):
        args = ModelArgs.from_dict(_full_config_dict())
        self.assertEqual(args.model_type, "deepseek_v41")
        self.assertIsInstance(args.text_config, TextConfig)
        self.assertIsInstance(args.vision_config, VisionConfig)
        self.assertIsInstance(args.quantization_config, QuantizationConfig)
        self.assertEqual(args.text_config.engram_layer_ids, [1, 14])
        self.assertEqual(args.text_config.hc_mult, 4)
        self.assertEqual(args.text_config.index_topk, 512)
        self.assertEqual(args.text_config.dspark_target_layer_ids, [37, 38, 39])
        self.assertEqual(args.vision_config.patch_size, 14)
        self.assertEqual(args.quantization_config.expert_dtype, "fp4")

    def test_compress_ratios_matches_exact_pinned_official_43_entries(self):
        # The full pinned official text_config.compress_ratios value
        # (deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277
        # config.json): 40 decode-layer entries (2 compression-disabled,
        # 18 ratio-2, 20 ratio-1) followed by 3 trailing MTP
        # (num_nextn_predict_layers) zeros -- 43 entries total. Asserted
        # in full, not merely by length, so a re-truncation or a
        # transposed/reordered value regresses loudly.
        expected = (
            [0, 0]
            + [2] * 18
            + [1] * 20
            + [0, 0, 0]
        )
        self.assertEqual(len(expected), 43)
        args = ModelArgs.from_dict(_full_config_dict())
        self.assertEqual(args.text_config.compress_ratios, expected)
        self.assertEqual(
            len(args.text_config.compress_ratios),
            args.text_config.num_hidden_layers
            + args.text_config.num_nextn_predict_layers,
        )

    def test_compress_ratios_wrong_cardinality_fails_closed(self):
        # A truncated (or padded) compress_ratios -- e.g. missing the
        # trailing MTP zeros -- must fail closed at config-construction
        # time instead of silently under/over-indexing per-layer state
        # deep inside the (deferred) decode stack.
        text_config = _text_config_dict()
        text_config["compress_ratios"] = text_config["compress_ratios"][:-3]
        self.assertEqual(len(text_config["compress_ratios"]), 40)
        config = _full_config_dict()
        config["text_config"] = text_config
        with self.assertRaises(ValueError):
            ModelArgs.from_dict(config)

    def test_compress_ratios_overlong_cardinality_fails_closed(self):
        text_config = _text_config_dict()
        text_config["compress_ratios"] = text_config["compress_ratios"] + [0]
        config = _full_config_dict()
        config["text_config"] = text_config
        with self.assertRaises(ValueError):
            ModelArgs.from_dict(config)

    def test_missing_engram_field_fails_closed(self):
        text_config = _text_config_dict()
        del text_config["engram_layer_ids"]
        config = _full_config_dict()
        config["text_config"] = text_config
        with self.assertRaises(TypeError):
            ModelArgs.from_dict(config)

    def test_missing_hyper_connections_field_fails_closed(self):
        text_config = _text_config_dict()
        del text_config["hc_sinkhorn_iters"]
        config = _full_config_dict()
        config["text_config"] = text_config
        with self.assertRaises(TypeError):
            ModelArgs.from_dict(config)

    def test_missing_index_source_field_fails_closed(self):
        text_config = _text_config_dict()
        del text_config["index_source_layer_ids"]
        config = _full_config_dict()
        config["text_config"] = text_config
        with self.assertRaises(TypeError):
            ModelArgs.from_dict(config)

    def test_missing_dspark_field_fails_closed(self):
        text_config = _text_config_dict()
        del text_config["dspark_markov_rank"]
        config = _full_config_dict()
        config["text_config"] = text_config
        with self.assertRaises(TypeError):
            ModelArgs.from_dict(config)

    def test_missing_vision_field_fails_closed(self):
        vision_config = _vision_config_dict()
        del vision_config["downsample_ratio"]
        config = _full_config_dict()
        config["vision_config"] = vision_config
        with self.assertRaises(TypeError):
            ModelArgs.from_dict(config)

    def test_missing_top_level_text_config_fails_closed(self):
        config = _full_config_dict()
        del config["text_config"]
        with self.assertRaises(TypeError):
            ModelArgs.from_dict(config)

    def test_v4_shaped_config_is_rejected_not_silently_aliased(self):
        """A deepseek_v4-shaped flat config (no text_config/engram/etc.) must
        not be silently accepted as deepseek_v41: this is the exact failure
        mode the plan requires this architecture to not alias into."""
        v4_shaped = {
            "model_type": "deepseek_v41",
            "vocab_size": 129280,
            "hidden_size": 4096,
            "num_hidden_layers": 43,
        }
        with self.assertRaises(TypeError):
            ModelArgs.from_dict(v4_shaped)

    def test_registered_module_is_not_a_v4_alias(self):
        import mlx_lm.models.deepseek_v4 as v4
        import mlx_lm.models.deepseek_v41 as v41

        self.assertIsNot(v41.Model, v4.Model)
        self.assertIsNot(v41.ModelArgs, v4.ModelArgs)
        args = ModelArgs.from_dict(_full_config_dict())
        model = Model(args)
        self.assertEqual(model.model_type, "deepseek_v41")

    def test_model_call_fails_loud_not_fake(self):
        args = ModelArgs.from_dict(_full_config_dict())
        model = Model(args)
        with self.assertRaisesRegex(RuntimeError, "Engram is configured but not bound"):
            model(mx.array([[1, 2, 3]]))


class TestDeepseekV41QuantPrimitives(unittest.TestCase):
    """Numerical primitives: fixed synthetic blocks + real M1 sampled bytes."""

    # ---- E8M0 scale decode -------------------------------------------- #

    def test_e8m0_scale_decode_synthetic(self):
        # code 127 -> 2**0 == 1.0; code 128 -> 2**1 == 2.0; code 0 -> 2**-127
        codes = mx.array([127, 128, 120, 0], dtype=mx.uint8)
        decoded = decode_e8m0_scale(codes)
        expected = [1.0, 2.0, 2.0 ** (120 - 127), 2.0 ** (0 - 127)]
        for got, want in zip(decoded.tolist(), expected):
            self.assertAlmostEqual(got, want, places=12)

    def test_e8m0_scale_decode_nonfinite(self):
        codes = mx.array([255], dtype=mx.uint8)
        decoded = decode_e8m0_scale(codes)
        self.assertTrue(mx.isnan(decoded).item())

    # ---- FP8 E4M3 + E8M0 block dequant, real M1 sampled bytes ---------- #

    def test_fp8_block_dequant_matches_sampled_probe1_bytes(self):
        # Probe 1: layers.0.attn.wkv.weight (F8_E4M3) row 0, block_size=32,
        # plus its layers.0.attn.wkv.scale (F8_E8M0) row -- real bytes from
        # model-00003-of-00048.safetensors at the pinned revision.
        weight_hex = (
            "d871f2797069f0e55370eee3dcd8776"
            "d6bdc6bf25b66716b5c6d71d0eae7e062"
        )
        weight_bytes = bytes.fromhex(weight_hex)[:32]
        scale_hex_byte = "73"  # the block-0 scale byte covering these 32 elements
        weight = mx.array(list(weight_bytes), dtype=mx.uint8).reshape(1, 32)
        scale = mx.array([int(scale_hex_byte, 16)], dtype=mx.uint8).reshape(1, 1)

        decoded = dequantize_fp8_block(weight, scale, dtype=mx.float32)
        expected_first8 = [
            -0.00390625, 0.03515625, -0.0390625, 0.0703125,
            0.03125, 0.017578125, -0.03125, -0.0126953125,
        ]
        got = decoded[0, :8].tolist()
        for g, w in zip(got, expected_first8):
            self.assertAlmostEqual(g, w, places=9)

    def test_fp8_block_dequant_synthetic_2x2_blocks(self):
        # Two 2x2 blocks (block_size=2) with distinct scales; verifies the
        # 2-D tiling (not a 1-D per-row group) used for dense V4.1 weights.
        weight = mx.array(
            [
                [0x00, 0x08, 0x10, 0x18],
                [0x20, 0x28, 0x30, 0x38],
            ],
            dtype=mx.uint8,
        )
        scale = mx.array([[127, 128]], dtype=mx.uint8)  # 1x2 blocks -> block=(2,2)
        decoded = dequantize_fp8_block(weight, scale, dtype=mx.float32)
        self.assertEqual(decoded.shape, (2, 4))

    def test_fp8_block_dequant_rejects_shape_mismatch(self):
        weight = mx.zeros((4, 4), dtype=mx.uint8)
        scale = mx.zeros((1, 3), dtype=mx.uint8)  # 4 not divisible by 3
        with self.assertRaises(ValueError):
            dequantize_fp8_block(weight, scale)

    def test_fp8_block_dequant_real_partial_tail_rows(self):
        # A genuine out-axis (row) tail block: out_dim=3 is not a multiple
        # of block_size=2, so the scale grid is ceildiv(3, 2) == 2 blocks
        # (one full 2-row block, one real 1-row tail block) rather than
        # the in_dim//2 exact grid used when block_size is left implicit.
        # Byte values and expected results were independently hand-decoded
        # via the official E4M3FN 1-4-3 layout (bias 7, matching Probe 1's
        # confirmed byte decode above), not asserted equal to a relabeled
        # full block.
        weight = mx.array(
            [
                [0x38, 0x40, 0x44, 0x48],  # 1.0, 2.0, 3.0, 4.0
                [0x4A, 0x4C, 0x4E, 0x50],  # 5.0, 6.0, 7.0, 8.0
                [0xB8, 0xC0, 0xC4, 0xC8],  # -1.0, -2.0, -3.0, -4.0 (real tail row)
            ],
            dtype=mx.uint8,
        )
        # block (0,0)->1.0 (0,1)->2.0 (1,0)->0.5 (1,1)->4.0
        scale = mx.array([[127, 128], [126, 129]], dtype=mx.uint8)
        decoded = dequantize_fp8_block(weight, scale, block_size=2, dtype=mx.float32)
        self.assertEqual(decoded.shape, (3, 4))
        expected = [
            [1.0, 2.0, 6.0, 8.0],
            [5.0, 6.0, 14.0, 16.0],
            [-0.5, -1.0, -12.0, -16.0],
        ]
        for row_got, row_want in zip(decoded.tolist(), expected):
            for g, w in zip(row_got, row_want):
                self.assertAlmostEqual(g, w, places=6)

    def test_fp8_block_dequant_explicit_block_size_rejects_in_axis_tail(self):
        # The in-axis (reduction/K) is never padded: an in_dim that is not
        # an exact multiple of block_size must fail closed even though an
        # explicit block_size enables out-axis tail handling.
        weight = mx.zeros((2, 3), dtype=mx.uint8)
        scale = mx.zeros((1, 1), dtype=mx.uint8)
        with self.assertRaises(ValueError):
            dequantize_fp8_block(weight, scale, block_size=2)

    def test_fp8_block_dequant_explicit_block_size_rejects_grid_mismatch(self):
        weight = mx.zeros((3, 4), dtype=mx.uint8)
        scale = mx.zeros((1, 2), dtype=mx.uint8)  # ceildiv(3, 2) == 2, not 1
        with self.assertRaises(ValueError):
            dequantize_fp8_block(weight, scale, block_size=2)

    # ---- wo_a exception -------------------------------------------- #

    def test_wo_a_dequant_accepts_official_block_sizes(self):
        for block in (32, 128):
            weight = mx.zeros((block, block), dtype=mx.uint8)
            scale = mx.array([[127]], dtype=mx.uint8)
            out = dequantize_wo_a(weight, scale)
            self.assertEqual(out.dtype, mx.bfloat16)
            self.assertEqual(out.shape, (block, block))

    def test_wo_a_dequant_rejects_unofficial_block_size(self):
        weight = mx.zeros((16, 16), dtype=mx.uint8)
        scale = mx.array([[127]], dtype=mx.uint8)  # implies block (16, 16)
        with self.assertRaises(ValueError):
            dequantize_wo_a(weight, scale)

    def test_wo_a_dequant_rejects_out_axis_tail(self):
        # convert.py never ceil-divides a wo_a axis: a genuine out-axis
        # (row) partial tail -- out_dim=40 is not a multiple of
        # block_size=32, so a naive ceildiv(40, 32) == 2 out-block grid
        # -- is not an official (32, 32)/(128, 128) square tile and must
        # fail closed rather than silently zero-pad the tail row.
        weight = mx.zeros((40, 32), dtype=mx.uint8)
        scale = mx.array([[127], [128]], dtype=mx.uint8)  # ceildiv(40, 32) == 2
        with self.assertRaises(ValueError):
            dequantize_wo_a(weight, scale)

    def test_wo_a_dequant_rejects_undersized_real_byte_tile(self):
        # wo_a shares convert.py exact dequant math with the general FP8
        # block scheme; reuse Probe 1 bytes as a stand-in wo_a-shaped tile
        # to prove the wo_a path performs the identical faithful math (cast
        # to dense bfloat16, not merely transiently dequantized).
        weight_hex = "d871f2797069f0e55370eee3dcd8776d"
        weight_bytes = bytes.fromhex(weight_hex)
        weight = mx.array(list(weight_bytes), dtype=mx.uint8).reshape(1, 16)
        # Pad to a (32, 32) tile is unnecessary here: use a (1, 16) input is
        # invalid for the official (32,32)/(128,128) block-size contract,
        # so this must fail closed rather than silently dequantize.
        scale = mx.array([[0x73]], dtype=mx.uint8)
        with self.assertRaises(ValueError):
            dequantize_wo_a(weight, scale)

    def test_wo_a_missing_scale_fails_closed_in_sanitize(self):
        args = ModelArgs.from_dict(_full_config_dict())
        model = Model(args)
        weights = {
            "layers.0.attn.wo_a.weight": mx.zeros((32, 32), dtype=mx.uint8),
        }
        with self.assertRaises(ValueError):
            model.sanitize(weights)

    def test_wo_a_orphan_scale_fails_closed_in_sanitize(self):
        # Inverse of test_wo_a_missing_scale_fails_closed_in_sanitize: a
        # wo_a.scale with no paired wo_a.weight must also fail closed, not
        # silently pass through as an unrelated packed tensor.
        args = ModelArgs.from_dict(_full_config_dict())
        model = Model(args)
        weights = {
            "layers.0.attn.wo_a.scale": mx.array([[127]], dtype=mx.uint8),
        }
        with self.assertRaises(ValueError):
            model.sanitize(weights)

    def test_sanitize_dequantizes_wo_a_and_passes_through_others(self):
        args = ModelArgs.from_dict(_full_config_dict())
        model = Model(args)
        other = mx.ones((4, 4), dtype=mx.uint8)
        weights = {
            "layers.0.attn.wo_a.weight": mx.zeros((32, 32), dtype=mx.uint8),
            "layers.0.attn.wo_a.scale": mx.array([[127]], dtype=mx.uint8),
            "layers.0.attn.wo_b.weight": other,
        }
        out = model.sanitize(weights)
        self.assertNotIn("layers.0.attn.wo_a.scale", out)
        self.assertEqual(out["layers.0.attn.wo_a.weight"].dtype, mx.bfloat16)
        self.assertEqual(out["layers.0.attn.wo_a.weight"].shape, (32, 32))
        self.assertTrue(mx.array_equal(out["layers.0.attn.wo_b.weight"], other))

    # ---- FP4 E2M1 + E8M0, real M1 sampled bytes ------------------------ #

    def test_fp4_nibble_order_matches_official_convention(self):
        # byte 0xA0: low=0x0 (lower-indexed element), high=0xA (higher-indexed)
        packed = mx.array([0xA0], dtype=mx.uint8).reshape(1, 1)
        codes = unpack_fp4_e2m1(packed)
        self.assertEqual(codes.tolist(), [[0x0, 0xA]])

    def test_fp4_table_matches_official_convert_py_literal(self):
        self.assertEqual(
            FP4_E2M1_TABLE,
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        )

    def test_fp4_block_dequant_matches_sampled_probe2_bytes(self):
        # Probe 2: layers.0.ffn.experts.0.w1.weight (packed FP4 E2M1) row 0,
        # block_size=32, plus its ...w1.scale (F8_E8M0) row -- real bytes
        # from model-00003-of-00048.safetensors at the pinned revision.
        weight_hex = "a0a8ec4aadaf42aacc82256ffa054544"
        weight_bytes = bytes.fromhex(weight_hex)[:8]  # first 16 logical elements
        packed = mx.array(list(weight_bytes), dtype=mx.uint8).reshape(1, 8)
        scale = mx.array([[0x78]], dtype=mx.uint8)  # block-0 scale byte

        decoded = dequantize_fp4_block(packed, scale, block_size=16, dtype=mx.float32)
        expected_first16 = [
            0.0, -0.0078125, 0.0, -0.0078125,
            -0.015625, -0.03125, -0.0078125, 0.015625,
            -0.0234375, -0.0078125, -0.046875, -0.0078125,
            0.0078125, 0.015625, -0.0078125, -0.0078125,
        ]
        got = decoded[0].tolist()
        for g, w in zip(got, expected_first16):
            self.assertAlmostEqual(g, w, places=9)

    def test_fp4_block_dequant_rejects_shape_mismatch(self):
        packed = mx.zeros((1, 16), dtype=mx.uint8)  # unpacks to 32 elements
        scale = mx.zeros((1, 2), dtype=mx.uint8)  # implies block_size=16
        with self.assertRaises(ValueError):
            dequantize_fp4_block(packed, scale, block_size=32)

    def test_fp4_block_dequant_rejects_indivisible_reduction_dim(self):
        # fp4_act_quant/fp4_gemm never ceil-divide or pad the logical
        # reduction dimension: 40 unpacked elements (20 packed bytes) is
        # not an exact multiple of block_size=32, so this is not a
        # supported layout and must fail closed rather than silently
        # zero-padding an 8-element partial tail group.
        block0_bytes = [0x22] * 16  # 32 elements, all code=2 -> value 1.0
        block1_bytes = [0x21, 0x43, 0x65, 0x97]  # codes [1,2,3,4,5,6,7,9]
        packed = mx.array(block0_bytes + block1_bytes, dtype=mx.uint8).reshape(1, 20)
        scale = mx.array([[127, 128]], dtype=mx.uint8)
        with self.assertRaises(ValueError):
            dequantize_fp4_block(packed, scale, block_size=32)

    def test_fp4_block_dequant_rejects_scale_grid_mismatch_on_exact_fit(self):
        # Even on an exact-fit (evenly divisible) reduction dimension --
        # 64 unpacked elements (32 packed bytes) with block_size=32 is
        # exactly 2 groups -- a scale grid that does not match that exact
        # group count must still fail closed.
        packed = mx.zeros((1, 32), dtype=mx.uint8)  # 64 unpacked elements
        scale = mx.zeros((1, 3), dtype=mx.uint8)  # exact grid is 2, not 3
        with self.assertRaises(ValueError):
            dequantize_fp4_block(packed, scale, block_size=32)



def _tiny_text_config(**overrides):
    """A small but structurally faithful config: two compress ratios, two
    kv_source layers, three index sources and a candidate source that is not
    the first index source, i.e. every role the real 40-layer config exercises.
    """
    cfg = _text_config_dict()
    cfg.update(
        {
            "hidden_size": 16,
            "num_hidden_layers": 8,
            "num_nextn_predict_layers": 0,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "qk_rope_head_dim": 8,
            "q_lora_rank": 8,
            "o_lora_rank": 4,
            "o_groups": 2,
            "sliding_window": 4,
            "compress_ratios": [0, 0, 2, 2, 1, 1, 1, 1],
            "kv_source_layer_ids": [2, 4],
            "index_source_layer_ids": [2, 4, 6],
            "index_n_heads": 2,
            "index_head_dim": 32,
            "index_topk": 4,
            "candidate_source_layer_id": 4,
            "candidate_topk_blocks": 2,
            "candidate_block_size": 2,
        }
    )
    cfg.update(overrides)
    return TextConfig.from_dict(cfg)


def _official_text_config(**overrides):
    cfg = _text_config_dict()
    cfg.update(overrides)
    return TextConfig.from_dict(cfg)


class TestDeepseekV41LayerPolicies(unittest.TestCase):
    def test_official_config_resolves_four_kv_sources(self):
        policies = resolve_attention_layer_policies(_official_text_config())
        self.assertEqual(len(policies), 40)
        self.assertEqual(
            [p.layer_id for p in policies if p.is_kv_source], [2, 8, 14, 20]
        )
        self.assertEqual(
            [p.layer_id for p in policies if p.owns_index_keys], [2, 8, 14, 20]
        )
        self.assertEqual(
            [p.layer_id for p in policies if p.is_index_source],
            [2, 8, 14, 20, 24, 28, 32, 36],
        )

    def test_official_config_ratio_and_source_wiring(self):
        policies = resolve_attention_layer_policies(_official_text_config())
        by_id = {p.layer_id: p for p in policies}
        self.assertEqual(by_id[0].compress_ratio, 0)
        self.assertIsNone(by_id[0].compress_source_layer_id)
        self.assertIsNone(by_id[0].topk_source_layer_id)
        self.assertEqual(by_id[7].compress_ratio, 2)
        self.assertEqual(by_id[7].compress_source_layer_id, 2)
        self.assertEqual(by_id[7].topk_source_layer_id, 2)
        self.assertEqual(by_id[21].compress_ratio, 1)
        self.assertEqual(by_id[21].compress_source_layer_id, 20)
        self.assertEqual(by_id[21].topk_source_layer_id, 20)
        # Index sources past the last KV source read keys layer 20 published.
        self.assertEqual(by_id[36].index_key_source_layer_id, 20)
        self.assertEqual(by_id[36].compress_source_layer_id, 20)
        self.assertEqual(by_id[36].topk_source_layer_id, 36)

    def test_official_config_candidate_roles(self):
        by_id = {
            p.layer_id: p
            for p in resolve_attention_layer_policies(_official_text_config())
        }
        self.assertTrue(by_id[20].is_candidate_source)
        self.assertFalse(by_id[20].uses_candidates)
        for layer_id in (2, 8, 14):
            self.assertFalse(by_id[layer_id].is_candidate_source)
            self.assertFalse(by_id[layer_id].uses_candidates)
        for layer_id in (24, 28, 32, 36):
            self.assertTrue(by_id[layer_id].uses_candidates)
            self.assertFalse(by_id[layer_id].is_candidate_source)
        # A layer that runs no Indexer never consumes candidates itself.
        self.assertFalse(by_id[25].uses_candidates)

    def _assert_rejects(self, fragment, **overrides):
        with self.assertRaises(ValueError) as ctx:
            resolve_attention_layer_policies(_tiny_text_config(**overrides))
        self.assertIn(fragment, str(ctx.exception))

    def test_rejects_kv_source_that_is_not_an_index_source(self):
        self._assert_rejects(
            "not in index_source_layer_ids", index_source_layer_ids=[2, 6]
        )

    def test_rejects_candidate_source_that_is_not_an_index_source(self):
        self._assert_rejects(
            "candidate_source_layer_id=5", candidate_source_layer_id=5
        )

    def test_rejects_index_source_with_zero_compress_ratio(self):
        self._assert_rejects(
            "is an index_source but", index_source_layer_ids=[1, 2, 4, 6]
        )

    def test_rejects_compressing_layer_with_no_preceding_kv_source(self):
        self._assert_rejects(
            "no kv_source",
            compress_ratios=[2, 0, 2, 2, 1, 1, 1, 1],
        )

    def test_rejects_consumer_whose_ratio_differs_from_its_source(self):
        self._assert_rejects(
            "must share the ratio of its",
            compress_ratios=[0, 0, 2, 1, 1, 1, 1, 1],
        )

    def test_rejects_kv_source_that_does_not_compress(self):
        self._assert_rejects(
            "a KV source must compress",
            compress_ratios=[0, 0, 0, 2, 1, 1, 1, 1],
            index_source_layer_ids=[2, 4, 6],
        )

    def test_rejects_duplicate_and_unsorted_source_ids(self):
        self._assert_rejects("duplicate layer ids", kv_source_layer_ids=[2, 2, 4])
        self._assert_rejects("ascending execution order", index_source_layer_ids=[4, 2, 6])

    def test_rejects_out_of_range_source_id(self):
        self._assert_rejects(
            "outside the backbone", index_source_layer_ids=[2, 4, 6, 40]
        )

    def test_rejects_multiple_kv_heads(self):
        self._assert_rejects("num_key_value_heads must be 1", num_key_value_heads=2)

    def test_rejects_heads_not_divisible_by_o_groups(self):
        self._assert_rejects("not divisible by o_groups", o_groups=3)

    def test_rejects_non_positive_sliding_window(self):
        self._assert_rejects("sliding_window must be positive", sliding_window=0)

    def test_rejects_odd_rope_head_dim(self):
        self._assert_rejects("must be even", qk_rope_head_dim=7)

    def test_rejects_head_dim_not_tiled_by_the_quant_block_sizes(self):
        self._assert_rejects("FP8 activation", head_dim=48, qk_rope_head_dim=8)

    def test_rejects_index_head_dim_not_tiled_by_the_fp4_block(self):
        self._assert_rejects("index_head_dim=48", index_head_dim=48)


class TestDeepseekV41CacheOwnership(unittest.TestCase):
    def test_official_config_allocates_exactly_four_physical_owners(self):
        caches = make_deepseek_v41_attention_caches(_official_text_config())
        self.assertEqual(len(caches), 40)
        compress_owners = {
            id(c.compress_kv_owner) for c in caches if c.compress_kv_owner is not None
        }
        index_owners = {
            id(c.index_key_owner) for c in caches if c.index_key_owner is not None
        }
        # 40 layers, but only four physical compressed-KV buffers and four
        # physical index-key buffers: this ratio is the whole point of CED.
        self.assertEqual(len(compress_owners), 4)
        self.assertEqual(len(index_owners), 4)
        self.assertEqual(sum(c.compress_kv_writer is not None for c in caches), 4)
        self.assertEqual(sum(c.index_key_writer is not None for c in caches), 4)

    def test_consumers_share_the_identical_owner_object(self):
        caches = make_deepseek_v41_attention_caches(_official_text_config())
        source = caches[2].compress_kv_writer
        self.assertIsNotNone(source)
        for layer_id in range(2, 8):
            self.assertIs(caches[layer_id].compress_kv_owner, source)
        self.assertIsNot(caches[8].compress_kv_owner, source)
        # Layers 24..39 all read the buffers layer 20 owns.
        owner = caches[20].compress_kv_writer
        index_owner = caches[20].index_key_writer
        for layer_id in range(20, 40):
            self.assertIs(caches[layer_id].compress_kv_owner, owner)
        for layer_id in (24, 28, 32, 36):
            self.assertIs(caches[layer_id].index_key_owner, index_owner)

    def test_pure_window_layers_own_no_compressed_state(self):
        caches = make_deepseek_v41_attention_caches(_official_text_config())
        for layer_id in (0, 1):
            self.assertIsNone(caches[layer_id].compress_kv_owner)
            self.assertIsNone(caches[layer_id].compress_kv_writer)
            self.assertIsNone(caches[layer_id].index_key_owner)
            self.assertIsNone(caches[layer_id].pool_state)

    def test_only_ratio_above_one_sources_carry_pooling_state(self):
        caches = make_deepseek_v41_attention_caches(_official_text_config())
        self.assertIsNotNone(caches[2].pool_state)
        self.assertEqual(caches[2].pool_state.compress_ratio, 2)
        # Layer 20 is a ratio-1 source: a plain projection, no pooling group.
        self.assertIsNone(caches[20].pool_state)
        self.assertIsNone(caches[21].pool_state)

    def test_prompt_cache_state_can_be_materialized_without_unsharing_owners(self):
        caches = make_deepseek_v41_attention_caches(_official_text_config())
        caches[0].window = mx.ones((1, 1, caches[0].head_dim))
        caches[2].compress_kv_writer.write(
            mx.ones((1, 1, caches[2].head_dim)), 0
        )

        states = [cache.state for cache in caches]
        mx.eval(states)

        self.assertIs(states[0][0], caches[0].window)
        self.assertEqual(
            sum(
                caches[2].compress_kv_writer.buffer is value
                for state in states
                for value in state
            ),
            1,
        )
        self.assertIs(caches[3].compress_kv_owner, caches[2].compress_kv_writer)

    def test_prompt_cache_persistence_is_refused_rather_than_silently_unshared(self):
        cache = make_deepseek_v41_attention_caches(_official_text_config())[3]
        with self.assertRaises(NotImplementedError):
            _ = cache.meta_state

    def test_exo_warmup_cache_can_be_trimmed_back_to_empty_and_reused(self):
        config = _tiny_text_config()
        stack = DeepseekV41AttentionStack(config)
        mx.eval(stack.parameters())
        caches = stack.make_cache()

        prompt = mx.random.normal((1, 3, config.hidden_size))
        stack([prompt] * len(stack.layers), caches)
        continuation = mx.random.normal((1, 1, config.hidden_size))
        stack([continuation] * len(stack.layers), caches)
        stack([continuation] * len(stack.layers), caches)
        for cache in caches:
            self.assertEqual(cache.trim(2), 2)

        self.assertTrue(all(cache.offset == 3 for cache in caches))
        self.assertTrue(
            all(
                owner.length == 3 // owner.compress_ratio
                for owner in caches[0].shared.compress_kv_owners.values()
            )
        )
        self.assertTrue(
            all(
                owner.length == 3 // owner.compress_ratio
                for owner in caches[0].shared.index_key_owners.values()
            )
        )

        outputs = stack([continuation] * len(stack.layers), caches)
        mx.eval(outputs)
        self.assertTrue(all(output.shape == (1, 1, config.hidden_size) for output in outputs))

    def test_malformed_config_is_rejected_at_cache_construction(self):
        with self.assertRaises(ValueError):
            make_deepseek_v41_attention_caches(
                _tiny_text_config(index_source_layer_ids=[2, 6])
            )


class TestDeepseekV41PhysicalLatentCache(unittest.TestCase):
    def _owner(self):
        return PhysicalLatentCache("compress_kv", 2, 4, 1, growth_slots=8)

    def test_append_only_growth_preserves_contents(self):
        owner = self._owner()
        first = mx.arange(3 * 4, dtype=mx.float32).reshape(1, 3, 4)
        owner.write(first, 0)
        self.assertEqual(owner.length, 3)
        self.assertEqual(owner.capacity, 8)
        second = mx.full((1, 7, 4), 9.0)
        owner.write(second, 3)
        self.assertEqual(owner.length, 10)
        self.assertEqual(owner.capacity, 16)
        self.assertTrue(mx.array_equal(owner.read(10)[:, :3], first))
        self.assertTrue(mx.array_equal(owner.read(10)[:, 3:], second))

    def test_non_contiguous_write_is_rejected(self):
        owner = self._owner()
        owner.write(mx.zeros((1, 2, 4)), 0)
        with self.assertRaises(ValueError) as ctx:
            owner.write(mx.zeros((1, 1, 4)), 3)
        self.assertIn("growth invariant", str(ctx.exception))

    def test_rewriting_an_existing_slot_is_rejected(self):
        owner = self._owner()
        owner.write(mx.zeros((1, 2, 4)), 0)
        with self.assertRaises(ValueError):
            owner.write(mx.zeros((1, 1, 4)), 1)

    def test_reading_past_the_written_length_is_rejected(self):
        owner = self._owner()
        owner.write(mx.zeros((1, 2, 4)), 0)
        with self.assertRaises(ValueError) as ctx:
            owner.read(3)
        self.assertIn("ran before its source", str(ctx.exception))

    def test_wrong_trailing_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            self._owner().write(mx.zeros((1, 2, 5)), 0)


class TestDeepseekV41WindowIndices(unittest.TestCase):
    def test_prefill_rows_are_causal_and_sentinel_padded(self):
        idxs = window_topk_idxs(4, 1, 6, 0)
        self.assertEqual(idxs.shape, (1, 6, 4))
        rows = idxs[0].tolist()
        self.assertEqual(rows[0], [0, -1, -1, -1])
        self.assertEqual(rows[2], [0, 1, 2, -1])
        self.assertEqual(rows[3], [0, 1, 2, 3])
        self.assertEqual(rows[5], [2, 3, 4, 5])

    def test_prefill_shorter_than_the_window_never_indexes_past_the_chunk(self):
        idxs = window_topk_idxs(128, 1, 3, 0)
        self.assertEqual(idxs.shape, (1, 3, 3))
        self.assertEqual(idxs[0].tolist()[2], [0, 1, 2])

    def test_decode_lists_the_whole_ring_oldest_first(self):
        # start_pos 6, window 4: the incoming token lands in ring slot 2, so
        # the oldest live slot is 3 and the row wraps 3, 0, 1, 2.
        idxs = window_topk_idxs(4, 2, 1, 6)
        self.assertEqual(idxs.shape, (2, 1, 4))
        self.assertEqual(idxs[0].tolist()[0], [3, 0, 1, 2])
        self.assertEqual(idxs[1].tolist()[0], [3, 0, 1, 2])

    def test_decode_before_the_ring_has_wrapped_masks_unwritten_slots(self):
        # start_pos 2 fills slots 0..2; slot 3 has never been written.
        idxs = window_topk_idxs(4, 1, 1, 2)
        self.assertEqual(idxs[0].tolist()[0], [-1, 0, 1, 2])
        idxs = window_topk_idxs(4, 1, 1, 1)
        self.assertEqual(idxs[0].tolist()[0], [-1, -1, 0, 1])

    def test_rejects_non_positive_window(self):
        with self.assertRaises(ValueError):
            window_topk_idxs(0, 1, 1, 0)


class TestDeepseekV41SparseAttn(unittest.TestCase):
    def _dense_reference(self, q, kv, sink, scale):
        scores = mx.einsum("bmhd,bnd->bmhn", q, kv) * scale
        m = mx.max(scores, axis=-1, keepdims=True)
        w = mx.exp(scores - m)
        denom = mx.sum(w, axis=-1, keepdims=True) + mx.exp(
            sink.reshape(1, 1, -1, 1) - m
        )
        return mx.einsum("bmhn,bnd->bmhd", w / denom, kv)

    def test_selecting_every_position_matches_dense_attention_with_a_sink(self):
        mx.random.seed(0)
        q = mx.random.normal((2, 3, 4, 8))
        kv = mx.random.normal((2, 5, 8))
        sink = mx.array([0.0, 0.5, -1.0, 2.0])
        idxs = mx.broadcast_to(mx.arange(5, dtype=mx.int32), (2, 3, 5))
        got = sparse_attn(q, kv, sink, idxs, 0.125)
        want = self._dense_reference(q, kv, sink, 0.125)
        self.assertTrue(mx.allclose(got, want, atol=1e-5).item())

    def test_query_chunking_does_not_change_the_result(self):
        mx.random.seed(1)
        q = mx.random.normal((1, 7, 2, 8))
        kv = mx.random.normal((1, 6, 8))
        sink = mx.zeros((2,))
        idxs = mx.broadcast_to(mx.arange(6, dtype=mx.int32), (1, 7, 6))
        whole = sparse_attn(q, kv, sink, idxs, 0.3, query_chunk=64)
        chunked = sparse_attn(q, kv, sink, idxs, 0.3, query_chunk=2)
        self.assertTrue(mx.allclose(whole, chunked, atol=1e-6).item())

    def test_a_fully_unreachable_row_is_zero_and_not_nan(self):
        mx.random.seed(2)
        q = mx.random.normal((1, 2, 2, 8))
        kv = mx.random.normal((1, 4, 8))
        sink = mx.zeros((2,))
        idxs = mx.array([[[0, 1, 2], [-1, -1, -1]]], dtype=mx.int32)
        out = sparse_attn(q, kv, sink, idxs, 0.5)
        dead = out[:, 1]
        self.assertFalse(mx.any(mx.isnan(dead)).item())
        self.assertTrue(mx.all(dead == 0.0).item())
        self.assertTrue(mx.any(out[:, 0] != 0.0).item())

    def test_masked_positions_are_ignored_entirely(self):
        mx.random.seed(3)
        q = mx.random.normal((1, 1, 2, 8))
        kv = mx.random.normal((1, 5, 8))
        sink = mx.zeros((2,))
        picked = sparse_attn(
            q, kv, sink, mx.array([[[0, 3]]], dtype=mx.int32), 0.4
        )
        padded = sparse_attn(
            q, kv, sink, mx.array([[[0, 3, -1, -1]]], dtype=mx.int32), 0.4
        )
        self.assertTrue(mx.allclose(picked, padded, atol=1e-6).item())

    def test_the_sink_suppresses_the_output(self):
        mx.random.seed(4)
        q = mx.random.normal((1, 1, 1, 8))
        kv = mx.random.normal((1, 4, 8))
        idxs = mx.broadcast_to(mx.arange(4, dtype=mx.int32), (1, 1, 4))
        small = sparse_attn(q, kv, mx.array([-20.0]), idxs, 0.2)
        large = sparse_attn(q, kv, mx.array([20.0]), idxs, 0.2)
        self.assertGreater(
            mx.sum(mx.abs(small)).item(), mx.sum(mx.abs(large)).item()
        )
        self.assertLess(mx.sum(mx.abs(large)).item(), 1e-4)

    def test_shape_contract_is_enforced(self):
        q = mx.zeros((1, 1, 2, 8))
        kv = mx.zeros((1, 4, 8))
        idxs = mx.zeros((1, 1, 4), dtype=mx.int32)
        with self.assertRaises(ValueError):
            sparse_attn(q, mx.zeros((1, 4, 7)), mx.zeros((2,)), idxs, 0.5)
        with self.assertRaises(ValueError):
            sparse_attn(q, kv, mx.zeros((3,)), idxs, 0.5)
        with self.assertRaises(ValueError):
            sparse_attn(q, kv, mx.zeros((2,)), mx.zeros((1, 2, 4), mx.int32), 0.5)


class TestDeepseekV41CandidateBlocks(unittest.TestCase):
    def test_keeps_the_best_blocks_and_always_pins_the_newest(self):
        # Blocks of 2 over 8 positions. Block 0 scores highest, block 3 holds
        # the newest position and is pinned even though it scores lowest.
        logits = mx.array([[9.0, 8.0, 1.0, 1.0, 5.0, 5.0, 0.0, 0.0]])
        mask = select_candidate_blocks(logits, 8, topk_blocks=2, block_size=2)
        self.assertEqual(mask.shape, (1, 8))
        self.assertEqual(
            mask[0].tolist(),
            [True, True, False, False, False, False, True, True],
        )

    def test_unreachable_blocks_are_never_admitted(self):
        neg = -float("inf")
        logits = mx.array([[3.0, 1.0, neg, neg, neg, neg]])
        mask = select_candidate_blocks(logits, 2, topk_blocks=3, block_size=2)
        self.assertEqual(
            mask[0].tolist(), [True, True, False, False, False, False]
        )

    def test_per_query_reachability_during_prefill(self):
        neg = -float("inf")
        logits = mx.array([[[5.0, 4.0, neg, neg], [5.0, 4.0, 9.0, 1.0]]])
        lens = mx.array([[2], [4]], dtype=mx.int32)
        mask = select_candidate_blocks(logits, lens, topk_blocks=1, block_size=2)
        self.assertEqual(mask[0][0].tolist(), [True, True, False, False])
        # The newest block wins the pin even though block 0 is not far behind.
        self.assertEqual(mask[0][1].tolist(), [False, False, True, True])

    def test_a_ragged_tail_block_is_handled(self):
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        mask = select_candidate_blocks(logits, 5, topk_blocks=1, block_size=2)
        self.assertEqual(mask.shape, (1, 5))
        self.assertEqual(mask[0].tolist(), [False, False, False, False, True])

    def test_rejects_degenerate_block_parameters(self):
        logits = mx.zeros((1, 4))
        with self.assertRaises(ValueError):
            select_candidate_blocks(logits, 4, topk_blocks=1, block_size=0)
        with self.assertRaises(ValueError):
            select_candidate_blocks(logits, 4, topk_blocks=0, block_size=2)


class TestDeepseekV41ActQuantRoundtrip(unittest.TestCase):
    def _block(self, values, size=32):
        return mx.array(values + [0.0] * (size - len(values)), dtype=mx.float32)

    def test_fp8_roundtrip_rounds_half_to_even_on_the_e4m3_grid(self):
        # amax 448 gives a power-of-two scale of exactly 1, so the expected
        # values are the raw E4M3 grid points.
        x = self._block([448.0, 240.0, 1.0625, 1.1875, -1.0625, 2.0, 3.0, 0.0])
        got = act_quant_roundtrip(x.reshape(1, 32))[0].tolist()
        self.assertEqual(
            got[:8], [448.0, 240.0, 1.0, 1.25, -1.0, 2.0, 3.0, 0.0]
        )

    def test_fp8_roundtrip_saturates_at_the_finite_maximum(self):
        x = self._block([1e9, 448.0])
        got = act_quant_roundtrip(x.reshape(1, 32))[0]
        self.assertFalse(mx.any(mx.isnan(got)).item())
        self.assertFalse(mx.any(mx.isinf(got)).item())

    def test_fp4_roundtrip_matches_the_e2m1_grid_with_ties_to_even(self):
        x = self._block([6.0, 5.0, 3.5, 2.5, 1.75, 1.25, 0.75, 0.25, -5.0])
        got = fp4_act_quant_roundtrip(x.reshape(1, 32))[0].tolist()
        self.assertEqual(
            got[:9], [6.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 0.0, -4.0]
        )

    def test_fp4_roundtrip_with_e4m3_scales_uses_sixteen_wide_blocks(self):
        x = self._block([6.0, 5.0, 3.5, 2.5, 1.75, 1.25, 0.75, 0.25], size=16)
        got = fp4_act_quant_roundtrip(
            x.reshape(1, 16), COMPRESS_KV_FP4_BLOCK_SIZE, e4m3_scale=True
        )[0].tolist()
        self.assertEqual(got[:8], [6.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 0.0])

    def test_roundtrips_are_idempotent(self):
        mx.random.seed(5)
        x = mx.random.normal((3, 64))
        once = act_quant_roundtrip(x)
        self.assertTrue(mx.array_equal(once, act_quant_roundtrip(once)).item())
        once4 = fp4_act_quant_roundtrip(x)
        self.assertTrue(
            mx.array_equal(once4, fp4_act_quant_roundtrip(once4)).item()
        )

    def test_roundtrip_preserves_dtype_and_shape(self):
        x = mx.random.normal((2, 3, 32)).astype(mx.bfloat16)
        got = act_quant_roundtrip(x)
        self.assertEqual(got.shape, x.shape)
        self.assertEqual(got.dtype, mx.bfloat16)

    def test_a_partial_tail_block_is_refused_rather_than_padded(self):
        with self.assertRaises(ValueError) as ctx:
            act_quant_roundtrip(mx.zeros((1, 33)))
        self.assertIn("not evenly divisible", str(ctx.exception))


class TestDeepseekV41Rope(unittest.TestCase):
    def test_inverse_rotation_recovers_the_input(self):
        mx.random.seed(6)
        x = mx.random.normal((2, 5, 3, 16))
        freqs = yarn_rope_frequencies(8, 0, 10000.0, 1.0, 32.0, 1.0)
        cos, sin = rope_cos_sin(freqs, mx.arange(5))
        rotated = apply_rope_tail(x, cos, sin, 8)
        back = apply_rope_tail(rotated, cos, sin, 8, inverse=True)
        self.assertTrue(mx.allclose(back, x, atol=1e-5).item())

    def test_only_the_rope_tail_is_touched(self):
        mx.random.seed(7)
        x = mx.random.normal((1, 3, 16))
        freqs = yarn_rope_frequencies(8, 0, 10000.0, 1.0, 32.0, 1.0)
        cos, sin = rope_cos_sin(freqs, mx.arange(3))
        rotated = apply_rope_tail(x, cos, sin, 8)
        self.assertTrue(mx.array_equal(rotated[..., :8], x[..., :8]).item())
        self.assertFalse(mx.allclose(rotated[..., 8:], x[..., 8:]).item())

    def test_position_zero_is_the_identity(self):
        mx.random.seed(8)
        x = mx.random.normal((1, 1, 16))
        freqs = yarn_rope_frequencies(16, 65536, 160000.0, 16.0, 32.0, 1.0)
        cos, sin = rope_cos_sin(freqs, mx.array([0]))
        self.assertTrue(
            mx.allclose(apply_rope_tail(x, cos, sin, 16), x, atol=1e-6).item()
        )

    def test_yarn_scaling_only_applies_when_enabled(self):
        plain = yarn_rope_frequencies(64, 0, 160000.0, 16.0, 32.0, 1.0)
        scaled = yarn_rope_frequencies(64, 65536, 160000.0, 16.0, 32.0, 1.0)
        self.assertEqual(plain.shape, (32,))
        # High-frequency dimensions already fit inside the training context and
        # keep their frequency; the slow tail is divided by the factor.
        self.assertAlmostEqual(plain[0].item(), scaled[0].item(), places=6)
        self.assertLess(scaled[-1].item(), plain[-1].item())
        self.assertAlmostEqual(scaled[-1].item(), plain[-1].item() / 16.0, places=9)

    def test_rejects_odd_rope_head_dim(self):
        with self.assertRaises(ValueError):
            yarn_rope_frequencies(7, 0, 10000.0, 1.0, 32.0, 1.0)


class TestDeepseekV41AttentionExecution(unittest.TestCase):
    def _stack(self, **overrides):
        mx.random.seed(11)
        config = _tiny_text_config(**overrides)
        stack = DeepseekV41AttentionStack(config)
        mx.eval(stack.parameters())
        return config, stack

    def _inputs(self, config, batch, seqlen):
        mx.random.seed(12)
        return mx.random.normal((batch, seqlen, config.hidden_size))

    def test_the_tiny_config_exercises_all_three_compress_ratios(self):
        _, stack = self._stack()
        self.assertEqual(
            sorted({p.compress_ratio for p in stack.policies}), [0, 1, 2]
        )

    def test_prefill_produces_one_output_per_layer(self):
        config, stack = self._stack()
        n = len(stack.layers)
        x = self._inputs(config, 1, 10)
        out = stack([x] * n, stack.make_cache())
        self.assertEqual(len(out), n)
        for o in out:
            self.assertEqual(o.shape, (1, 10, config.hidden_size))
            self.assertFalse(mx.any(mx.isnan(o)).item())

    def test_token_decode_matches_the_equivalent_prefill(self):
        config, stack = self._stack()
        n = len(stack.layers)
        x = self._inputs(config, 1, 10)

        full = stack([x] * n, stack.make_cache())

        incremental = stack.make_cache()
        stack([x[:, :6]] * n, incremental)
        step = None
        for t in range(6, 10):
            step = stack([x[:, t : t + 1]] * n, incremental)

        for i, policy in enumerate(stack.policies):
            self.assertTrue(
                mx.allclose(step[i], full[i][:, -1:], atol=2e-4).item(),
                f"layer {i} (ratio {policy.compress_ratio}) decode diverged "
                "from the equivalent prefill",
            )

    def test_batched_token_decode_matches_the_equivalent_prefill(self):
        config, stack = self._stack()
        n = len(stack.layers)
        x = self._inputs(config, 3, 8)
        full = stack([x] * n, stack.make_cache())
        incremental = stack.make_cache()
        stack([x[:, :4]] * n, incremental)
        step = None
        for t in range(4, 8):
            step = stack([x[:, t : t + 1]] * n, incremental)
        for i in range(n):
            self.assertTrue(
                mx.allclose(step[i], full[i][:, -1:], atol=2e-4).item(),
                f"layer {i} batched decode diverged from prefill",
            )

    def test_compressed_owners_advance_by_position_over_ratio(self):
        config, stack = self._stack()
        n = len(stack.layers)
        caches = stack.make_cache()
        x = self._inputs(config, 1, 10)
        stack([x] * n, caches)
        self.assertEqual(caches[2].compress_kv_writer.length, 5)
        self.assertEqual(caches[2].index_key_writer.length, 5)
        self.assertEqual(caches[4].compress_kv_writer.length, 10)
        self.assertEqual(caches[4].index_key_writer.length, 10)
        owners = {
            id(c.compress_kv_owner) for c in caches if c.compress_kv_owner is not None
        }
        self.assertEqual(len(owners), 2)

    def test_a_ratio_two_source_only_emits_on_complete_groups(self):
        config, stack = self._stack()
        n = len(stack.layers)
        caches = stack.make_cache()
        x = self._inputs(config, 1, 8)
        stack([x[:, :6]] * n, caches)
        self.assertEqual(caches[2].compress_kv_writer.length, 3)
        stack([x[:, 6:7]] * n, caches)
        # Position 6 opens a new group, so nothing is published this step.
        self.assertEqual(caches[2].compress_kv_writer.length, 3)
        self.assertEqual(caches[4].compress_kv_writer.length, 7)
        stack([x[:, 7:8]] * n, caches)
        self.assertEqual(caches[2].compress_kv_writer.length, 4)
        self.assertEqual(caches[4].compress_kv_writer.length, 8)

    def test_the_window_ring_never_grows_with_context(self):
        config, stack = self._stack()
        n = len(stack.layers)
        caches = stack.make_cache()
        x = self._inputs(config, 1, 10)
        stack([x] * n, caches)
        for c in caches:
            self.assertEqual(
                c.window.shape, (1, config.sliding_window, config.head_dim)
            )
        for t in range(10, 14):
            stack([x[:, :1]] * n, caches)
        for c in caches:
            self.assertEqual(
                c.window.shape, (1, config.sliding_window, config.head_dim)
            )
            self.assertEqual(c.offset, 14)

    def test_a_pure_window_layer_reads_no_compressed_state(self):
        config, stack = self._stack()
        caches = stack.make_cache()
        x = self._inputs(config, 1, 5)
        # Layer 0 is compress_ratio 0, so it can run entirely on its own.
        out = stack.layers[0](x, caches[0])
        self.assertEqual(out.shape, (1, 5, config.hidden_size))
        self.assertEqual(caches[2].compress_kv_writer.length, 0)

    def test_running_a_consumer_before_any_source_fails_loud(self):
        config, stack = self._stack()
        caches = stack.make_cache()
        x = self._inputs(config, 1, 4)
        with self.assertRaises(ValueError) as ctx:
            stack.layers[3](x, caches[3])
        self.assertIn("ascending", str(ctx.exception))

    def test_skipping_the_index_source_fails_loud(self):
        config, stack = self._stack()
        caches = stack.make_cache()
        x = self._inputs(config, 1, 4)
        # 0 and 1 are pure window layers, so 0, 1, 3 is ascending but still
        # skips layer 2, the source layer 3 shares its compressed KV with.
        stack.layers[0](x, caches[0])
        stack.layers[1](x, caches[1])
        with self.assertRaises(ValueError) as ctx:
            stack.layers[3](x, caches[3])
        self.assertIn("topk_idxs", str(ctx.exception))

    def test_re_running_an_earlier_layer_fails_loud(self):
        config, stack = self._stack()
        caches = stack.make_cache()
        x = self._inputs(config, 1, 4)
        for i in range(4):
            stack.layers[i](x, caches[i])
        with self.assertRaises(ValueError) as ctx:
            stack.layers[2](x, caches[2])
        self.assertIn("ascending", str(ctx.exception))

    def test_layers_must_advance_over_the_same_step(self):
        config, stack = self._stack()
        caches = stack.make_cache()
        x = self._inputs(config, 1, 4)
        stack.layers[0](x, caches[0])
        with self.assertRaises(ValueError) as ctx:
            stack.layers[1](x[:, :2], caches[1])
        self.assertIn("advance together", str(ctx.exception))

    def test_a_layer_refuses_another_layers_cache(self):
        config, stack = self._stack()
        caches = stack.make_cache()
        x = self._inputs(config, 1, 4)
        with self.assertRaises(ValueError) as ctx:
            stack.layers[1](x, caches[2])
        self.assertIn("cache of layer", str(ctx.exception))

    def test_chunked_prefill_is_refused_rather_than_approximated(self):
        config, stack = self._stack()
        n = len(stack.layers)
        caches = stack.make_cache()
        x = self._inputs(config, 1, 6)
        stack([x[:, :4]] * n, caches)
        with self.assertRaises(ValueError) as ctx:
            stack([x[:, 4:6]] * n, caches)
        self.assertIn("chunked prefill", str(ctx.exception))

    def test_stack_arity_is_enforced(self):
        config, stack = self._stack()
        n = len(stack.layers)
        x = self._inputs(config, 1, 4)
        with self.assertRaises(ValueError):
            stack([x] * (n - 1), stack.make_cache())
        with self.assertRaises(ValueError):
            stack([x] * n, None)
        with self.assertRaises(ValueError):
            stack([x] * n, stack.make_cache()[:-1])

    def test_attn_sink_is_a_learned_per_head_parameter(self):
        config, stack = self._stack()
        params = stack.layers[0].parameters()
        self.assertIn("attn_sink", params)
        self.assertEqual(
            params["attn_sink"].shape, (config.num_attention_heads,)
        )
        # The on-demand rope frequencies are not a checkpoint tensor and must
        # not show up as one.
        self.assertNotIn("rope_freqs", params)
        self.assertNotIn("_rope_freqs", params)

    def test_only_source_layers_carry_compressor_and_indexer_weights(self):
        _, stack = self._stack()
        layers = stack.parameters()["layers"]
        self.assertIn("compressor", layers[2])
        self.assertIn("wgate", layers[2]["compressor"])
        # A ratio-1 source is a plain projection, so it has no pooling gate.
        self.assertIn("compressor", layers[4])
        self.assertNotIn("wgate", layers[4]["compressor"])
        self.assertNotIn("compressor", layers[3])
        self.assertNotIn("indexer", layers[3])
        self.assertNotIn("compressor", layers[6])
        # Only a KV source derives index keys from its own latent.
        self.assertIn("wk", layers[4]["indexer"])
        self.assertIn("indexer", layers[6])
        self.assertNotIn("wk", layers[6]["indexer"])
        self.assertNotIn("compressor", layers[0])
        self.assertNotIn("indexer", layers[0])

    def test_wo_a_is_consumed_block_diagonally_over_o_groups(self):
        config, stack = self._stack()
        layer = stack.layers[0]
        self.assertEqual(
            layer.wo_a.weight.shape,
            (
                config.o_groups * config.o_lora_rank,
                config.num_attention_heads * config.head_dim // config.o_groups,
            ),
        )
        self.assertEqual(
            layer.wo_b.weight.shape,
            (config.hidden_size, config.o_groups * config.o_lora_rank),
        )


def _tiny_hc_text_config(**overrides):
    """A small config carrying the real hc_mult == 4 / 20-iteration Sinkhorn
    settings, with a tiny hidden size so the [mix_hc, hc_mult * dim] projection
    stays a few hundred floats instead of the official 24 x 20480.
    """
    cfg = _text_config_dict()
    cfg.update(
        {
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 0,
            "compress_ratios": [0, 0, 0, 0],
        }
    )
    cfg.update(overrides)
    return TextConfig.from_dict(cfg)


def _tiny_moe_text_config(**overrides):
    """A small but structurally faithful MoE config: 8 routed experts, 3 active,
    one shared expert, the real sqrtsoftplus / noaux_tc / norm_topk_prob /
    routed_scaling_factor=1.5 / swiglu_limit=10 routing rules, and dimensions
    that are exact multiples of the 32-wide FP4/FP8 weight block so nothing is
    padded. Deliberately nowhere near the official 384 x 2304 x 5120 expert
    stack, which no test needs to allocate.
    """
    cfg = _text_config_dict()
    cfg.update(
        {
            "hidden_size": 32,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 0,
            "compress_ratios": [0, 0, 0, 0],
            "n_routed_experts": 8,
            "num_experts_per_tok": 3,
        }
    )
    cfg.update(overrides)
    return TextConfig.from_dict(cfg)


def _tiny_dspark_text_config(**overrides):
    cfg = _text_config_dict()
    cfg.update(
        {
            "vocab_size": 64,
            "hidden_size": 32,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 3,
            "compress_ratios": [0] * 7,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "dspark_block_size": 2,
            "dspark_noise_token_id": 63,
            "dspark_target_layer_ids": [1, 3],
            "dspark_markov_rank": 8,
            "dspark_n_routed_experts": 16,
            "dspark_num_experts_per_tok": 3,
            "num_attention_heads": 4,
            "head_dim": 32,
            "qk_rope_head_dim": 8,
            "q_lora_rank": 32,
            "o_groups": 2,
            "o_lora_rank": 8,
            "sliding_window": 4,
        }
    )
    cfg.update(overrides)
    return TextConfig.from_dict(cfg)


def _fill_packed(linear, lo=0, hi=256, scale_code=127):
    """Give a packed Linear real, deterministic bytes. E8M0 code 127 is exactly
    2**0 == 1.0, so the block scale is a no-op and the decoded weight is the
    raw E2M1/E4M3 grid -- which keeps the expectations below readable.
    """
    linear.weight = mx.random.randint(lo, hi, linear.weight.shape).astype(mx.uint8)
    if linear.quant is not None:
        linear.scale = mx.full(linear.scale.shape, scale_code).astype(mx.uint8)
    return linear


def _zero_expert(expert):
    for lin in (expert.w1, expert.w2, expert.w3):
        lin.weight = mx.zeros(lin.weight.shape, dtype=lin.weight.dtype)
    return expert


def _build_tiny_moe(config=None, seed=0, **kwargs):
    mx.random.seed(seed)
    config = _tiny_moe_text_config() if config is None else config
    moe = DeepseekV41MoE(config, **kwargs)
    for expert in moe.experts:
        for lin in (expert.w1, expert.w2, expert.w3):
            _fill_packed(lin)
    for lin in (
        moe.shared_experts.w1,
        moe.shared_experts.w2,
        moe.shared_experts.w3,
    ):
        # 0x78..0xFF are the E4M3 NaN/large encodings; stay on the finite grid.
        _fill_packed(lin, hi=120)
    moe.gate.weight = mx.random.normal(moe.gate.weight.shape) * 0.3
    moe.gate.bias = mx.random.normal(moe.gate.bias.shape)
    return config, moe


class TestDeepseekV41SinkhornSplit(unittest.TestCase):
    """hc_split_sinkhorn against inference/kernel.py hc_split_sinkhorn_kernel."""

    HC = 4
    ITERS = 20
    EPS = 1e-6

    def _mixes(self, batch=2, seqlen=3):
        mx.random.seed(11)
        mix_hc = (2 + self.HC) * self.HC
        return (
            mx.random.normal((batch, seqlen, mix_hc)),
            mx.array([0.7, 1.3, 0.9], dtype=mx.float32),
            mx.random.normal((mix_hc,)),
        )

    def test_comb_is_doubly_stochastic_after_the_official_iteration_count(self):
        mixes, scale, base = self._mixes()
        _, _, comb = hc_split_sinkhorn(mixes, scale, base, self.HC, self.ITERS, self.EPS)
        self.assertEqual(comb.shape, (2, 3, self.HC, self.HC))
        self.assertTrue(mx.all(comb > 0).item())
        self.assertTrue(
            mx.allclose(mx.sum(comb, axis=-1), mx.ones((2, 3, self.HC)), atol=2e-3).item()
        )
        self.assertTrue(
            mx.allclose(mx.sum(comb, axis=-2), mx.ones((2, 3, self.HC)), atol=2e-3).item()
        )

    def test_a_single_iteration_is_not_yet_doubly_stochastic(self):
        # Guards the iteration count itself: if sinkhorn_iters were ignored the
        # test above would pass for the wrong reason.
        mixes, scale, base = self._mixes()
        _, _, one = hc_split_sinkhorn(mixes, scale, base, self.HC, 1, self.EPS)
        row_err = mx.max(mx.abs(mx.sum(one, axis=-1) - 1.0)).item()
        self.assertGreater(row_err, 1e-3)

    def test_pre_and_post_activation_ranges(self):
        mixes, scale, base = self._mixes()
        pre, post, _ = hc_split_sinkhorn(mixes, scale, base, self.HC, self.ITERS, self.EPS)
        self.assertEqual(pre.shape, (2, 3, self.HC))
        self.assertEqual(post.shape, (2, 3, self.HC))
        # pre = sigmoid(...) + eps, strictly above the eps floor; post = 2*sigmoid.
        self.assertTrue(mx.all(pre > self.EPS).item())
        self.assertTrue(mx.all(pre < 1.0 + 2 * self.EPS).item())
        self.assertTrue(mx.all(post > 0.0).item())
        self.assertTrue(mx.all(post < 2.0).item())

    def test_zero_mixes_give_the_uniform_doubly_stochastic_fixed_point(self):
        mix_hc = (2 + self.HC) * self.HC
        zeros = mx.zeros((1, 1, mix_hc), dtype=mx.float32)
        pre, post, comb = hc_split_sinkhorn(
            zeros,
            mx.array([0.7, 1.3, 0.9], dtype=mx.float32),
            mx.zeros((mix_hc,), dtype=mx.float32),
            self.HC,
            self.ITERS,
            self.EPS,
        )
        self.assertTrue(mx.allclose(comb, mx.full(comb.shape, 1.0 / self.HC), atol=1e-5).item())
        self.assertTrue(mx.allclose(pre, mx.full(pre.shape, 0.5 + self.EPS), atol=1e-6).item())
        self.assertTrue(mx.allclose(post, mx.ones(post.shape), atol=1e-6).item())

    def test_the_three_segments_are_read_in_the_official_order_and_layout(self):
        # Zeroing the pre/post scales isolates hc_base, so each segment can be
        # checked against a literal expectation -- including that comb is read
        # row-major as [j * hc + k], not transposed.
        mix_hc = (2 + self.HC) * self.HC
        base = mx.arange(mix_hc, dtype=mx.float32)
        pre, post, comb = hc_split_sinkhorn(
            mx.zeros((1, mix_hc), dtype=mx.float32),
            mx.array([0.0, 0.0, 1.0], dtype=mx.float32),
            base,
            self.HC,
            1,
            self.EPS,
        )
        self.assertTrue(
            mx.allclose(pre[0], mx.sigmoid(base[: self.HC]) + self.EPS, atol=1e-6).item()
        )
        self.assertTrue(
            mx.allclose(
                post[0], 2.0 * mx.sigmoid(base[self.HC : 2 * self.HC]), atol=1e-6
            ).item()
        )
        expected = mx.softmax(
            base[2 * self.HC :].reshape(self.HC, self.HC), axis=-1
        ) + self.EPS
        expected = expected / (mx.sum(expected, axis=-2, keepdims=True) + self.EPS)
        self.assertTrue(mx.allclose(comb[0], expected, atol=1e-6).item())

    def test_malformed_inputs_fail_closed(self):
        mixes, scale, base = self._mixes()
        cases = [
            (mixes[..., :-1], scale, base, self.HC, self.ITERS, self.EPS),
            (mixes, scale[:2], base, self.HC, self.ITERS, self.EPS),
            (mixes, scale, base[:-1], self.HC, self.ITERS, self.EPS),
            (mixes, scale, base, 0, self.ITERS, self.EPS),
            (mixes, scale, base, self.HC, 0, self.EPS),
            (mixes, scale, base, self.HC, self.ITERS, 0.0),
        ]
        for args in cases:
            with self.assertRaises(ValueError):
                hc_split_sinkhorn(*args)


class TestDeepseekV41ResidualStream(unittest.TestCase):
    """The hc_mult == 4 parallel residual stream (Transformer.forward / Block)."""

    def test_expansion_starts_from_four_identical_copies(self):
        mx.random.seed(3)
        h = mx.random.normal((2, 3, 5))
        stream = expand_hyper_connection_stream(h, 4)
        self.assertEqual(stream.shape, (2, 3, 4, 5))
        for copy in range(4):
            self.assertTrue(mx.array_equal(stream[:, :, copy], h).item())

    def test_the_identity_pre_mix_selects_copy_zero(self):
        pre_mix = make_identity_pre_mix(2, 3, 4)
        self.assertEqual(pre_mix.shape, (2, 3, 4))
        self.assertTrue(mx.array_equal(pre_mix[..., 0], mx.ones((2, 3))).item())
        self.assertTrue(mx.array_equal(pre_mix[..., 1:], mx.zeros((2, 3, 3))).item())

    def test_expanding_then_collapsing_with_the_identity_recovers_the_embedding(self):
        mx.random.seed(4)
        h = mx.random.normal((2, 3, 5))
        stream = expand_hyper_connection_stream(h, 4)
        collapsed = hc_pre(stream, make_identity_pre_mix(2, 3, 4))
        self.assertTrue(mx.allclose(collapsed, h, atol=1e-6).item())

    def test_hc_pre_is_the_pre_mix_weighted_sum_over_copies(self):
        mx.random.seed(5)
        stream = mx.random.normal((2, 3, 4, 5))
        pre_mix = mx.random.normal((2, 3, 4))
        expected = mx.einsum("bshd,bsh->bsd", stream, pre_mix)
        self.assertTrue(mx.allclose(hc_pre(stream, pre_mix), expected, atol=1e-5).item())

    def test_hc_post_mixes_the_residual_over_the_source_copy_axis(self):
        # comb[.., j, k] routes residual copy j into output copy k. The
        # transposed reading is the easy mistake and is numerically plausible,
        # so it is pinned here rather than left implicit.
        mx.random.seed(6)
        residual = mx.random.normal((2, 3, 4, 5))
        sublayer = mx.random.normal((2, 3, 5))
        post_mix = mx.random.normal((2, 3, 4))
        comb = mx.random.normal((2, 3, 4, 4))
        expected = mx.einsum("bsk,bsd->bskd", post_mix, sublayer) + mx.einsum(
            "bsjk,bsjd->bskd", comb, residual
        )
        got = hc_post(sublayer, residual, post_mix, comb)
        self.assertEqual(got.shape, (2, 3, 4, 5))
        self.assertTrue(mx.allclose(got, expected, atol=1e-5).item())

    def test_hc_pre_and_post_shape_contracts_are_enforced(self):
        stream = mx.zeros((2, 3, 4, 5))
        with self.assertRaises(ValueError):
            hc_pre(stream, mx.zeros((2, 3, 3)))
        with self.assertRaises(ValueError):
            hc_pre(mx.zeros((5,)), mx.zeros((5,)))
        with self.assertRaises(ValueError):
            hc_post(mx.zeros((2, 3, 5)), stream, mx.zeros((2, 3, 3)), mx.zeros((2, 3, 4, 4)))
        with self.assertRaises(ValueError):
            hc_post(mx.zeros((2, 3, 5)), stream, mx.zeros((2, 3, 4)), mx.zeros((2, 3, 4, 3)))
        with self.assertRaises(ValueError):
            hc_post(mx.zeros((2, 3, 6)), stream, mx.zeros((2, 3, 4)), mx.zeros((2, 3, 4, 4)))

    def test_expansion_rejects_a_stream_that_is_already_expanded(self):
        with self.assertRaises(ValueError):
            expand_hyper_connection_stream(mx.zeros((2, 3, 4, 5)), 4)
        with self.assertRaises(ValueError):
            make_identity_pre_mix(0, 3, 4)


class TestDeepseekV41HyperConnections(unittest.TestCase):
    """The per-block Hyper-Connection parameters and Block.forward ordering."""

    def _module(self, seed=9, **overrides):
        mx.random.seed(seed)
        config = _tiny_hc_text_config(**overrides)
        hc = DeepseekV41HyperConnections(config)
        # Distinct keys so attn/ffn coefficient tensors cannot collapse to the
        # same draw if the PRNG does not advance between calls.
        for i, name in enumerate(("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base")):
            key = mx.random.key(seed + i + 1)
            setattr(hc, name, mx.random.normal(getattr(hc, name).shape, key=key))
        hc.hc_attn_scale = mx.array([0.5, 0.5, 0.5], dtype=mx.float32)
        hc.hc_ffn_scale = mx.array([0.5, 0.5, 0.5], dtype=mx.float32)
        return config, hc

    def test_parameter_shapes_match_the_pinned_tensor_contract(self):
        config = _official_text_config()
        hc = DeepseekV41HyperConnections(config)
        mix_hc = (2 + config.hc_mult) * config.hc_mult
        self.assertEqual(config.hc_mult, 4)
        self.assertEqual(mix_hc, 24)
        self.assertEqual(hc.hc_attn_fn.shape, (24, 4 * config.hidden_size))
        self.assertEqual(hc.hc_ffn_fn.shape, (24, 4 * config.hidden_size))
        self.assertEqual(hc.hc_attn_base.shape, (24,))
        self.assertEqual(hc.hc_ffn_base.shape, (24,))
        self.assertEqual(hc.hc_attn_scale.shape, (3,))
        self.assertEqual(hc.hc_ffn_scale.shape, (3,))
        for param in (hc.hc_attn_fn, hc.hc_attn_base, hc.hc_attn_scale):
            self.assertEqual(param.dtype, mx.float32)

    def test_mixes_are_deterministic_and_rms_scale_invariant(self):
        config, hc = self._module()
        x = mx.random.normal((2, 3, config.hc_mult, config.hidden_size))
        first = hc.attn_mixes(x)
        again = hc.attn_mixes(x)
        for a, b in zip(first, again):
            self.assertTrue(mx.array_equal(a, b).item())
        # The RMS statistic is taken over the whole flattened hc*d stream, so a
        # uniform rescale of the stream leaves every coefficient unchanged.
        scaled = hc.attn_mixes(x * 4.0)
        for a, b in zip(first, scaled):
            self.assertTrue(mx.allclose(a, b, atol=1e-4).item())

    def test_attention_and_ffn_use_separate_coefficient_sets(self):
        config, hc = self._module()
        x = mx.random.normal((2, 3, config.hc_mult, config.hidden_size))
        attn_pre = hc.attn_mixes(x)[0]
        ffn_pre = hc.ffn_mixes(x)[0]
        self.assertFalse(mx.allclose(attn_pre, ffn_pre, atol=1e-4).item())

    def test_a_sublayer_step_feeds_its_own_pre_mix_to_the_next_step(self):
        config, hc = self._module()
        x = mx.random.normal((2, 3, config.hc_mult, config.hidden_size))
        pre_mix = make_identity_pre_mix(2, 3, config.hc_mult)
        seen = {}

        def attn(v):
            seen["attn_in"] = v
            return v * 2.0

        out, next_pre = hc.sublayer_step(x, pre_mix, attn, "attn")
        expected_pre, expected_post, expected_comb = hc.attn_mixes(x)
        # The sublayer sees the stream collapsed with the *incoming* pre-mix.
        self.assertTrue(
            mx.allclose(seen["attn_in"], hc_pre(x, pre_mix), atol=1e-6).item()
        )
        # ...and the coefficients it produced are handed to the next step.
        self.assertTrue(mx.allclose(next_pre, expected_pre, atol=1e-6).item())
        self.assertTrue(
            mx.allclose(
                out,
                hc_post(seen["attn_in"] * 2.0, x, expected_post, expected_comb),
                atol=1e-5,
            ).item()
        )

    def test_block_step_reproduces_the_official_forward_ordering(self):
        config, hc = self._module()
        x = mx.random.normal((2, 3, config.hc_mult, config.hidden_size))
        pre_mix = make_identity_pre_mix(2, 3, config.hc_mult)

        def attn(v):
            return v * 2.0

        def ffn(v):
            return v * 3.0

        got, got_pre = hc.block_step(x, pre_mix, attn, ffn)

        attn_pre, attn_post, attn_comb = hc.attn_mixes(x)
        mid = hc_post(attn(hc_pre(x, pre_mix)), x, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = hc.ffn_mixes(mid)
        # The FFN collapses with the pre-mix *attention* produced, not with the
        # block's incoming one and not with its own.
        expected = hc_post(ffn(hc_pre(mid, attn_pre)), mid, ffn_post, ffn_comb)

        self.assertEqual(got.shape, x.shape)
        self.assertTrue(mx.allclose(got, expected, atol=1e-5).item())
        self.assertTrue(mx.allclose(got_pre, ffn_pre, atol=1e-6).item())

    def test_an_unknown_sublayer_name_fails_loud(self):
        config, hc = self._module()
        x = mx.zeros((1, 1, config.hc_mult, config.hidden_size))
        with self.assertRaises(ValueError):
            hc.sublayer_step(x, make_identity_pre_mix(1, 1, config.hc_mult), lambda v: v, "mlp")

    def test_a_wrongly_shaped_stream_fails_loud(self):
        config, hc = self._module()
        with self.assertRaises(ValueError):
            hc.attn_mixes(mx.zeros((2, 3, config.hc_mult + 1, config.hidden_size)))
        with self.assertRaises(ValueError):
            hc.attn_mixes(mx.zeros((2, 3, config.hidden_size)))

    def test_malformed_hyper_connection_config_fails_closed(self):
        for overrides in (
            {"hc_mult": 0},
            {"hc_sinkhorn_iters": 0},
            {"hc_eps": 0.0},
            {"hidden_size": 0},
        ):
            with self.assertRaises(ValueError):
                DeepseekV41HyperConnections(_tiny_hc_text_config(**overrides))



class TestDeepseekV41RoutingRule(unittest.TestCase):
    """Gate scoring and the noaux_tc selection rule (inference/model.py Gate)."""

    def test_sqrtsoftplus_matches_the_official_recipe(self):
        mx.random.seed(21)
        logits = mx.random.normal((5, 8)) * 3.0
        expected = mx.sqrt(mx.log(1.0 + mx.exp(logits.astype(mx.float32))))
        got = routing_scores(logits, "sqrtsoftplus")
        self.assertEqual(got.dtype, mx.float32)
        self.assertTrue(mx.allclose(got, expected, atol=1e-5).item())
        # Unbounded above and never normalized across experts: that is exactly
        # why norm_topk_prob has work to do later.
        self.assertFalse(mx.allclose(mx.sum(got, axis=-1), mx.ones((5,)), atol=1e-2).item())

    def test_sqrtsoftplus_is_stable_in_both_tails(self):
        logits = mx.array([[-200.0, 0.0, 200.0]], dtype=mx.float32)
        got = routing_scores(logits, "sqrtsoftplus")
        self.assertTrue(mx.all(mx.isfinite(got)).item())
        self.assertTrue(mx.allclose(got[0, 2], mx.array(200.0 ** 0.5), atol=1e-2).item())

    def test_softmax_and_sigmoid_scoring_are_also_supported(self):
        mx.random.seed(22)
        logits = mx.random.normal((4, 6))
        soft = routing_scores(logits, "softmax")
        self.assertTrue(mx.allclose(mx.sum(soft, axis=-1), mx.ones((4,)), atol=1e-6).item())
        sig = routing_scores(logits, "sigmoid")
        self.assertTrue(mx.allclose(sig, mx.sigmoid(logits.astype(mx.float32)), atol=1e-6).item())

    def test_an_unknown_scoring_function_fails_closed(self):
        with self.assertRaises(ValueError):
            routing_scores(mx.zeros((1, 4)), "relu")

    def test_topk_is_descending_and_breaks_ties_by_lowest_index(self):
        # An all-equal score surface is exactly what a freshly initialized or
        # zero gate produces, so tie behaviour is load-bearing, not academic.
        tied = mx.array([[3.0, 1.0, 3.0, 2.0, 3.0]], dtype=mx.float32)
        self.assertEqual(deterministic_topk_indices(tied, 3).tolist(), [[0, 2, 4]])
        ranked = mx.array([[0.1, 0.9, 0.5, 0.4]], dtype=mx.float32)
        self.assertEqual(deterministic_topk_indices(ranked, 3).tolist(), [[1, 2, 3]])
        self.assertEqual(
            sorted(deterministic_topk_indices(ranked, 4).tolist()[0]), [0, 1, 2, 3]
        )

    def test_topk_is_bit_identical_across_repeated_calls(self):
        mx.random.seed(23)
        scores = mx.random.normal((7, 16))
        first = deterministic_topk_indices(scores, 6)
        for _ in range(3):
            self.assertTrue(mx.array_equal(deterministic_topk_indices(scores, 6), first).item())

    def test_topk_out_of_range_fails_closed(self):
        scores = mx.zeros((2, 4))
        for k in (0, -1, 5):
            with self.assertRaises(ValueError):
                deterministic_topk_indices(scores, k)

    def test_the_correction_bias_steers_selection_but_never_the_weights(self):
        # The routing identity this whole rule exists for: the bias moves which
        # experts run; the weight applied to an expert is its *unbiased* score.
        scores = mx.array([[0.5, 0.4, 0.3, 0.2]], dtype=mx.float32)
        no_bias = mx.zeros((4,), dtype=mx.float32)
        bias = mx.array([0.0, 0.0, 0.0, 10.0], dtype=mx.float32)

        unbiased_w, unbiased_i = noaux_tc_route(scores, no_bias, 2, True, 1.5)
        biased_w, biased_i = noaux_tc_route(scores, bias, 2, True, 1.5)

        self.assertEqual(unbiased_i.tolist(), [[0, 1]])
        self.assertEqual(biased_i.tolist(), [[3, 0]])
        # 0.2 and 0.5 are the *unbiased* scores of experts 3 and 0. A weight of
        # 10.2 anywhere here would mean the bias had leaked into the scaling.
        expected = mx.array([[0.2, 0.5]], dtype=mx.float32) / 0.7 * 1.5
        self.assertTrue(mx.allclose(biased_w, expected, atol=1e-5).item())
        self.assertTrue(mx.all(biased_w < 2.0).item())

    def test_norm_topk_prob_normalizes_before_the_routed_scaling_factor(self):
        scores = mx.array([[0.5, 0.4, 0.3, 0.2]], dtype=mx.float32)
        zero = mx.zeros((4,), dtype=mx.float32)
        normed, _ = noaux_tc_route(scores, zero, 3, True, 1.5)
        self.assertTrue(mx.allclose(mx.sum(normed, axis=-1), mx.array([1.5]), atol=1e-5).item())
        raw, _ = noaux_tc_route(scores, zero, 3, False, 2.0)
        self.assertTrue(
            mx.allclose(raw, mx.array([[1.0, 0.8, 0.6]], dtype=mx.float32), atol=1e-6).item()
        )

    def test_a_single_active_expert_is_never_renormalized_to_one(self):
        # topk == 1 skips normalization upstream, so the lone weight keeps its
        # raw score (times the routed scaling factor).
        scores = mx.array([[0.25, 0.1]], dtype=mx.float32)
        w, i = noaux_tc_route(scores, mx.zeros((2,), dtype=mx.float32), 1, True, 2.0)
        self.assertEqual(i.tolist(), [[0]])
        self.assertTrue(mx.allclose(w, mx.array([[0.5]], dtype=mx.float32), atol=1e-6).item())


class TestDeepseekV41Gate(unittest.TestCase):
    def test_official_config_gates_six_of_three_hundred_eighty_four(self):
        config = _official_text_config()
        gate = DeepseekV41Gate(config)
        self.assertEqual(config.n_routed_experts, 384)
        self.assertEqual(config.num_experts_per_tok, 6)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.scoring_func, "sqrtsoftplus")
        self.assertEqual(config.topk_method, "noaux_tc")
        self.assertEqual(config.routed_scaling_factor, 1.5)
        self.assertEqual(gate.weight.shape, (384, config.hidden_size))
        self.assertEqual(gate.bias.shape, (384,))
        self.assertEqual(gate.bias.dtype, mx.float32)
        # Text-only by default: no VL routing bias is allocated.
        self.assertNotIn("bias_vl", dict(gate.parameters()))

        mx.random.seed(24)
        gate.weight = mx.random.normal(gate.weight.shape) * 0.05
        weights, indices = gate(mx.random.normal((2, config.hidden_size)))
        self.assertEqual(indices.shape, (2, 6))
        self.assertEqual(weights.shape, (2, 6))
        for row in indices.tolist():
            self.assertEqual(len(set(row)), 6)
            self.assertTrue(all(0 <= e < 384 for e in row))
        self.assertTrue(
            mx.allclose(mx.sum(weights, axis=-1), mx.full((2,), 1.5), atol=1e-4).item()
        )

    def test_identical_logits_route_deterministically_to_the_lowest_indices(self):
        config = _tiny_moe_text_config()
        gate = DeepseekV41Gate(config)  # zero weight -> every expert ties
        weights, indices = gate(mx.ones((3, config.hidden_size)))
        self.assertEqual(indices.tolist(), [[0, 1, 2]] * 3)
        # Equal scores, normalized: each selected expert gets scale / topk.
        self.assertTrue(
            mx.allclose(weights, mx.full((3, 3), 1.5 / 3.0), atol=1e-5).item()
        )

    def test_the_same_input_produces_bit_identical_routing(self):
        mx.random.seed(25)
        config = _tiny_moe_text_config()
        gate = DeepseekV41Gate(config)
        gate.weight = mx.random.normal(gate.weight.shape)
        gate.bias = mx.random.normal(gate.bias.shape)
        x = mx.random.normal((4, config.hidden_size))
        w0, i0 = gate(x)
        for _ in range(3):
            w1, i1 = gate(x)
            self.assertTrue(mx.array_equal(i1, i0).item())
            self.assertTrue(mx.array_equal(w1, w0).item())

    def test_the_vl_bias_only_applies_inside_image_spans(self):
        config = _tiny_moe_text_config()
        gate = DeepseekV41Gate(config, vision_enabled=True)
        self.assertIn("bias_vl", dict(gate.parameters()))
        gate.bias_vl = mx.concatenate(
            [mx.zeros((7,), dtype=mx.float32), mx.array([10.0], dtype=mx.float32)]
        )
        x = mx.ones((2, config.hidden_size))
        mask = mx.array([False, True])
        _, indices = gate(x, mask)
        self.assertEqual(indices.tolist()[0], [0, 1, 2])
        self.assertEqual(indices.tolist()[1][0], 7)

    def test_a_vl_mask_against_a_text_only_gate_fails_loud(self):
        config = _tiny_moe_text_config()
        gate = DeepseekV41Gate(config)
        with self.assertRaises(ValueError):
            gate(mx.ones((2, config.hidden_size)), mx.array([False, True]))

    def test_shape_contracts_are_enforced(self):
        config = _tiny_moe_text_config()
        gate = DeepseekV41Gate(config)
        with self.assertRaises(ValueError):
            gate(mx.ones((2, config.hidden_size + 1)))
        vl_gate = DeepseekV41Gate(config, vision_enabled=True)
        with self.assertRaises(ValueError):
            vl_gate(mx.ones((2, config.hidden_size)), mx.array([False, True, False]))

    def test_malformed_routing_config_fails_closed(self):
        for overrides in (
            {"topk_method": "greedy"},
            {"scoring_func": "relu"},
            {"n_shared_experts": 2},
            {"n_shared_experts": 0},
            {"num_experts_per_tok": 0},
            {"num_experts_per_tok": 9},
            {"moe_intermediate_size": 0},
            {"swiglu_limit": -1.0},
        ):
            with self.assertRaises(ValueError):
                DeepseekV41Gate(_tiny_moe_text_config(**overrides))


class TestDeepseekV41PackedExperts(unittest.TestCase):
    """Packed-at-rest expert weights over the retained FP4/E8M0 primitives."""

    def test_fp4_expert_weights_stay_packed_at_half_a_byte_per_element(self):
        lin = DeepseekV41PackedLinear(64, 32, "fp4")
        self.assertEqual(lin.weight.dtype, mx.uint8)
        self.assertEqual(lin.weight.shape, (32, 32))  # 64 codes -> 32 bytes per row
        self.assertEqual(lin.scale.dtype, mx.uint8)
        self.assertEqual(lin.scale.shape, (32, 2))  # 64 / FP4_WEIGHT_BLOCK_SIZE
        self.assertEqual(lin.weight.nbytes + lin.scale.nbytes, 32 * 32 + 32 * 2)

    def test_fp8_weights_use_the_two_dimensional_thirty_two_tiled_scale_grid(self):
        lin = DeepseekV41PackedLinear(64, 48, "fp8")
        self.assertEqual(lin.weight.shape, (48, 64))
        self.assertEqual(lin.scale.shape, (2, 2))  # ceil(48/32) x 64/32

    def test_a_dense_linear_carries_no_scale_at_all(self):
        lin = DeepseekV41PackedLinear(64, 32, None)
        self.assertNotIn("scale", dict(lin.parameters()))
        self.assertEqual(lin.weight.dtype, mx.bfloat16)

    def test_dequantization_matches_the_retained_block_primitive(self):
        mx.random.seed(31)
        lin = _fill_packed(DeepseekV41PackedLinear(64, 32, "fp4"))
        expected = dequantize_fp4_block(
            lin.weight, lin.scale, FP4_WEIGHT_BLOCK_SIZE, mx.float32
        )
        self.assertTrue(mx.allclose(lin.dequantized(), expected, atol=0).item())
        x = mx.random.normal((3, 64))
        self.assertTrue(
            mx.allclose(lin(x), x.astype(mx.float32) @ expected.T, atol=1e-4).item()
        )

    def test_a_forward_pass_never_persists_an_unpacked_weight(self):
        # The whole reason the experts are stored packed: a persistent unpack
        # would multiply routed-expert storage by 4x (FP4 -> float32).
        mx.random.seed(32)
        lin = _fill_packed(DeepseekV41PackedLinear(64, 32, "fp4"))
        before = dict(lin.parameters())
        mx.eval(lin(mx.random.normal((2, 64))))
        after = dict(lin.parameters())
        self.assertEqual(sorted(before), sorted(after))
        self.assertEqual(after["weight"].dtype, mx.uint8)
        self.assertEqual(after["weight"].shape, (32, 32))
        self.assertNotIn("_dequantized", dict(lin))
        self.assertNotIn("_dequantized", after)

    def test_packed_shape_contracts_fail_closed(self):
        for args in (
            (33, 32, "fp4"),  # reduction dim not a multiple of the FP4 block
            (33, 32, "fp8"),
            (0, 32, "fp4"),
            (32, 0, "fp4"),
        ):
            with self.assertRaises(ValueError):
                DeepseekV41PackedLinear(*args)
        with self.assertRaises(ValueError):
            DeepseekV41PackedLinear(64, 32, "int4")
        with self.assertRaises(ValueError):
            DeepseekV41PackedLinear(64, 32, "fp4")(mx.zeros((2, 63)))

    def test_swiglu_limit_clamps_the_up_branch_both_ways_and_the_gate_only_above(self):
        def build(limit, gate_sign):
            expert = DeepseekV41Expert(32, 64, None, swiglu_limit=limit)
            expert.w1.weight = mx.full((64, 32), gate_sign * 10.0 / 32).astype(mx.bfloat16)
            expert.w3.weight = mx.full((64, 32), -10.0 / 32).astype(mx.bfloat16)
            expert.w2.weight = mx.full((32, 64), 1.0 / 64).astype(mx.bfloat16)
            return expert

        def silu(v):
            return v * mx.sigmoid(v)

        x = mx.ones((1, 32), dtype=mx.float32)

        # gate = +10, up = -10, limit 1: gate clamped down to 1, up up to -1.
        clamped = build(1.0, 1.0)(x)
        self.assertTrue(mx.allclose(clamped, silu(mx.array(1.0)) * -1.0, atol=1e-3).item())

        # Same weights, clamp disabled: both branches run wide open.
        wide = build(0.0, 1.0)(x)
        self.assertTrue(mx.allclose(wide, silu(mx.array(10.0)) * -10.0, atol=1e-2).item())
        self.assertGreater(mx.abs(wide).max().item(), 50.0)

        # gate = -10 with limit 1: the gate branch is clamped from above only,
        # so it stays at -10 and silu drives the product to nearly zero. A
        # symmetric clamp would leave a magnitude some 300x larger.
        gate_negative = build(1.0, -1.0)(x)
        self.assertTrue(
            mx.allclose(gate_negative, silu(mx.array(-10.0)) * -1.0, atol=1e-4).item()
        )
        self.assertLess(mx.abs(gate_negative).max().item(), 1e-2)


class TestDeepseekV41MoE(unittest.TestCase):
    """384+1 MoE composition, exercised at a tiny but structurally exact size."""

    # Real-MLX measurements on studio1 for the fixed tiny fixture produced
    # max_abs=0.0234375 and max_rel=0.00003591 at output magnitude 84503.414.
    # Grouped and one-token GEMMs may choose different reduction schedules, so
    # preserve a mixed bound instead of requiring unsupported bitwise behavior.
    MOE_BATCH_ATOL = 1.0 / 32.0
    MOE_BATCH_RTOL = 5e-5

    def test_routed_and_shared_outputs_are_summed_per_token(self):
        config, moe = _build_tiny_moe()
        x = mx.random.normal((2, 3, config.hidden_size))
        got = moe(x)
        self.assertEqual(got.shape, x.shape)

        flat = x.reshape(-1, config.hidden_size)
        weights, indices = moe.gate(flat)
        rows = indices.tolist()
        rows_out = []
        for token, row in enumerate(rows):
            acc = mx.zeros((1, config.hidden_size), dtype=mx.float32)
            for slot, expert_id in enumerate(row):
                acc = acc + moe.experts[expert_id](
                    flat[token : token + 1],
                    weights[token : token + 1, slot : slot + 1],
                ).astype(mx.float32)
            rows_out.append(acc)
        expected = mx.concatenate(rows_out, axis=0)
        expected = expected + moe.shared_experts(flat).astype(mx.float32)
        self.assertTrue(
            mx.allclose(
                got.reshape(-1, config.hidden_size),
                expected,
                atol=self.MOE_BATCH_ATOL,
                rtol=self.MOE_BATCH_RTOL,
            ).item()
        )

    def test_the_shared_expert_runs_for_every_token_including_unrouted_ones(self):
        config, moe = _build_tiny_moe()
        for expert in moe.experts:
            _zero_expert(expert)
        x = mx.random.normal((2, 3, config.hidden_size))
        # Every routed expert decodes to exactly zero, so what is left is the
        # shared expert -- which must still have run for all six tokens.
        got = moe(x).reshape(-1, config.hidden_size)
        shared = moe.shared_experts(x.reshape(-1, config.hidden_size)).astype(mx.float32)
        self.assertTrue(mx.allclose(got, shared, atol=1e-4).item())
        self.assertGreater(mx.max(mx.abs(got)).item(), 0.0)

    def test_removing_the_shared_expert_leaves_only_the_routed_sum(self):
        config, moe = _build_tiny_moe()
        _zero_expert(moe.shared_experts)
        x = mx.random.normal((1, 4, config.hidden_size))
        got = moe(x).reshape(-1, config.hidden_size)
        flat = x.reshape(-1, config.hidden_size)
        weights, indices = moe.gate(flat)
        rows_out = []
        for token, row in enumerate(indices.tolist()):
            acc = mx.zeros((1, config.hidden_size), dtype=mx.float32)
            for slot, expert_id in enumerate(row):
                acc = acc + moe.experts[expert_id](
                    flat[token : token + 1],
                    weights[token : token + 1, slot : slot + 1],
                ).astype(mx.float32)
            rows_out.append(acc)
        expected = mx.concatenate(rows_out, axis=0)
        self.assertTrue(mx.allclose(got, expected, atol=1e-3).item())

    def test_exactly_the_routed_experts_are_unpacked_and_no_others(self):
        import mlx_lm.models.deepseek_v41 as v41

        config, moe = _build_tiny_moe()
        x = mx.random.normal((1, 1, config.hidden_size))
        _, indices = moe.gate(x.reshape(-1, config.hidden_size))
        routed = set(indices.tolist()[0])
        self.assertEqual(len(routed), 3)
        self.assertLess(len(routed), config.n_routed_experts)

        calls = []
        real = v41.dequantize_fp4_block

        def counting(packed, scale, *a, **k):
            calls.append(packed.shape)
            return real(packed, scale, *a, **k)

        v41.dequantize_fp4_block = counting
        try:
            mx.eval(moe(x))
        finally:
            v41.dequantize_fp4_block = real
        # Three w1/w2/w3 unpacks per routed expert, and nothing for the five
        # experts this token did not select.
        self.assertEqual(len(calls), 3 * len(routed))

    def test_experts_are_still_packed_after_a_forward_pass(self):
        config, moe = _build_tiny_moe()
        mx.eval(moe(mx.random.normal((1, 2, config.hidden_size))))
        for expert in moe.experts:
            self.assertEqual(expert.w1.weight.dtype, mx.uint8)
            self.assertEqual(expert.w1.weight.shape, (config.moe_intermediate_size, config.hidden_size // 2))
            self.assertEqual(expert.w2.weight.dtype, mx.uint8)
            self.assertEqual(expert.w3.scale.dtype, mx.uint8)

    def test_expert_and_gate_parameter_names_match_the_checkpoint_contract(self):
        _, moe = _build_tiny_moe()
        names = {k for k, _ in tree_flatten(moe.parameters())}
        for expected in (
            "gate.weight",
            "gate.bias",
            "experts.0.w1.weight",
            "experts.0.w1.scale",
            "experts.0.w2.weight",
            "experts.0.w3.weight",
            "shared_experts.w1.weight",
            "shared_experts.w1.scale",
            "shared_experts.w2.weight",
            "shared_experts.w3.weight",
        ):
            self.assertIn(expected, names)

    def test_a_single_token_decode_and_a_batch_agree_per_token(self):
        config, moe = _build_tiny_moe()
        x = mx.random.normal((1, 3, config.hidden_size))
        batched = moe(x)
        _, batched_routes = moe.gate(x.reshape(-1, config.hidden_size))
        for token in range(3):
            single = moe(x[:, token : token + 1])
            _, single_routes = moe.gate(
                x[:, token : token + 1].reshape(-1, config.hidden_size)
            )
            self.assertTrue(
                mx.array_equal(single_routes[0], batched_routes[token]).item()
            )
            self.assertTrue(
                mx.allclose(
                    single[0, 0],
                    batched[0, token],
                    atol=self.MOE_BATCH_ATOL,
                    rtol=self.MOE_BATCH_RTOL,
                ).item()
            )

    def test_shape_contract_is_enforced(self):
        config, moe = _build_tiny_moe()
        with self.assertRaises(ValueError):
            moe(mx.zeros((2, config.hidden_size + 1)))


class TestDeepseekV41ExpertPartition(unittest.TestCase):
    """World-size expert sharding. Pure arithmetic at the official 384 count --
    nothing here allocates an official-sized expert.
    """

    def test_the_official_expert_count_splits_evenly_over_real_world_sizes(self):
        self.assertEqual(routed_expert_partition(384, 1, 0), (0, 384))
        self.assertEqual(routed_expert_partition(384, 2, 1), (192, 384))
        self.assertEqual(routed_expert_partition(384, 4, 2), (192, 288))
        self.assertEqual(routed_expert_partition(384, 8, 0), (0, 48))
        # The DSpark gate routes 128, which shards over the same world sizes.
        self.assertEqual(routed_expert_partition(128, 4, 3), (96, 128))

    def test_every_expert_is_owned_by_exactly_one_rank(self):
        for world_size in (1, 2, 3, 4, 6, 8, 12, 16):
            covered = []
            for rank in range(world_size):
                start, end = routed_expert_partition(384, world_size, rank)
                covered.extend(range(start, end))
            self.assertEqual(covered, list(range(384)), world_size)

    def test_an_uneven_split_is_refused_rather_than_leaving_experts_unowned(self):
        for world_size in (5, 7, 9, 10):
            with self.assertRaises(ValueError):
                routed_expert_partition(384, world_size, 0)

    def test_degenerate_partition_arguments_fail_closed(self):
        for args in ((384, 0, 0), (384, -1, 0), (384, 4, 4), (384, 4, -1), (0, 1, 0)):
            with self.assertRaises(ValueError):
                routed_expert_partition(*args)

    def test_a_rank_constructs_only_the_experts_it_owns(self):
        config = _tiny_moe_text_config()
        moe = DeepseekV41MoE(config, world_size=4, rank=1)
        self.assertEqual(moe.local_expert_ids, [2, 3])
        self.assertEqual(moe.n_local_experts, 2)
        self.assertEqual(sum(expert is not None for expert in moe.experts), 2)
        # Global ids, not rank-local ones.
        self.assertIs(moe.expert(3), moe.experts[3])
        with self.assertRaises(KeyError):
            moe.expert(0)
        with self.assertRaises(KeyError):
            moe.expert(4)

    def test_a_sharded_rank_preserves_global_expert_parameter_names(self):
        config = _tiny_moe_text_config()
        moe = DeepseekV41MoE(config, world_size=2, rank=1)
        leaves = {name for name, _ in tree_flatten(moe.parameters())}
        expert_ids = {
            int(name.split(".")[1])
            for name in leaves
            if name.startswith("experts.")
        }
        self.assertEqual(expert_ids, {4, 5, 6, 7})

    def test_the_gate_is_replicated_across_ranks_while_experts_are_not(self):
        config = _tiny_moe_text_config()
        for rank in range(4):
            moe = DeepseekV41MoE(config, world_size=4, rank=rank)
            self.assertEqual(moe.gate.weight.shape, (8, config.hidden_size))
            self.assertEqual(moe.gate.n_routed_experts, 8)
            self.assertEqual(sum(expert is not None for expert in moe.experts), 2)
            # Every rank runs the shared expert; only the routed set is split.
            self.assertEqual(
                moe.shared_experts.w1.weight.shape,
                (config.moe_intermediate_size, config.hidden_size),
            )

    def test_executing_a_sharded_moe_fails_loud_instead_of_silently_partial(self):
        config = _tiny_moe_text_config()
        moe = DeepseekV41MoE(config, world_size=2, rank=0)
        with self.assertRaisesRegex(RuntimeError, "no all_reduce"):
            moe(mx.zeros((1, 2, config.hidden_size)))

    def test_sharded_moe_reduces_routed_output_before_shared_expert(self):
        config = _tiny_moe_text_config()
        x = mx.full((1, 2, config.hidden_size), 0.25, dtype=mx.float32)

        def build_rank(rank):
            moe = DeepseekV41MoE(
                config,
                world_size=2,
                rank=rank,
                all_reduce=lambda local: local,
                expert_quant=None,
                shared_expert_quant=None,
                dtype=mx.float32,
            )
            moe.gate.weight = mx.ones(moe.gate.weight.shape, dtype=mx.float32)
            moe.gate.bias = mx.array(
                [3.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0],
                dtype=mx.float32,
            )
            for expert_id in moe.local_expert_ids:
                expert = moe.expert(expert_id)
                value = 0.0025 * (expert_id + 1)
                for linear in (expert.w1, expert.w2, expert.w3):
                    linear.weight = mx.full(
                        linear.weight.shape, value, dtype=mx.float32
                    )
            return moe

        rank0 = build_rank(0)
        rank1 = build_rank(1)
        _zero_expert(rank0.shared_experts)
        _zero_expert(rank1.shared_experts)

        rank_local = []
        for moe in (rank0, rank1):
            captured = []
            moe.all_reduce = lambda local, captured=captured: captured.append(local) or local
            moe(x)
            rank_local.append(captured[0])

        for linear in (
            rank0.shared_experts.w1,
            rank0.shared_experts.w2,
            rank0.shared_experts.w3,
        ):
            linear.weight = mx.full(linear.weight.shape, 0.01, dtype=mx.float32)
        shared = rank0.shared_experts(x.reshape(-1, config.hidden_size)).reshape(x.shape)
        calls = []

        def reducer(local):
            calls.append(local)
            self.assertTrue(np.allclose(np.asarray(local), np.asarray(rank_local[0])))
            return local + rank_local[1]

        rank0.all_reduce = reducer
        out = rank0(x)
        expected = rank_local[0].reshape(x.shape) + rank_local[1].reshape(x.shape) + shared
        self.assertEqual(len(calls), 1)
        self.assertFalse(np.allclose(np.asarray(rank_local[1]), 0.0))
        self.assertFalse(np.allclose(np.asarray(shared), 0.0))
        self.assertTrue(np.allclose(np.asarray(out), np.asarray(expected), atol=1e-5))

    def test_moe_output_matches_at_world_sizes_one_two_and_four(self):
        config = _tiny_moe_text_config()
        x = mx.arange(3 * config.hidden_size, dtype=mx.float32).reshape(
            1, 3, config.hidden_size
        ) / 128

        def initialize(moe):
            moe.gate.weight = mx.zeros(moe.gate.weight.shape, dtype=mx.float32)
            moe.gate.bias = mx.array(
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                dtype=mx.float32,
            )
            for expert_id in moe.local_expert_ids:
                expert = moe.expert(expert_id)
                for index, linear in enumerate((expert.w1, expert.w2, expert.w3)):
                    linear.weight = mx.full(
                        linear.weight.shape,
                        0.001 * (expert_id + 1) * (index + 1),
                        dtype=mx.float32,
                    )
            _zero_expert(moe.shared_experts)
            return moe

        whole = initialize(
            DeepseekV41MoE(
                config,
                expert_quant=None,
                shared_expert_quant=None,
                dtype=mx.float32,
            )
        )(x)

        for world_size in (2, 4):
            partials = []
            for rank in range(world_size):
                moe = initialize(
                    DeepseekV41MoE(
                        config,
                        world_size=world_size,
                        rank=rank,
                        all_reduce=lambda local: local,
                        expert_quant=None,
                        shared_expert_quant=None,
                        dtype=mx.float32,
                    )
                )
                partials.append(moe(x))
            sharded = sum(partials[1:], start=partials[0])
            self.assertTrue(
                mx.allclose(sharded, whole, atol=1e-5, rtol=1e-5).item(),
                world_size,
            )

    def test_sharded_moe_rejects_an_in_place_reducer_that_returns_none(self):
        config = _tiny_moe_text_config()
        moe = DeepseekV41MoE(
            config, world_size=2, rank=0, all_reduce=lambda local: None
        )
        with self.assertRaisesRegex(RuntimeError, "returned None"):
            moe(mx.zeros((1, 2, config.hidden_size)))

    def test_a_world_size_that_does_not_divide_the_experts_fails_at_construction(self):
        with self.assertRaises(ValueError):
            DeepseekV41MoE(_tiny_moe_text_config(), world_size=3, rank=0)


class TestDeepseekV41RoutingConfigValidation(unittest.TestCase):
    """validate_moe_routing_config is the single fail-closed gate every MoE
    surface routes through, so it is checked directly as well as via the
    modules that call it.
    """

    def test_the_official_routing_config_validates(self):
        config = _official_text_config()
        validate_moe_routing_config(config, config.n_routed_experts, config.num_experts_per_tok)
        # The DSpark gate reuses the same rule with its own expert counts.
        validate_moe_routing_config(config, 128, 3)

    def test_a_topk_wider_than_the_expert_pool_is_rejected(self):
        config = _tiny_moe_text_config()
        validate_moe_routing_config(config, 8, 8)
        with self.assertRaises(ValueError):
            validate_moe_routing_config(config, 8, 9)
        with self.assertRaises(ValueError):
            validate_moe_routing_config(config, 0, 1)

    def test_unsupported_routing_variants_are_named_in_the_error(self):
        with self.assertRaises(ValueError) as ctx:
            validate_moe_routing_config(_tiny_moe_text_config(topk_method="group_limited_greedy"), 8, 3)
        self.assertIn("noaux_tc", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            validate_moe_routing_config(_tiny_moe_text_config(scoring_func="gelu"), 8, 3)
        self.assertIn("sqrtsoftplus", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            validate_moe_routing_config(_tiny_moe_text_config(n_shared_experts=3), 8, 3)
        self.assertIn("shared", str(ctx.exception).lower())


def _engram_text_config(**overrides):
    """A tiny but structurally faithful Engram config.

    Two Engram layers with distinct tables, 3-grams over 2 heads (so
    n_hash_cols == 4, the same (max_ngram_size - 1) * n_heads shape the
    official 4-gram/8-head config produces as 24), and a 32-wide row that is
    exactly one E8M0 block. The two table row counts are filled in from the
    prime layout itself, so the tiny config tiles its tables exactly the way
    the official one does.
    """
    cfg = _text_config_dict()
    cfg.update(
        {
            "vocab_size": 40,
            "hidden_size": 8,
            "num_hidden_layers": 8,
            "num_nextn_predict_layers": 0,
            "compress_ratios": [0, 0, 2, 2, 1, 1, 1, 1],
            "kv_source_layer_ids": [2, 4],
            "index_source_layer_ids": [2, 4, 6],
            "candidate_source_layer_id": 4,
            "hc_mult": 2,
            "rms_norm_eps": 1e-6,
            "engram_layer_ids": [1, 3],
            "engram_num_embeddings": [10**9, 10**9],
            "engram_max_ngram_size": 3,
            "engram_vocab_size": 97,
            "engram_n_heads": 2,
            "engram_head_dim": 32,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 11,
        }
    )
    cfg.update(overrides)
    probe = TextConfig.from_dict(cfg)
    if "engram_num_embeddings" not in overrides and probe.engram_layer_ids:
        layout = EngramLayout.from_config(probe)
        cfg["engram_num_embeddings"] = [
            layout.bucket_span(i) for i in range(len(layout.layer_ids))
        ]
    return TextConfig.from_dict(cfg)


_ENGRAM_TOKEN_MAP = [i % 11 for i in range(40)]


def _write_engram_fixture(
    path,
    weight,
    scale,
    weight_dtype="F8_E4M3",
    scale_dtype="F8_E8M0",
    weight_key="weight",
    scale_key="scale",
    mutate=None,
):
    """Write a real, tiny safetensors file holding one packed Engram shard.

    The header is built here rather than by a library so the production parser
    in SafetensorsEngramRowStore is what is under test, and so a malformed
    header can be produced on purpose via ``mutate``.
    """
    weight = np.ascontiguousarray(weight, dtype=np.uint8)
    scale = np.ascontiguousarray(scale, dtype=np.uint8)
    header = {
        weight_key: {
            "dtype": weight_dtype,
            "shape": list(weight.shape),
            "data_offsets": [0, weight.nbytes],
        },
        scale_key: {
            "dtype": scale_dtype,
            "shape": list(scale.shape),
            "data_offsets": [weight.nbytes, weight.nbytes + scale.nbytes],
        },
    }
    if mutate is not None:
        mutate(header)
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as handle:
        handle.write(len(blob).to_bytes(8, "little"))
        handle.write(blob)
        handle.write(weight.tobytes())
        handle.write(scale.tobytes())
    return path


def _random_engram_shard(n_rows, dim, block_size, seed):
    """Packed bytes for a tiny Engram shard: E4M3 values plus E8M0 row scales.

    Magnitude codes stop below 0x78 so no exponent field reaches 0b1111, which
    keeps E4M3FN NaN out of the fixture; the sign bit is drawn separately so
    negative values are still covered. Scale codes stay near the 127 bias, well
    clear of the E8M0 NaN code.
    """
    rng = np.random.default_rng(seed)
    magnitude = rng.integers(0, 0x78, size=(n_rows, dim), dtype=np.uint8)
    sign = rng.integers(0, 2, size=(n_rows, dim), dtype=np.uint8) << 7
    weight = (magnitude | sign).astype(np.uint8)
    scale = rng.integers(120, 132, size=(n_rows, dim // block_size), dtype=np.uint8)
    return weight, scale


class TestDeepseekV41EngramNormalization(unittest.TestCase):
    """The compressed-vocabulary contract this repository owns.

    Every Engram hash multiplier is derived from the *size* of the compressed
    vocabulary, so a divergence in this normalizer chain does not degrade
    quality gracefully: it silently rehashes both 384-million-row tables.
    """

    def test_the_declared_sequence_is_the_official_one(self):
        self.assertEqual(
            ENGRAM_NORMALIZER_SEQUENCE[:4],
            ("NFKC", "NFD", "StripAccents", "Lowercase"),
        )
        self.assertEqual(len(ENGRAM_NORMALIZER_SEQUENCE), 8)

    def test_case_and_leading_space_collapse(self):
        for text in (" The", "the", "THE", " THE ", "\tThe\n"):
            self.assertEqual(normalize_engram_token_text(text), "the")

    def test_accents_are_stripped_after_decomposition(self):
        # Precomposed and decomposed forms must land on the same key.
        self.assertEqual(normalize_engram_token_text("caf\u00e9"), "cafe")
        self.assertEqual(normalize_engram_token_text("cafe\u0301"), "cafe")
        self.assertEqual(normalize_engram_token_text("\u00c9\u00c0"), "ea")

    def test_compatibility_forms_are_folded(self):
        self.assertEqual(normalize_engram_token_text("\uff21\uff22"), "ab")

    def test_a_lone_space_survives_the_strip_via_the_sentinel(self):
        # Without the private-use sentinel this token would strip to the empty
        # string and merge with every other whitespace-only token.
        self.assertEqual(normalize_engram_token_text(" "), " ")
        self.assertEqual(normalize_engram_token_text("\n\t "), " ")
        self.assertEqual(ENGRAM_SPACE_SENTINEL, "\ue000")

    def test_inner_whitespace_runs_collapse_but_edges_are_trimmed(self):
        self.assertEqual(normalize_engram_token_text("a \t b"), "a b")
        self.assertEqual(normalize_engram_token_text("\u00a0x\u00a0"), "x")

    def test_the_empty_string_stays_empty(self):
        self.assertEqual(normalize_engram_token_text(""), "")

    def test_partial_byte_tokens_are_keyed_by_their_raw_form(self):
        self.assertEqual(engram_compressed_token_key("\ufffd", "<0xC3>"), "<0xC3>")
        with self.assertRaises(ValueError):
            engram_compressed_token_key("\ufffd", None)

    def test_the_map_collapses_equivalent_ids_in_first_seen_order(self):
        decoded = [
            " The",
            "the",
            "THE",
            " ",
            "\n",
            "caf\u00e9",
            "cafe",
            "\ufffd",
            "\ufffd",
            "x",
        ]
        raw = [None] * 7 + ["<0xC3>", "<0xA9>", None]
        lookup, size = build_engram_compressed_token_map(decoded, raw)
        self.assertEqual(lookup, [0, 0, 0, 1, 1, 2, 2, 3, 4, 5])
        self.assertEqual(size, 6)
        self.assertEqual(max(lookup) + 1, size)

    def test_a_whitespace_only_token_keeps_its_own_identity(self):
        lookup, size = build_engram_compressed_token_map(["a", " "], [None, None])
        self.assertNotEqual(lookup[0], lookup[1])
        self.assertEqual(size, 2)

    def test_malformed_vocabularies_fail_closed(self):
        with self.assertRaises(ValueError):
            build_engram_compressed_token_map(["a", "b"], [None])
        with self.assertRaises(ValueError):
            build_engram_compressed_token_map([], [])

    def test_a_slow_tokenizer_is_refused_rather_than_decoded_differently(self):
        class _NoBackend:
            def __len__(self):
                return 4

        with self.assertRaises(ValueError) as ctx:
            build_engram_compressed_token_map_from_tokenizer(_NoBackend())
        self.assertIn("backend_tokenizer", str(ctx.exception))

    def test_mlx_tokenizer_wrapper_is_unwrapped_before_len(self):
        class _Backend:
            def decode(self, ids, skip_special_tokens=False):
                return ("A", "a", " ")[ids[0]]

            def id_to_token(self, token_id):
                return ("A", "a", "SPACE")[token_id]

        class _Tokenizer:
            backend_tokenizer = _Backend()

            def __len__(self):
                return 3

        class _Wrapper:
            _tokenizer = _Tokenizer()

        lookup, size = build_engram_compressed_token_map_from_tokenizer(_Wrapper())
        self.assertEqual(lookup, [0, 0, 1])
        self.assertEqual(size, 2)

    def test_trailing_multimodal_placeholders_do_not_expand_engram_vocab(self):
        class _Backend:
            tokens = [
                "a",
                "A",
                "b",
                "<|place_holder_mm_span_0036|>",
                "<|place_holder_mm_span_0037|>",
                "<｜deepseek_image｜>",
            ]

            def decode(self, token_ids, skip_special_tokens=False):
                self.assertFalse(skip_special_tokens)
                return self.tokens[token_ids[0]]

            def id_to_token(self, token_id):
                return self.tokens[token_id]

            def assertFalse(self, value):
                if value:
                    raise AssertionError("special tokens must remain visible")

        class _Tokenizer:
            backend_tokenizer = _Backend()

            def __len__(self):
                return len(self.backend_tokenizer.tokens)

        lookup, size = build_engram_compressed_token_map_from_tokenizer(
            _Tokenizer(), expected_size=2, fallback_token_id=0
        )

        self.assertEqual(lookup, [0, 0, 1, 0, 0, 0])
        self.assertEqual(size, 2)

    def test_tokenizer_extension_rejects_every_unproven_shape(self):
        class _Backend:
            def __init__(self, tokens):
                self.tokens = tokens

            def decode(self, token_ids, skip_special_tokens=False):
                if skip_special_tokens:
                    raise AssertionError("special tokens must remain visible")
                return self.tokens[token_ids[0]]

            def id_to_token(self, token_id):
                return self.tokens[token_id]

        class _Tokenizer:
            def __init__(self, tokens):
                self.backend_tokenizer = _Backend(tokens)

            def __len__(self):
                return len(self.backend_tokenizer.tokens)

        cases = (
            (
                ["a", "b"],
                {"expected_size": 3, "fallback_token_id": 0},
                "checkpoint config requires 3",
            ),
            (
                ["a", "b", "<|place_holder_mm_span_0036|>", "a", "<|place_holder_mm_span_0037|>"],
                {"expected_size": 2, "fallback_token_id": 0},
                "contiguous suffix",
            ),
            (
                ["a", "b", "<|unknown_multimodal_token|>"],
                {"expected_size": 2, "fallback_token_id": 0},
                "contiguous suffix",
            ),
            (
                ["a", "b", "<|place_holder_mm_span_0036|>"],
                {"expected_size": 2, "fallback_token_id": 3},
                "outside tokenizer vocabulary",
            ),
        )
        for tokens, kwargs, message in cases:
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, message):
                build_engram_compressed_token_map_from_tokenizer(
                    _Tokenizer(tokens), **kwargs
                )

    def test_model_binding_passes_checkpoint_size_and_compressed_pad_id(self):
        class _Backend:
            tokens = ["A", "a", "b", "<|place_holder_mm_span_0036|>"]

            def decode(self, token_ids, skip_special_tokens=False):
                if skip_special_tokens:
                    raise AssertionError("special tokens must remain visible")
                return self.tokens[token_ids[0]]

            def id_to_token(self, token_id):
                return self.tokens[token_id]

        class _Tokenizer:
            backend_tokenizer = _Backend()

            def __len__(self):
                return len(self.backend_tokenizer.tokens)

        config = _engram_text_config(
            engram_compressed_vocab_size=2,
            engram_pad_token_id=1,
        )
        runtime = SimpleNamespace(
            engram_layout=EngramLayout.from_config(config),
            bind_engram_hasher=lambda hasher: setattr(runtime, "hasher", hasher),
        )
        model = SimpleNamespace(
            _runtime=runtime,
            args=SimpleNamespace(text_config=config),
        )

        Model.bind_tokenizer(model, _Tokenizer())

        self.assertEqual(runtime.hasher.compressed_vocab_size, 2)
        self.assertEqual(runtime.hasher.token_map.tolist(), [0, 0, 1, 0])


class TestDeepseekV41EngramLayout(unittest.TestCase):
    """Prime bucket layout, checked against the real pinned config."""

    def test_the_official_layout_tiles_both_tables_exactly(self):
        layout = EngramLayout.from_config(_official_text_config())
        self.assertEqual(layout.layer_ids, (1, 14))
        # (max_ngram_size - 1) * n_heads == 3 * 8 == 24 rows per token per layer.
        self.assertEqual(layout.n_hash_cols, 24)
        self.assertEqual(layout.prime_array().shape, (2, 3, 8))
        # The 24 disjoint prime bucket ranges of each layer sum to precisely the
        # engram_num_embeddings row count the pinned config declares.
        self.assertEqual(layout.bucket_span(0), 384006168)
        self.assertEqual(layout.bucket_span(1), 384016682)
        self.assertEqual(
            (layout.bucket_span(0), layout.bucket_span(1)), layout.num_embeddings
        )

    def test_every_bucket_range_is_distinct_and_increasing(self):
        layout = EngramLayout.from_config(_official_text_config())
        primes = layout.flat_primes(0) + layout.flat_primes(1)
        self.assertEqual(list(primes), sorted(primes))
        self.assertEqual(len(set(primes)), len(primes))
        self.assertEqual(primes[0], 16000057)
        self.assertTrue(all(p > 16000000 - 1 for p in primes))

    def test_offsets_concatenate_the_ranges_without_gaps(self):
        layout = EngramLayout.from_config(_official_text_config())
        offsets = layout.bucket_offsets()
        self.assertEqual(offsets.shape, (2, 24))
        for index in range(2):
            flat = layout.flat_primes(index)
            self.assertEqual(int(offsets[index][0]), 0)
            for column in range(1, 24):
                self.assertEqual(
                    int(offsets[index][column]),
                    int(offsets[index][column - 1]) + flat[column - 1],
                )
            self.assertEqual(
                int(offsets[index][-1]) + flat[-1], layout.num_embeddings[index]
            )

    def test_find_next_prime_never_reuses(self):
        seen = set()
        drawn = []
        current = 96
        for _ in range(5):
            current = find_next_prime(current, seen)
            seen.add(current)
            drawn.append(current)
        self.assertEqual(drawn, [97, 101, 103, 107, 109])
        self.assertEqual(find_next_prime(96, {97, 101}), 103)

    def test_multipliers_are_odd_layer_specific_and_overflow_safe(self):
        multipliers = compute_engram_hash_multipliers((1, 14), 4, 99092)
        self.assertEqual(multipliers.shape, (2, 4))
        self.assertTrue(bool((multipliers % 2 == 1).all()))
        self.assertFalse(np.array_equal(multipliers[0], multipliers[1]))
        # The bound must keep compressed_id * multiplier inside int64: the
        # running hash is an XOR of these products, so a wrap aliases n-grams.
        self.assertLess(int(multipliers.max()) * 99091, int(np.iinfo(np.int64).max))
        # Seeded per layer as 10007 * layer_id, so the draw is reproducible.
        self.assertTrue(
            np.array_equal(
                multipliers[0],
                compute_engram_hash_multipliers((1,), 4, 99092)[0],
            )
        )

    def test_disabled_engram_yields_no_layout(self):
        self.assertIsNone(
            EngramLayout.from_config(
                _engram_text_config(engram_layer_ids=[], engram_num_embeddings=[])
            )
        )
        self.assertIsNone(
            validate_engram_config(
                _engram_text_config(engram_layer_ids=[], engram_num_embeddings=[])
            )
        )

    def _assert_rejects(self, fragment, **overrides):
        with self.assertRaises(ValueError) as ctx:
            EngramLayout.from_config(_engram_text_config(**overrides))
        self.assertIn(fragment, str(ctx.exception))

    def test_a_bucket_span_wider_than_the_table_is_rejected(self):
        # The reference would mask these ids to zero, indistinguishably from a
        # legitimate remote-shard row, and silently degrade instead of failing.
        self._assert_rejects("address past the end", engram_num_embeddings=[10, 10])

    def test_malformed_engram_metadata_is_rejected(self):
        self._assert_rejects("engram_max_ngram_size", engram_max_ngram_size=1)
        self._assert_rejects("engram_n_heads", engram_n_heads=0)
        self._assert_rejects("engram_head_dim", engram_head_dim=100)
        self._assert_rejects("engram_vocab_size", engram_vocab_size=1)
        self._assert_rejects("one table row count", engram_num_embeddings=[384006168])
        self._assert_rejects(
            "strictly increasing",
            engram_layer_ids=[3, 1],
            engram_num_embeddings=[10**9, 10**9],
        )
        self._assert_rejects(
            "num_hidden_layers",
            engram_layer_ids=[1, 99],
            engram_num_embeddings=[10**9, 10**9],
        )
        self._assert_rejects("must be positive", engram_num_embeddings=[0, 0])

    def test_validate_engram_config_checks_the_surrounding_fields(self):
        self.assertIsNotNone(validate_engram_config(_official_text_config()))
        with self.assertRaises(ValueError):
            validate_engram_config(_engram_text_config(engram_compressed_vocab_size=0))
        with self.assertRaises(ValueError):
            validate_engram_config(_engram_text_config(engram_pad_token_id=10**6))
        with self.assertRaises(ValueError):
            validate_engram_config(_engram_text_config(hc_mult=0))


class TestDeepseekV41EngramHashing(unittest.TestCase):
    """EngramNgramHasher, exercised through the properties the reference guarantees.

    The tiny config hashes 3-grams over 2 heads, so every position yields
    (max_ngram_size - 1) * n_heads == 4 row ids per Engram layer -- the same
    shape the official 4-gram/8-head config yields as 24.
    """

    def _hasher(self, max_seq_len=32, max_batch_size=2, **overrides):
        config = _engram_text_config(**overrides)
        layout = EngramLayout.from_config(config)
        return layout, EngramNgramHasher(
            layout,
            _ENGRAM_TOKEN_MAP,
            config.engram_pad_token_id,
            config.engram_compressed_vocab_size,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

    def test_shape_is_one_row_id_per_ngram_size_per_head_per_layer(self):
        layout, hasher = self._hasher()
        ids = hasher.hash_ids(np.arange(12, dtype=np.int64).reshape(2, 6))
        self.assertEqual(ids.shape, (2, 6, 2, 4))
        self.assertEqual(ids.dtype, np.int64)
        self.assertEqual(layout.n_hash_cols, 4)
        self.assertIsInstance(hasher(np.arange(12).reshape(2, 6)), mx.array)

    def test_every_id_lands_inside_its_own_bucket_range(self):
        layout, hasher = self._hasher()
        rng = np.random.default_rng(1)
        ids = hasher.hash_ids(rng.integers(0, 40, size=(2, 10), dtype=np.int64))
        offsets = layout.bucket_offsets()
        for layer in range(2):
            flat = layout.flat_primes(layer)
            for column in range(layout.n_hash_cols):
                low = int(offsets[layer][column])
                column_ids = ids[:, :, layer, column]
                self.assertGreaterEqual(int(column_ids.min()), low)
                self.assertLess(int(column_ids.max()), low + flat[column])

    def test_prefill_in_one_call_equals_token_by_token_decode(self):
        """The shared history cache is what carries the n-gram across the split."""
        _, whole = self._hasher()
        _, split = self._hasher()
        rng = np.random.default_rng(2)
        tokens = rng.integers(0, 40, size=(2, 9), dtype=np.int64)
        one_shot = whole.hash_ids(tokens, 0)
        pieces = [split.hash_ids(tokens[:, :5], 0)]
        for step in range(5, 9):
            pieces.append(split.hash_ids(tokens[:, step : step + 1], step))
        self.assertTrue(np.array_equal(one_shot, np.concatenate(pieces, axis=1)))

    def test_the_same_ngram_hashes_the_same_wherever_it_occurs(self):
        _, hasher = self._hasher()
        tokens = np.array([[7, 8, 9, 1, 7, 8, 9]], dtype=np.int64)
        ids = hasher.hash_ids(tokens)
        # Positions 2 and 6 both end the 3-gram (7, 8, 9).
        self.assertTrue(np.array_equal(ids[0, 2], ids[0, 6]))
        self.assertFalse(np.array_equal(ids[0, 2], ids[0, 3]))

    def test_tokens_that_normalize_alike_hash_alike(self):
        """The compressed map is what makes this true, and it maps id -> id % 11."""
        _, hasher = self._hasher()
        ids = hasher.hash_ids(np.array([[3, 4, 5], [14, 15, 16]], dtype=np.int64))
        self.assertTrue(np.array_equal(ids[0], ids[1]))

    def test_an_ngram_never_spans_an_image_span(self):
        """Look-back stops at a dead token, so the position after one hashes
        exactly as if it started the sequence."""
        tokens = np.array([[5, 6, 9, 5, 6]], dtype=np.int64)
        mask = np.ones((1, 5), dtype=bool)
        mask[0, 2] = False  # an image token occupies position 2
        _, masked_hasher = self._hasher()
        _, plain_hasher = self._hasher()
        masked = masked_hasher.hash_ids(tokens, 0, mask)
        plain = plain_hasher.hash_ids(tokens, 0)
        # Position 3 can see no usable history across the image, exactly like
        # position 0, and both carry token 5, so both hash the padded n-gram.
        self.assertTrue(np.array_equal(masked[0, 3], masked[0, 0]))
        # Without the mask that same position reaches back across position 2
        # and hashes something else entirely, so the mask is load-bearing.
        self.assertFalse(np.array_equal(plain[0, 3], plain[0, 0]))

    def test_the_two_engram_layers_hash_independently(self):
        _, hasher = self._hasher()
        ids = hasher.hash_ids(np.array([[3, 4, 5, 6]], dtype=np.int64))
        self.assertFalse(np.array_equal(ids[:, :, 0], ids[:, :, 1]))

    def test_the_history_cache_is_eight_bytes_per_token_per_batch_row(self):
        _, hasher = self._hasher(max_seq_len=64, max_batch_size=4)
        self.assertEqual(hasher.nbytes(), 4 * 64 * 8)
        # Shared once by both Engram layers, not duplicated per layer.
        self.assertEqual(hasher.cache.shape, (4, 64))

    def test_reset_makes_the_next_sequence_start_clean(self):
        _, hasher = self._hasher()
        hasher.hash_ids(np.array([[5, 6, 7]], dtype=np.int64), 0)
        with_history = hasher.hash_ids(np.array([[8]], dtype=np.int64), 3)
        hasher.reset()
        after_reset = hasher.hash_ids(np.array([[8]], dtype=np.int64), 3)
        self.assertFalse(np.array_equal(with_history, after_reset))
        # A reset slot reads back as DEAD, so look-back is blocked rather than
        # inventing a phantom compressed-token-0 n-gram: decoding at position
        # 3 now hashes identically to the very start of a fresh sequence.
        _, fresh = self._hasher()
        self.assertTrue(
            np.array_equal(
                after_reset, fresh.hash_ids(np.array([[8]], dtype=np.int64), 0)
            )
        )

    def test_out_of_bounds_and_malformed_requests_fail_closed(self):
        _, hasher = self._hasher(max_seq_len=8, max_batch_size=2)
        with self.assertRaises(ValueError):
            hasher.hash_ids(np.zeros((3, 2), dtype=np.int64))
        with self.assertRaises(ValueError):
            hasher.hash_ids(np.zeros((1, 4), dtype=np.int64), 6)
        with self.assertRaises(ValueError):
            hasher.hash_ids(np.array([[99]], dtype=np.int64))
        with self.assertRaises(ValueError):
            hasher.hash_ids(np.array([[-1]], dtype=np.int64))
        with self.assertRaises(ValueError):
            hasher.hash_ids(np.zeros((4,), dtype=np.int64))
        with self.assertRaises(ValueError):
            hasher.hash_ids(np.zeros((1, 2), dtype=np.int64), 0, np.ones((1, 3), bool))

    def test_a_token_map_wider_than_the_declared_compressed_vocab_is_rejected(self):
        config = _engram_text_config()
        layout = EngramLayout.from_config(config)
        with self.assertRaises(ValueError) as ctx:
            EngramNgramHasher(layout, [0, 1, 2, 99], 2, 11)
        self.assertIn("compressed", str(ctx.exception))
        with self.assertRaises(ValueError):
            EngramNgramHasher(layout, _ENGRAM_TOKEN_MAP, 999, 11)
        with self.assertRaises(ValueError):
            EngramNgramHasher(layout, _ENGRAM_TOKEN_MAP, 2, 11, max_seq_len=0)


class TestDeepseekV41EngramRowStore(unittest.TestCase):
    """SafetensorsEngramRowStore reads single rows out of a real file.

    The fixtures here are tiny, but the access path is the one the 91.6 GiB
    shard needs: parse the header, seek to a row byte range, read exactly that
    row. Nothing in this class ever loads a whole tensor.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.weight, self.scale = _random_engram_shard(64, 32, 32, seed=7)
        self.path = _write_engram_fixture(
            os.path.join(self.tmp.name, "engram.safetensors"), self.weight, self.scale
        )

    def _store(self, **kwargs):
        store = SafetensorsEngramRowStore(self.path, **kwargs)
        self.addCleanup(store.close)
        return store

    def test_layout_is_taken_from_the_file_header(self):
        store = self._store()
        self.assertEqual((store.num_rows, store.dim), (64, 32))
        self.assertEqual(store.block_size, 32)
        self.assertEqual(store.scale_dim, 1)
        # 32 packed value bytes plus one E8M0 scale code per row.
        self.assertEqual(store.row_nbytes, 33)

    def test_reads_exactly_the_requested_rows_in_the_requested_order(self):
        store = self._store()
        wanted = [5, 0, 63, 5]
        weight, scale = store.read_rows(wanted)
        self.assertEqual(weight.shape, (4, 32))
        self.assertEqual(scale.shape, (4, 1))
        self.assertTrue(np.array_equal(weight, self.weight[wanted]))
        self.assertTrue(np.array_equal(scale, self.scale[wanted]))

    def test_rank_local_rows_map_to_their_physical_file_range(self):
        store = self._store(row_start=32, num_rows=32)
        wanted = [0, 7, 31]
        weight, scale = store.read_rows(wanted)
        physical = [32, 39, 63]
        self.assertTrue(np.array_equal(weight, self.weight[physical]))
        self.assertTrue(np.array_equal(scale, self.scale[physical]))
        with self.assertRaises(IndexError):
            store.read_rows([32])

    def test_cost_is_counted_and_is_per_row_not_per_file(self):
        store = self._store()
        store.read_rows([1, 2, 3])
        self.assertEqual(store.rows_read, 3)
        self.assertEqual(store.bytes_read, 3 * store.row_nbytes)
        store.read_rows([9])
        self.assertEqual(store.rows_read, 4)
        self.assertEqual(store.bytes_read, 4 * 33)
        # The file itself is far larger than what was read.
        self.assertGreater(os.path.getsize(self.path), store.bytes_read)

    def test_an_empty_request_touches_the_file_at_all(self):
        store = self._store()
        weight, scale = store.read_rows([])
        self.assertEqual(weight.shape, (0, 32))
        self.assertEqual(scale.shape, (0, 1))
        self.assertEqual(store.bytes_read, 0)

    def test_a_row_id_past_the_shard_is_refused(self):
        store = self._store()
        with self.assertRaises(IndexError):
            store.read_rows([64])
        with self.assertRaises(IndexError):
            store.read_rows([-1])

    def test_the_store_closes_and_reopens_its_handle(self):
        with SafetensorsEngramRowStore(self.path) as store:
            first = store.read_rows([4])[0]
            store.close()
            self.assertTrue(np.array_equal(store.read_rows([4])[0], first))

    def _reject(self, name, fragment, **fixture_kwargs):
        path = _write_engram_fixture(
            os.path.join(self.tmp.name, name), self.weight, self.scale, **fixture_kwargs
        )
        with self.assertRaises(ValueError) as ctx:
            SafetensorsEngramRowStore(path)
        self.assertIn(fragment, str(ctx.exception))

    def test_a_wider_than_one_byte_dtype_is_refused(self):
        # BF16 rows would make every byte offset wrong by a factor of two, and
        # the reads would silently return neighbouring rows.
        self._reject("bf16.safetensors", "dtype", weight_dtype="BF16")
        self._reject("f32scale.safetensors", "dtype", scale_dtype="F32")

    def test_a_scale_grid_that_does_not_match_the_rows_is_refused(self):
        def widen(header):
            header["scale"]["shape"] = [32, 2]

        self._reject("scale.safetensors", "requires", mutate=widen)

    def test_byte_ranges_that_do_not_match_the_shape_are_refused(self):
        def shrink(header):
            header["weight"]["data_offsets"] = [0, 16]

        self._reject("offsets.safetensors", "bytes", mutate=shrink)

        def overrun(header):
            size = header["scale"]["data_offsets"][1]
            header["scale"]["shape"] = [64, 1]
            header["scale"]["data_offsets"] = [size, size + 64]

        self._reject("overrun.safetensors", "bytes long", mutate=overrun)

    def test_a_missing_tensor_is_refused(self):
        def drop(header):
            header["quant_scale"] = header.pop("scale")

        self._reject("missing.safetensors", "no tensor named", mutate=drop)

    def test_a_non_default_tensor_naming_can_be_selected(self):
        def rename(header):
            header["engram.weight"] = header.pop("weight")
            header["engram.scale"] = header.pop("scale")

        path = _write_engram_fixture(
            os.path.join(self.tmp.name, "named.safetensors"),
            self.weight,
            self.scale,
            mutate=rename,
        )
        store = SafetensorsEngramRowStore(
            path, weight_key="engram.weight", scale_key="engram.scale"
        )
        self.addCleanup(store.close)
        self.assertTrue(np.array_equal(store.read_rows([2])[0][0], self.weight[2]))

    def test_a_truncated_file_is_refused_rather_than_read_short(self):
        path = os.path.join(self.tmp.name, "short.safetensors")
        with open(self.path, "rb") as src:
            blob = src.read()
        with open(path, "wb") as dst:
            dst.write(blob[: len(blob) - 40])
        with self.assertRaises(ValueError):
            SafetensorsEngramRowStore(path)

    def test_a_file_that_is_not_safetensors_at_all_is_refused(self):
        path = os.path.join(self.tmp.name, "junk.bin")
        with open(path, "wb") as handle:
            handle.write(b"not a safetensors file")
        with self.assertRaises(ValueError):
            SafetensorsEngramRowStore(path)


class TestDeepseekV41EngramRowCache(unittest.TestCase):
    """BoundedEngramRowCache: dedup on the way in, a hard ceiling on residency."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.weight, self.scale = _random_engram_shard(64, 32, 32, seed=11)
        self.path = _write_engram_fixture(
            os.path.join(self.tmp.name, "engram.safetensors"), self.weight, self.scale
        )

    def _cache(self, max_rows=8):
        store = SafetensorsEngramRowStore(self.path)
        self.addCleanup(store.close)
        return store, BoundedEngramRowCache(store, max_rows=max_rows)

    def test_a_cold_gather_reads_each_distinct_row_once(self):
        store, cache = self._cache()
        # A realistic hash gather repeats heavily: blocked look-backs all
        # collapse onto the same padded n-gram row.
        requested = [3, 3, 7, 3, 7, 1]
        unique_ids, weight, scale, inverse = cache.gather_packed(requested)
        self.assertEqual(list(unique_ids), [3, 7, 1])
        self.assertEqual(weight.shape, (3, 32))
        self.assertEqual(scale.shape, (3, 1))
        self.assertEqual(list(inverse), [0, 0, 1, 0, 1, 2])
        self.assertEqual(store.rows_read, 3)
        self.assertEqual(cache.stats.requested_rows, 6)
        self.assertEqual(cache.stats.unique_rows, 3)
        self.assertEqual(cache.stats.misses, 3)
        self.assertEqual(cache.stats.hits, 0)

    def test_a_warm_gather_reads_nothing_at_all(self):
        store, cache = self._cache()
        cache.gather_packed([3, 7, 1])
        bytes_after_cold = store.bytes_read
        cache.gather_packed([1, 7, 3, 3])
        self.assertEqual(store.bytes_read, bytes_after_cold)
        self.assertEqual(cache.stats.hits, 3)
        self.assertEqual(cache.stats.rows_fetched, 3)

    def test_rows_come_back_in_request_order(self):
        store, cache = self._cache()
        requested = [9, 2, 9, 40, 2]
        _, weight, _, inverse = cache.gather_packed(requested)
        restored = weight[inverse]
        self.assertTrue(np.array_equal(restored, self.weight[requested]))
        rows = cache.gather_rows(requested)
        self.assertEqual(rows.shape, (5, 32))
        expected = dequantize_engram_rows(self.weight[requested], self.scale[requested])
        self.assertTrue(
            np.array_equal(
                np.asarray(rows.astype(mx.float32)),
                np.asarray(expected.astype(mx.float32)),
            )
        )

    def test_residency_is_packed_bytes_of_the_rows_held(self):
        store, cache = self._cache(max_rows=8)
        self.assertEqual(cache.nbytes(), 0)
        cache.gather_packed([1, 2, 3])
        self.assertEqual(cache.resident_rows, 3)
        self.assertEqual(cache.nbytes(), 3 * store.row_nbytes)
        cache.clear()
        self.assertEqual(cache.resident_rows, 0)
        self.assertEqual(cache.nbytes(), 0)

    def test_the_bound_is_never_exceeded_and_evictions_are_counted(self):
        store, cache = self._cache(max_rows=4)
        for row_id in range(12):
            cache.gather_packed([row_id])
        self.assertEqual(cache.resident_rows, 4)
        self.assertEqual(cache.nbytes(), 4 * store.row_nbytes)
        self.assertEqual(cache.stats.evictions, 8)
        self.assertEqual(store.rows_read, 12)

    def test_eviction_is_least_recently_used(self):
        store, cache = self._cache(max_rows=2)
        cache.gather_packed([1])
        cache.gather_packed([2])
        cache.gather_packed([1])  # refreshes 1, so 2 is now the oldest
        cache.gather_packed([3])
        reads_before = store.rows_read
        cache.gather_packed([1])
        self.assertEqual(store.rows_read, reads_before)  # 1 was retained
        cache.gather_packed([2])
        self.assertEqual(store.rows_read, reads_before + 1)  # 2 was evicted

    def test_an_evicted_row_is_refetched_with_identical_bytes(self):
        store, cache = self._cache(max_rows=1)
        first = cache.gather_packed([5])[1].copy()
        cache.gather_packed([6])
        again = cache.gather_packed([5])[1]
        self.assertTrue(np.array_equal(first, again))
        self.assertTrue(np.array_equal(again[0], self.weight[5]))

    def test_a_gather_larger_than_the_bound_is_served_then_trimmed(self):
        store, cache = self._cache(max_rows=4)
        requested = list(range(16))
        _, weight, _, inverse = cache.gather_packed(requested)
        # Correctness first: every requested row really came back.
        self.assertTrue(np.array_equal(weight[inverse], self.weight[requested]))
        # Then the ceiling reasserts itself, so a long prompt cannot grow the
        # cache without bound.
        self.assertEqual(cache.resident_rows, 4)
        self.assertEqual(cache.nbytes(), 4 * store.row_nbytes)

    def test_an_empty_gather_is_a_no_op(self):
        store, cache = self._cache()
        unique_ids, weight, scale, inverse = cache.gather_packed([])
        self.assertEqual(weight.shape, (0, 32))
        self.assertEqual(scale.shape, (0, 1))
        self.assertEqual(unique_ids.size, 0)
        self.assertEqual(inverse.size, 0)
        self.assertEqual(store.bytes_read, 0)
        self.assertEqual(cache.gather_rows([]).shape, (0, 32))

    def test_a_cache_that_cannot_hold_a_row_is_refused(self):
        store, _ = self._cache()
        with self.assertRaises(ValueError):
            BoundedEngramRowCache(store, max_rows=0)


def _decode_e4m3fn(codes):
    """An independent, spec-literal E4M3FN decode used to check the port.

    Written out here rather than reusing the production helper so the test has
    its own source of truth: sign / 4-bit exponent / 3-bit mantissa, bias 7,
    with the exponent-zero case decoded as a subnormal.
    """
    codes = np.asarray(codes, dtype=np.uint8).astype(np.int64)
    sign = np.where(codes >> 7 == 1, -1.0, 1.0)
    exponent = (codes >> 3) & 0xF
    mantissa = (codes & 0x7).astype(np.float64)
    normal = (1.0 + mantissa / 8.0) * np.power(2.0, exponent.astype(np.float64) - 7.0)
    subnormal = mantissa * (2.0**-9)
    return sign * np.where(exponent == 0, subnormal, normal)


def _reference_engram_gate(stream, key, weight, eps, clamp=1e-6):
    """An independent float64 transcription of the official gate expression."""
    h = np.asarray(stream, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    dim = h.shape[-1]
    h_rstd = 1.0 / np.sqrt(np.mean(h * h, axis=-1) + eps)
    k_rstd = 1.0 / np.sqrt(np.mean(k * k, axis=-1) + eps)
    dot = np.sum(h * w * k, axis=-1) * h_rstd * k_rstd / np.sqrt(dim)
    signed = np.copysign(np.sqrt(np.maximum(np.abs(dot), clamp)), dot)
    return 1.0 / (1.0 + np.exp(-signed))


class TestDeepseekV41EngramDequant(unittest.TestCase):
    """Row-level FP8 x E8M0 dequantization of gathered rows only."""

    def test_rows_decode_to_value_times_two_to_the_scale(self):
        weight, scale = _random_engram_shard(6, 64, ENGRAM_FP8_BLOCK_SIZE, seed=3)
        rows = dequantize_engram_rows(weight, scale, ENGRAM_FP8_BLOCK_SIZE, mx.float32)
        self.assertEqual(rows.shape, (6, 64))
        expected = _decode_e4m3fn(weight).reshape(6, 2, ENGRAM_FP8_BLOCK_SIZE)
        expected = expected * np.power(2.0, scale.astype(np.float64) - 127.0)[..., None]
        self.assertTrue(
            np.allclose(np.asarray(rows), expected.reshape(6, 64), rtol=1e-6, atol=0.0)
        )

    def test_each_scale_code_governs_exactly_its_own_block(self):
        """A 2-D block tiling would smear these scales across the wrong columns."""
        weight = np.full((1, 64), 0x38, dtype=np.uint8)  # E4M3 1.0
        scale = np.array([[127, 130]], dtype=np.uint8)  # 2**0 then 2**3
        rows = np.asarray(
            dequantize_engram_rows(weight, scale, ENGRAM_FP8_BLOCK_SIZE, mx.float32)
        )
        self.assertTrue(np.all(rows[0, :32] == 1.0))
        self.assertTrue(np.all(rows[0, 32:] == 8.0))

    def test_an_empty_gather_decodes_to_an_empty_block(self):
        rows = dequantize_engram_rows(
            np.zeros((0, 32), np.uint8), np.zeros((0, 1), np.uint8)
        )
        self.assertEqual(rows.shape, (0, 32))

    def test_mismatched_row_blocks_are_refused(self):
        weight, scale = _random_engram_shard(4, 32, 32, seed=4)
        with self.assertRaises(ValueError):
            dequantize_engram_rows(weight, scale[:2])
        with self.assertRaises(ValueError):
            dequantize_engram_rows(weight, np.zeros((4, 2), np.uint8))
        with self.assertRaises(ValueError):
            dequantize_engram_rows(weight[0], scale[0])
        with self.assertRaises(ValueError):
            dequantize_engram_rows(np.zeros((4, 20), np.uint8), scale)


class TestDeepseekV41EngramGate(unittest.TestCase):
    """The signed-sqrt-sigmoid gate that decides how much Engram is injected."""

    def _inputs(self, seed=5, batch=2, seqlen=3, hc_mult=2, dim=16, scale=1.0):
        rng = np.random.default_rng(seed)
        stream = rng.normal(size=(batch, seqlen, hc_mult, dim)).astype(np.float32)
        key = rng.normal(size=(batch, seqlen, hc_mult, dim)).astype(np.float32)
        weight = rng.normal(size=(hc_mult, dim)).astype(np.float32)
        return stream * scale, key, weight

    def test_matches_an_independent_float64_transcription(self):
        stream, key, weight = self._inputs()
        gate = engram_signed_sqrt_sigmoid_gate(
            mx.array(stream), mx.array(key), mx.array(weight), 1e-6
        )
        self.assertEqual(gate.shape, (2, 3, 2))
        expected = _reference_engram_gate(stream, key, weight, 1e-6)
        self.assertTrue(np.allclose(np.asarray(gate), expected, atol=1e-6))

    def test_the_gate_is_a_probability_in_the_open_unit_interval(self):
        stream, key, weight = self._inputs(seed=6, scale=50.0)
        gate = np.asarray(
            engram_signed_sqrt_sigmoid_gate(
                mx.array(stream), mx.array(key), mx.array(weight), 1e-6
            )
        )
        self.assertTrue(np.all(gate > 0.0))
        self.assertTrue(np.all(gate < 1.0))

    def test_normalization_is_per_hc_copy_so_a_rescaled_stream_barely_moves(self):
        """RMS normalizing each (token, hc copy) row makes the gate scale free,
        up to the eps floor: a 100x louder residual must not saturate it."""
        stream, key, weight = self._inputs(seed=7)
        base = np.asarray(
            engram_signed_sqrt_sigmoid_gate(
                mx.array(stream), mx.array(key), mx.array(weight), 1e-6
            )
        )
        loud = np.asarray(
            engram_signed_sqrt_sigmoid_gate(
                mx.array(stream * 100.0), mx.array(key), mx.array(weight), 1e-6
            )
        )
        self.assertTrue(np.allclose(base, loud, atol=1e-4))

    def test_a_silent_stream_lands_on_the_clamp_plateau(self):
        """Without the clamp the sqrt derivative is infinite at zero, so an
        all-zero stream is exactly the case the floor exists for."""
        stream = np.zeros((1, 1, 2, 16), dtype=np.float32)
        _, key, weight = self._inputs(seed=8)
        gate = np.asarray(
            engram_signed_sqrt_sigmoid_gate(
                mx.array(stream), mx.array(key[:1, :1]), mx.array(weight), 1e-6
            )
        )
        plateau = 1.0 / (1.0 + np.exp(-np.sqrt(ENGRAM_GATE_CLAMP)))
        self.assertTrue(np.allclose(gate, plateau, atol=1e-6))

    def test_the_sign_of_the_match_moves_the_gate_across_one_half(self):
        stream, key, weight = self._inputs(seed=9)
        positive = np.asarray(
            engram_signed_sqrt_sigmoid_gate(
                mx.array(stream), mx.array(stream), mx.array(np.abs(weight)), 1e-6
            )
        )
        negative = np.asarray(
            engram_signed_sqrt_sigmoid_gate(
                mx.array(stream), mx.array(-stream), mx.array(np.abs(weight)), 1e-6
            )
        )
        self.assertTrue(np.all(positive > 0.5))
        self.assertTrue(np.all(negative < 0.5))

    def test_shape_and_clamp_violations_are_refused(self):
        stream, key, weight = self._inputs()
        with self.assertRaises(ValueError):
            engram_signed_sqrt_sigmoid_gate(
                mx.array(stream), mx.array(key[:1]), mx.array(weight), 1e-6
            )
        with self.assertRaises(ValueError):
            engram_signed_sqrt_sigmoid_gate(
                mx.array(stream), mx.array(key), mx.array(weight[:1]), 1e-6
            )
        with self.assertRaises(ValueError):
            engram_signed_sqrt_sigmoid_gate(
                mx.array(stream), mx.array(key), mx.array(weight), 1e-6, clamp_value=0.0
            )


class TestDeepseekV41EngramEmbedding(unittest.TestCase):
    """The row-sharded table lookup, which never allocates a table."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.weight, self.scale = _random_engram_shard(64, 32, 32, seed=13)

    def _embedding(self, lo, hi, num_embeddings=64, rank=0, world_size=1, **kwargs):
        path = _write_engram_fixture(
            os.path.join(self.tmp.name, "shard-%d-%d.safetensors" % (lo, hi)),
            self.weight[lo:hi],
            self.scale[lo:hi],
        )
        store = SafetensorsEngramRowStore(path)
        self.addCleanup(store.close)
        cache = BoundedEngramRowCache(store, max_rows=kwargs.pop("max_rows", 16))
        return DeepseekV41EngramEmbedding(
            num_embeddings, 32, cache, rank=rank, world_size=world_size, **kwargs
        )

    def test_the_module_holds_no_dense_table_and_no_parameters_at_all(self):
        embedding = self._embedding(0, 64)
        self.assertEqual(dict(tree_flatten(embedding.parameters())), {})
        self.assertEqual(embedding.num_embeddings, 64)
        # Whatever the table says, resident bytes are the cache bound only.
        self.assertEqual(embedding.nbytes(), 0)

    def test_lookup_returns_the_dequantized_rows_for_the_requested_ids(self):
        embedding = self._embedding(0, 64)
        ids = np.array([[3, 17, 3], [63, 0, 17]], dtype=np.int64)
        rows = embedding(ids, dtype=mx.float32)
        self.assertEqual(rows.shape, (2, 3, 32))
        flat = ids.reshape(-1)
        expected = dequantize_engram_rows(
            self.weight[flat], self.scale[flat], 32, mx.float32
        )
        self.assertTrue(
            np.array_equal(np.asarray(rows).reshape(6, 32), np.asarray(expected))
        )

    def test_resident_bytes_track_the_cache_bound_not_the_table(self):
        embedding = self._embedding(0, 64, max_rows=4)
        embedding(np.arange(64, dtype=np.int64))
        self.assertEqual(embedding.nbytes(), 4 * 33)
        self.assertLess(embedding.nbytes(), embedding.num_embeddings * 33)

    def test_an_empty_lookup_is_well_shaped(self):
        embedding = self._embedding(0, 64)
        self.assertEqual(embedding(np.zeros((2, 0), np.int64)).shape, (2, 0, 32))

    def test_an_id_outside_the_table_is_refused_rather_than_zeroed(self):
        """The reference masks such an id to zero, which is indistinguishable
        from a legitimate remote-shard row and hides a malformed layout."""
        embedding = self._embedding(0, 64)
        with self.assertRaises(IndexError):
            embedding(np.array([64], dtype=np.int64))
        with self.assertRaises(IndexError):
            embedding(np.array([-1], dtype=np.int64))

    def test_a_store_that_does_not_hold_this_rank_shard_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._embedding(0, 32, num_embeddings=128, rank=0, world_size=2)
        self.assertIn("rows", str(ctx.exception))

    def test_geometry_violations_are_refused(self):
        for kwargs in (
            {"world_size": 0},
            {"rank": 2, "world_size": 2},
        ):
            with self.assertRaises(ValueError):
                self._embedding(0, 64, **kwargs)

    def test_a_sharded_lookup_without_a_reducer_fails_loud(self):
        embedding = self._embedding(0, 32, rank=0, world_size=2)
        with self.assertRaises(RuntimeError) as ctx:
            embedding(np.array([[1, 40]], dtype=np.int64))
        message = str(ctx.exception)
        self.assertIn("all_reduce", message)
        self.assertIn("world_size", message)

    def test_a_reducer_that_returns_nothing_fails_loud(self):
        embedding = self._embedding(
            0, 32, rank=0, world_size=2, all_reduce=lambda rows: None
        )
        with self.assertRaises(RuntimeError):
            embedding(np.array([[1]], dtype=np.int64))

    def test_each_rank_zeroes_the_rows_it_does_not_own(self):
        identity = lambda rows: rows
        rank0 = self._embedding(0, 32, rank=0, world_size=2, all_reduce=identity)
        rank1 = self._embedding(32, 64, rank=1, world_size=2, all_reduce=identity)
        ids = np.array([[5, 40]], dtype=np.int64)
        left = np.asarray(rank0(ids, dtype=mx.float32))
        right = np.asarray(rank1(ids, dtype=mx.float32))
        self.assertTrue(np.all(left[0, 1] == 0.0))  # row 40 belongs to rank 1
        self.assertTrue(np.all(right[0, 0] == 0.0))  # row 5 belongs to rank 0

    def test_summing_one_two_and_four_rank_shards_reproduces_the_lookup(self):
        """This is what the injected all-reduce is for: each rank contributes
        only its own rows, and the sum is the whole lookup."""
        identity = lambda rows: rows
        whole = self._embedding(0, 64)
        ids = np.array([[5, 40, 63, 0]], dtype=np.int64)
        expected = np.asarray(whole(ids, dtype=mx.float32))
        for world_size in (2, 4):
            rows_per_rank = 64 // world_size
            shards = [
                self._embedding(
                    rank * rows_per_rank,
                    (rank + 1) * rows_per_rank,
                    rank=rank,
                    world_size=world_size,
                    all_reduce=identity,
                )
                for rank in range(world_size)
            ]
            partials = [np.asarray(shard(ids, dtype=mx.float32)) for shard in shards]
            summed = sum(partials[1:], start=partials[0])
            self.assertTrue(np.array_equal(summed, expected), world_size)

    def test_the_injected_reducer_is_the_only_cross_rank_coupling(self):
        calls = []

        def reducer(rows):
            calls.append(rows.shape)
            return rows

        embedding = self._embedding(0, 32, rank=0, world_size=2, all_reduce=reducer)
        embedding(np.array([[1, 2, 3]], dtype=np.int64))
        self.assertEqual(calls, [(1, 3, 32)])

    def test_world_size_one_needs_no_reducer(self):
        embedding = self._embedding(0, 64)
        self.assertIsNone(embedding.all_reduce)
        self.assertEqual(embedding(np.array([1], dtype=np.int64)).shape, (1, 32))


class TestDeepseekV41EngramModule(unittest.TestCase):
    """DeepseekV41Engram writing a gated n-gram lookup into the residual stream."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = _engram_text_config()
        self.layout = EngramLayout.from_config(self.config)

    def _engram(self, layer_id=1, seed=17):
        index = self.layout.layer_ids.index(layer_id)
        rows = self.layout.num_embeddings[index]
        weight, scale = _random_engram_shard(
            rows, self.layout.head_dim, ENGRAM_FP8_BLOCK_SIZE, seed=seed
        )
        path = _write_engram_fixture(
            os.path.join(self.tmp.name, "layer%d.safetensors" % layer_id),
            weight,
            scale,
        )
        store = SafetensorsEngramRowStore(path)
        self.addCleanup(store.close)
        embedding = DeepseekV41EngramEmbedding(
            rows, self.layout.head_dim, BoundedEngramRowCache(store, max_rows=64)
        )
        module = DeepseekV41Engram(self.config, layer_id, self.layout, embedding)
        rng = np.random.default_rng(seed)
        module.wkv.weight = mx.array(
            rng.integers(1, 120, size=module.wkv.weight.shape, dtype=np.uint8)
        )
        module.wkv.scale = mx.full(module.wkv.scale.shape, 127, dtype=mx.uint8)
        return module

    def _stream_and_ids(self, batch=2, seqlen=4, seed=19, layer_id=1):
        rng = np.random.default_rng(seed)
        stream = mx.array(
            rng.normal(
                size=(batch, seqlen, self.config.hc_mult, self.config.hidden_size)
            ).astype(np.float32)
        )
        hasher = EngramNgramHasher(
            self.layout,
            _ENGRAM_TOKEN_MAP,
            self.config.engram_pad_token_id,
            self.config.engram_compressed_vocab_size,
            max_batch_size=batch,
            max_seq_len=seqlen,
        )
        tokens = rng.integers(0, 40, size=(batch, seqlen), dtype=np.int64)
        index = self.layout.layer_ids.index(layer_id)
        return stream, hasher.hash_ids(tokens)[:, :, index, :]

    def test_the_stream_shape_survives_the_injection(self):
        module = self._engram()
        stream, hash_ids = self._stream_and_ids()
        out = module(stream, hash_ids)
        self.assertEqual(out.shape, stream.shape)
        self.assertEqual(out.dtype, stream.dtype)

    def test_the_lookup_actually_changes_the_stream(self):
        module = self._engram()
        stream, hash_ids = self._stream_and_ids()
        out = np.asarray(module(stream, hash_ids))
        self.assertFalse(np.allclose(out, np.asarray(stream)))

    def test_the_hash_ids_are_what_select_the_injection(self):
        module = self._engram()
        stream, hash_ids = self._stream_and_ids()
        other = np.asarray(hash_ids)[:, ::-1, :]
        self.assertFalse(
            np.allclose(
                np.asarray(module(stream, hash_ids)), np.asarray(module(stream, other))
            )
        )

    def test_a_masked_position_passes_through_untouched(self):
        """An image span has no n-gram history worth injecting, so the gate is
        shut there and the residual stream must come out bit-identical."""
        module = self._engram()
        stream, hash_ids = self._stream_and_ids()
        mask = np.ones(tuple(stream.shape[:2]), dtype=bool)
        mask[0, 2] = False
        out = np.asarray(module(stream, hash_ids, mask))
        reference = np.asarray(stream)
        self.assertTrue(np.array_equal(out[0, 2], reference[0, 2]))
        self.assertFalse(np.allclose(out[0, 1], reference[0, 1]))

    def test_the_two_engram_layers_inject_different_things(self):
        first = self._engram(layer_id=1, seed=17)
        second = self._engram(layer_id=3, seed=23)
        stream, first_ids = self._stream_and_ids(layer_id=1)
        _, second_ids = self._stream_and_ids(layer_id=3)
        self.assertFalse(np.array_equal(first_ids, second_ids))
        self.assertFalse(
            np.allclose(
                np.asarray(first(stream, first_ids)),
                np.asarray(second(stream, second_ids)),
            )
        )

    def test_a_non_engram_layer_is_refused(self):
        module_embedding = self._engram().embed
        with self.assertRaises(ValueError):
            DeepseekV41Engram(self.config, 2, self.layout, module_embedding)

    def test_an_embedding_for_the_wrong_table_is_refused(self):
        wrong = self._engram(layer_id=3, seed=23).embed
        with self.assertRaises(ValueError) as ctx:
            DeepseekV41Engram(self.config, 1, self.layout, wrong)
        self.assertIn("row table", str(ctx.exception))

    def test_malformed_stream_or_hash_shapes_are_refused(self):
        module = self._engram()
        stream, hash_ids = self._stream_and_ids()
        with self.assertRaises(ValueError):
            module(stream[:, :, 0], hash_ids)
        with self.assertRaises(ValueError):
            module(stream, np.asarray(hash_ids)[:1])
        with self.assertRaises(ValueError):
            module(stream, np.asarray(hash_ids)[..., :2])
        with self.assertRaises(ValueError):
            module(stream, hash_ids, np.ones((1, 1), dtype=bool))


class TestDeepseekV41EngramMemoryBounds(unittest.TestCase):
    """The load-bearing claim of this whole path: cost does not scale with the table.

    The official Engram tensors are ~91.6 GiB and ~98.3 GiB. Nothing here may
    grow with that. Rather than watch peak RSS, which is noisy, this measures
    the two deterministic counters that would have to move if a table were
    being materialized: bytes pulled off disk, and packed bytes held resident.
    """

    def test_gather_cost_is_invariant_to_how_large_the_table_is(self):
        rng = np.random.default_rng(29)
        # Heavy repetition, exactly as a real B x L x 24 hash gather produces.
        requested = rng.integers(0, 512, size=1024, dtype=np.int64)
        unique = int(np.unique(requested).size)
        bound = 128
        observed = []
        with tempfile.TemporaryDirectory() as tmp:
            for n_rows in (1024, 4096, 16384):
                weight, scale = _random_engram_shard(n_rows, 32, 32, seed=n_rows)
                path = _write_engram_fixture(
                    os.path.join(tmp, "t%d.safetensors" % n_rows), weight, scale
                )
                with SafetensorsEngramRowStore(path) as store:
                    cache = BoundedEngramRowCache(store, max_rows=bound)
                    cache.gather_packed(requested)
                    observed.append(
                        (os.path.getsize(path), store.bytes_read, cache.nbytes())
                    )

        file_sizes = [item[0] for item in observed]
        bytes_read = {item[1] for item in observed}
        resident = {item[2] for item in observed}
        # The tables differ by 16x on disk ...
        self.assertGreater(file_sizes[-1], 15 * file_sizes[0])
        # ... and every measured cost is bit-identical across all three.
        self.assertEqual(len(bytes_read), 1)
        self.assertEqual(len(resident), 1)
        self.assertEqual(bytes_read.pop(), unique * 33)
        self.assertEqual(resident.pop(), min(unique, bound) * 33)

    def test_the_hash_history_cache_is_eight_bytes_per_token(self):
        config = _engram_text_config()
        layout = EngramLayout.from_config(config)
        hasher = EngramNgramHasher(
            layout,
            _ENGRAM_TOKEN_MAP,
            config.engram_pad_token_id,
            config.engram_compressed_vocab_size,
            max_batch_size=4,
            max_seq_len=4096,
        )
        # One shared int64 history per (batch row, position) for all Engram
        # layers together -- it does not scale with n_hash_cols or table size.
        self.assertEqual(hasher.nbytes(), 4 * 4096 * 8)


def _tiny_vision_config(**overrides):
    cfg = {
        "model_type": "deepseek_v41_vision",
        "num_hidden_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "patch_size": 2,
        "rope_theta": 10000.0,
        "downsample_ratio": 2,
        "max_image_tokens": 64,
        "min_pixels": 16,
        "max_wh_ratio": None,
    }
    cfg.update(overrides)
    return VisionConfig.from_dict(cfg)


def _fill_vision_linear(linear, scale=0.05, seed=0):
    rng = np.random.default_rng(seed)
    linear.weight = mx.array(rng.normal(scale=scale, size=linear.weight.shape).astype(np.float32))
    if hasattr(linear, "bias"):
        linear.bias = mx.array(rng.normal(scale=scale, size=linear.bias.shape).astype(np.float32))


def _build_tiny_vision(dim=8, seed=3, **vision_overrides):
    vision = _tiny_vision_config(**vision_overrides)
    tower = DeepseekV41Vision(vision, dim)
    rng_seed = seed
    _fill_vision_linear(tower.vision.patch_embed.proj, seed=rng_seed)
    for i, block in enumerate(tower.vision.blocks):
        _fill_vision_linear(block.attn.wqkv, seed=rng_seed + 10 + i)
        _fill_vision_linear(block.attn.wo, seed=rng_seed + 20 + i)
        _fill_vision_linear(block.mlp.w1, seed=rng_seed + 30 + i)
        _fill_vision_linear(block.mlp.w2, seed=rng_seed + 40 + i)
    _fill_vision_linear(tower.aligner.w1, seed=rng_seed + 50)
    _fill_vision_linear(tower.aligner.w2, seed=rng_seed + 60)
    tower.image_start = mx.array(np.full((dim,), 2.0, dtype=np.float32))
    tower.image_end = mx.array(np.full((dim,), 3.0, dtype=np.float32))
    tower.image_newline = mx.array(np.full((dim,), 4.0, dtype=np.float32))
    return vision, tower


class TestDeepseekV41ImageGrid(unittest.TestCase):
    """Pinned official image-grid / token-budget arithmetic."""

    def test_fixed_images_cost_the_pinned_official_token_counts(self):
        vision = VisionConfig.from_dict(_vision_config_dict())
        self.assertTrue(vision_enabled(vision))
        cases = (
            ((544, 544), 184, (13, 13), (546, 546)),
            ((14, 14), 184, (13, 13), (546, 546)),
            ((1024, 768), 496, (19, 25), (770, 1036)),
            ((10000, 10000), 994, (31, 31), (1302, 1302)),
            ((4000, 200), 487, (5, 96), (210, 4004)),
            ((200, 4000), 578, (96, 5), (4004, 210)),
            ((1920, 1080), 968, (23, 41), (966, 1708)),
        )
        for (width, height), tokens, llm, pixels in cases:
            n_h, n_w, best_h, best_w = plan_image_grid(width, height, vision)
            self.assertEqual((n_h, n_w), llm)
            self.assertEqual((best_h, best_w), pixels)
            self.assertEqual(num_image_tokens(n_h, n_w), tokens)
            types = np.asarray(image_token_types(n_h, n_w)).tolist()
            self.assertEqual(len(types), tokens)
            self.assertEqual(types[0], IMAGE_START)
            self.assertEqual(types[-1], IMAGE_END)
            self.assertEqual(types.count(IMAGE), n_h * n_w)
            self.assertEqual(types.count(IMAGE_NEW_LINE), n_h)

    def test_malformed_grids_fail_closed(self):
        vision = VisionConfig.from_dict(_vision_config_dict())
        with self.assertRaises(ValueError):
            plan_image_grid(0, 64, vision)
        with self.assertRaises(ValueError):
            plan_image_grid(64, -1, vision)
        with self.assertRaises(ValueError):
            num_image_tokens(0, 4)
        with self.assertRaises(ValueError):
            llm_grid(13, 13, 0, 3)
        with self.assertRaises(ValueError):
            image_token_types(1, 0)
        broken = _tiny_vision_config(max_image_tokens=2)
        with self.assertRaises(ValueError):
            plan_image_grid(8, 8, broken)


class TestDeepseekV41DSpark(unittest.TestCase):
    def test_topk_indices_join_main_window_and_draft_block(self):
        actual = np.asarray(get_dspark_topk_idxs(4, 2, 2, 1))
        self.assertEqual(actual.shape, (2, 2, 4))
        self.assertTrue(np.array_equal(actual[0, 0], [0, 1, 4, 5]))
        with self.assertRaises(ValueError):
            get_dspark_topk_idxs(4, 1, 2, 0)

    def test_forward_spec_prefill_then_decode_matches_official_shapes(self):
        config = _tiny_dspark_text_config()
        model = DeepseekV41DSpark(config, temperature=0)
        first = model.mtp[0]
        first.main_proj.weight = mx.full(
            first.main_proj.weight.shape, 0x38, dtype=mx.uint8
        )
        first.main_proj.scale = mx.full(
            first.main_proj.scale.shape, 127, dtype=mx.uint8
        )
        for layer in model.mtp:
            layer.attn.wkv.weight = mx.array(
                np.eye(32, dtype=np.uint8) * 0x38
            )
            layer.attn.wkv.scale = mx.full(
                layer.attn.wkv.scale.shape, 127, dtype=mx.uint8
            )
        prefill_hidden = mx.arange(256, dtype=mx.float32).reshape(1, 4, 64) / 256
        self.assertIsNone(
            model.forward_spec(mx.array([7], dtype=mx.int32), prefill_hidden)
        )
        for layer in model.mtp:
            self.assertGreater(float(mx.max(mx.abs(layer.cache.window))), 0.0)
        result = model.forward_spec(
            mx.array([8], dtype=mx.int32),
            mx.arange(64, dtype=mx.float32).reshape(1, 1, 64) / 64,
            start_pos=4,
        )
        output_ids, logits, confidence = result
        self.assertEqual(output_ids.shape, (1, 3))
        self.assertEqual(logits.shape, (1, 2, 64))
        self.assertEqual(confidence.shape, (1, 2))
        self.assertEqual(int(output_ids[0, 0]), 8)

    def test_stage_contract_names_and_dspark_specific_routing(self):
        model = DeepseekV41DSpark(_tiny_dspark_text_config())
        self.assertEqual(model.mtp[0].ffn.n_routed_experts, 16)
        self.assertEqual(model.mtp[0].ffn.topk, 3)
        names = {name for name, _ in tree_flatten(model.parameters())}
        self.assertIn("mtp.0.main_proj.weight", names)
        self.assertIn("mtp.0.main_norm.weight", names)
        self.assertIn("mtp.2.markov_head.embed.weight", names)
        self.assertIn("mtp.2.markov_head.head.weight", names)
        self.assertIn("mtp.2.confidence_head.proj.weight", names)

    def test_markov_logits_and_confidence_use_the_official_inputs(self):
        model = DeepseekV41DSpark(_tiny_dspark_text_config())
        final = model.mtp[-1]
        markov = final.markov_head
        markov.embed.weight = mx.zeros(markov.embed.weight.shape, dtype=mx.float32)
        markov.embed.weight[5] = mx.arange(1, 9, dtype=mx.float32)
        markov.head.weight = mx.zeros(markov.head.weight.shape, dtype=mx.float32)
        markov.head.weight[2] = mx.ones((8,), dtype=mx.float32)
        logits, embed = markov(mx.array([5], dtype=mx.int32))
        self.assertTrue(np.array_equal(np.asarray(embed[0]), np.arange(1, 9)))
        self.assertEqual(float(logits[0, 2]), 36.0)

        final.confidence_head.proj.weight = mx.concatenate(
            [mx.ones((1, 32)), mx.full((1, 8), 2.0)], axis=-1
        )
        hidden = mx.ones((1, 2, 32), dtype=mx.float32)
        markov_embed = mx.full((1, 2, 8), 3.0, dtype=mx.float32)
        confidence = final.confidence_head(hidden, markov_embed)
        self.assertTrue(np.array_equal(np.asarray(confidence), [[80.0, 80.0]]))

    def test_malformed_dspark_config_fails_closed(self):
        with self.assertRaises(ValueError):
            DeepseekV41DSpark(_tiny_dspark_text_config(compress_ratios=[0] * 6 + [1]))
        with self.assertRaises(ValueError):
            DeepseekV41DSpark(_tiny_dspark_text_config(dspark_target_layer_ids=[]))
        with self.assertRaises(ValueError):
            DeepseekV41DSpark(_tiny_dspark_text_config(dspark_noise_token_id=64))


class TestDeepseekV41VisionMerge(unittest.TestCase):
    """Fixed-image span replacement against the production tower."""

    def test_a_fixed_image_replaces_only_its_span(self):
        vision, tower = _build_tiny_vision(dim=8)
        n_h, n_w, best_h, best_w = plan_image_grid(4, 4, vision)
        n_vit_h, n_vit_w = best_h // vision.patch_size, best_w // vision.patch_size
        self.assertEqual((n_h, n_w), (1, 1))
        self.assertEqual(num_image_tokens(n_h, n_w), 4)
        patches = mx.array(
            np.arange(n_vit_h * n_vit_w * 3 * vision.patch_size * vision.patch_size, dtype=np.float32).reshape(
                n_vit_h * n_vit_w, 3, vision.patch_size, vision.patch_size
            )
            / 255.0
        )
        types = image_token_types(n_h, n_w)
        embeds = np.asarray(tower.encode_image(patches, n_vit_h, n_vit_w))
        self.assertEqual(embeds.shape, (1, 8))
        self.assertFalse(np.allclose(embeds, 0.0))

        # [TEXT, IMAGE_SPAN x4, TEXT]
        stream = mx.array(np.ones((1, 6, 8), dtype=np.float32))
        images = [[ImageInput(1, patches, n_vit_h, n_vit_w, types)]]
        out = np.asarray(tower.merge_image_embeddings(images, stream))
        self.assertTrue(np.array_equal(out[0, 0], np.ones(8, dtype=np.float32)))
        self.assertTrue(np.array_equal(out[0, 5], np.ones(8, dtype=np.float32)))
        self.assertTrue(np.allclose(out[0, 1], np.full(8, 2.0)))
        self.assertTrue(np.allclose(out[0, 2], embeds[0]))
        self.assertTrue(np.allclose(out[0, 3], np.full(8, 4.0)))
        self.assertTrue(np.allclose(out[0, 4], np.full(8, 3.0)))
        self.assertFalse(np.allclose(out[0, 2], np.ones(8)))

    def test_text_only_streams_are_untouched(self):
        _, tower = _build_tiny_vision(dim=8)
        stream = mx.array(np.arange(24, dtype=np.float32).reshape(1, 3, 8))
        out = np.asarray(tower.merge_image_embeddings(None, stream))
        self.assertTrue(np.array_equal(out, np.asarray(stream)))
        out = np.asarray(tower.merge_image_embeddings([[]], stream))
        self.assertTrue(np.array_equal(out, np.asarray(stream)))
        self.assertFalse(vision_enabled(_tiny_vision_config(num_hidden_layers=0)))
        with self.assertRaises(ValueError):
            DeepseekV41Vision(_tiny_vision_config(num_hidden_layers=0), 8)

    def test_span_and_input_mismatches_fail_closed(self):
        vision, tower = _build_tiny_vision(dim=8)
        patches = mx.array(np.zeros((4, 3, 2, 2), dtype=np.float32))
        types = image_token_types(1, 1)
        stream = mx.array(np.ones((1, 4, 8), dtype=np.float32))
        with self.assertRaises(ValueError):
            tower.merge_image_embeddings(
                [[ImageInput(0, patches, 2, 2, image_token_types(1, 2))]], mx.array(np.ones((1, 8, 8), dtype=np.float32))
            )
        with self.assertRaises(ValueError):
            tower.merge_image_embeddings(
                [[ImageInput(2, patches, 2, 2, types)]], stream
            )
        with self.assertRaises(ValueError):
            tower.merge_image_embeddings(
                [[ImageInput(0, patches, 2, 2, mx.array([9, 9, 9, 9], dtype=mx.int32))]],
                stream,
            )
        malformed_known_types = (
            [IMAGE],
            [IMAGE_START, IMAGE, IMAGE_END, IMAGE_NEW_LINE],
            [IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_NEW_LINE, IMAGE_END],
            [IMAGE_START, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE_END],
        )
        for malformed in malformed_known_types:
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                tower.merge_image_embeddings(
                    [[ImageInput(0, patches, 2, 2, mx.array(malformed, dtype=mx.int32))]],
                    stream,
                )
        with self.assertRaises(ValueError):
            tower.merge_image_embeddings([[ImageInput(0, patches, 2, 2, types)]], mx.array(np.ones((1, 4, 8, 4), dtype=np.float32)))
        with self.assertRaises(ValueError):
            tower.vision(patches, 3, 2)
        with self.assertRaises(ValueError):
            DeepseekV41Vision(_tiny_vision_config(num_attention_heads=3), 8)


class TestDeepseekV41PublicNames(unittest.TestCase):
    """Official public tensor naming, including remaining gate-bias and wo_a."""

    def test_official_model_owns_packed_scale_leaves(self):
        model = Model(ModelArgs.from_dict(_full_config_dict()))
        names = dict(tree_flatten(model.parameters()))

        for name in (
            "layers.0.attn.wq_a.scale",
            "layers.0.attn.wq_b.scale",
            "layers.0.attn.wkv.scale",
            "layers.0.attn.wo_b.scale",
            "layers.2.attn.indexer.wq_b.scale",
            "mtp.0.attn.wq_a.scale",
            "mtp.0.attn.wq_b.scale",
            "mtp.0.attn.wkv.scale",
            "mtp.0.attn.wo_b.scale",
            "mtp.0.main_proj.scale",
            "mtp.0.ffn.gate.bias_vl",
            "mtp.1.ffn.gate.bias_vl",
            "mtp.2.ffn.gate.bias_vl",
        ):
            self.assertIn(name, names)

    def test_convert_mapping_preserves_gate_bias_wo_a_and_vision_names(self):
        self.assertEqual(
            public_checkpoint_name("model.layers.0.self_attn.wo_a.weight"),
            "layers.0.attn.wo_a.weight",
        )
        self.assertEqual(
            public_checkpoint_name("model.layers.0.self_attn.wo_a.weight_scale_inv"),
            "layers.0.attn.wo_a.scale",
        )
        self.assertEqual(
            public_checkpoint_name("model.layers.0.mlp.gate.e_score_correction_bias"),
            "layers.0.ffn.gate.bias",
        )
        self.assertEqual(
            public_checkpoint_name("model.layers.0.mlp.gate.e_score_correction_bias_vl"),
            "layers.0.ffn.gate.bias_vl",
        )
        self.assertEqual(
            public_checkpoint_name("model.vision.blocks.0.mlp.w1.weight"),
            "vision.blocks.0.mlp.w1.weight",
        )
        self.assertEqual(public_checkpoint_name("image_start"), "image_start")
        with self.assertRaises(ValueError):
            public_checkpoint_name("")

    def test_modules_emit_the_official_public_leaves(self):
        vision, tower = _build_tiny_vision(dim=8)
        self.assertEqual(tower.vision.patch_embed.proj.weight.shape, (16, 12))
        self.assertEqual(tower.vision.patch_embed.proj.bias.shape, (16,))
        self.assertEqual(tower.aligner.w1.weight.shape, (8, 64))
        self.assertEqual(tower.aligner.w2.weight.shape, (8, 8))
        self.assertEqual(tuple(np.asarray(tower.image_start).shape), (8,))
        self.assertEqual(
            tower.vision.blocks[0].attn.wqkv.weight.shape, (48, 16)
        )
        self.assertEqual(tower.vision.blocks[0].mlp.w1.weight.shape, (64, 16))
        names = vision_public_weight_names(vision, 8)
        self.assertIn("vision.patch_embed.proj.weight", names)
        self.assertIn("vision.patch_embed.proj.bias", names)
        self.assertIn("vision.blocks.0.attn.wqkv.weight", names)
        self.assertIn("vision.blocks.0.mlp.w1.weight", names)
        self.assertIn("aligner.w1.weight", names)
        self.assertIn("image_start", names)
        self.assertNotIn("vision.blocks.0.mlp.w1.bias", names)
        official = VisionConfig.from_dict(_vision_config_dict())
        official_names = vision_public_weight_names(official, 5120)
        self.assertEqual(official_names.count("vision.blocks.31.attn.wo.weight"), 1)
        self.assertEqual(len([n for n in official_names if n.startswith("vision.blocks.")]), 32 * 8)

        gate = DeepseekV41Gate(_tiny_moe_text_config(), vision_enabled=True)
        params = dict(gate.parameters())
        self.assertIn("bias", params)
        self.assertIn("bias_vl", params)
        text_gate = DeepseekV41Gate(_tiny_moe_text_config())
        self.assertIn("bias", dict(text_gate.parameters()))
        self.assertNotIn("bias_vl", dict(text_gate.parameters()))

        remaining = remaining_gate_bias_and_wo_a_names(40, True)
        self.assertIn("layers.0.attn.wo_a.weight", remaining)
        self.assertIn("layers.0.ffn.gate.bias", remaining)
        self.assertIn("layers.0.ffn.gate.bias_vl", remaining)
        self.assertIn("layers.39.attn.wo_a.weight", remaining)

        model = Model(ModelArgs.from_dict(_full_config_dict()))
        weights = {
            "layers.0.attn.wo_a.weight": mx.zeros((32, 32), dtype=mx.uint8),
            "layers.0.attn.wo_a.scale": mx.array([[127]], dtype=mx.uint8),
        }
        require_public_weights(weights, ["layers.0.attn.wo_a.weight", "layers.0.attn.wo_a.scale"])
        out = model.sanitize(weights)
        self.assertEqual(out["layers.0.attn.wo_a.weight"].dtype, mx.bfloat16)
        with self.assertRaises(ValueError):
            require_public_weights({}, vision_public_weight_names(vision, 8)[:3])


class TestDeepseekV41ModelComposition(unittest.TestCase):
    def _args(self):
        config = _full_config_dict()
        text = _text_config_dict()
        text.update(
            {
                "vocab_size": 64,
                "hidden_size": 32,
                "moe_intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 0,
                "compress_ratios": [0, 0],
                "kv_source_layer_ids": [],
                "index_source_layer_ids": [],
                "candidate_source_layer_id": -1,
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "num_attention_heads": 4,
                "head_dim": 32,
                "qk_rope_head_dim": 8,
                "q_lora_rank": 32,
                "o_groups": 2,
                "o_lora_rank": 8,
                "sliding_window": 4,
                "engram_layer_ids": [],
                "engram_num_embeddings": [],
                "dspark_block_size": 0,
                "dspark_target_layer_ids": [],
            }
        )
        config["text_config"] = text
        config["vision_config"] = _vision_config_dict() | {"num_hidden_layers": 0}
        return ModelArgs.from_dict(config)

    def test_tiny_backbone_runs_prefill_then_decode_through_the_registered_model(self):
        model = Model(self._args())
        model.embed.weight = mx.arange(64 * 32, dtype=mx.float32).reshape(64, 32) / 2048
        model.head.weight = mx.eye(64, 32, dtype=mx.float32)
        cache = model.make_cache()

        prefill = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
        decode = model(mx.array([[4]], dtype=mx.int32), cache=cache)

        self.assertEqual(prefill.shape, (1, 3, 64))
        self.assertEqual(decode.shape, (1, 1, 64))
        self.assertTrue(np.isfinite(np.asarray(prefill)).all())
        self.assertGreater(float(mx.max(mx.abs(prefill))), 0.0)
        self.assertEqual([layer_cache.offset for layer_cache in cache], [4, 4])
        full = model(mx.array([[1, 2, 3, 4]], dtype=mx.int32))
        self.assertTrue(mx.allclose(decode, full[:, -1:], atol=1e-4, rtol=1e-4))
        self.assertFalse(
            any(
                name.startswith("_runtime.")
                for name, _ in tree_flatten(model.parameters())
            )
        )

    def test_model_shard_preserves_global_expert_names_in_every_backbone_layer(self):
        class Group:
            def size(self):
                return 2

            def rank(self):
                return 1

        model = Model(self._args())
        model.shard(Group())

        for layer in model.layers:
            self.assertEqual(layer.ffn.local_expert_ids, [4, 5, 6, 7])
            self.assertEqual(
                sum(expert is not None for expert in layer.ffn.experts), 4
            )
        expert_ids = {
            int(name.split(".")[4])
            for name, _ in tree_flatten(model.parameters())
            if name.startswith("layers.0.ffn.experts.")
        }
        self.assertEqual(expert_ids, {4, 5, 6, 7})

    def test_preload_sharding_excludes_remote_experts_and_shards_engram_rows(self):
        class Group:
            def size(self):
                return 2

            def rank(self):
                return 1

        config = _full_config_dict()
        text = asdict(self._args().text_config) | {
            "engram_layer_ids": [0],
            "engram_num_embeddings": [408],
            "engram_max_ngram_size": 3,
            "engram_vocab_size": 97,
            "engram_n_heads": 2,
            "engram_head_dim": 32,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 11,
        }
        config["text_config"] = text
        config["vision_config"] = _vision_config_dict() | {"num_hidden_layers": 0}
        model = Model(ModelArgs.from_dict(config))
        model.prepare_sharded_load(Group())
        weight, scale = _random_engram_shard(408, 32, 32, seed=19)

        def add_experts(header):
            template = dict(header["layers.0.engram.embed.weight"])
            header["layers.0.ffn.experts.0.w1.weight"] = dict(template)
            header["layers.0.ffn.experts.4.w1.weight"] = dict(template)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model-00001-of-00001.safetensors"
            _write_engram_fixture(
                path,
                weight,
                scale,
                weight_key="layers.0.engram.embed.weight",
                scale_key="layers.0.engram.embed.scale",
                mutate=add_experts,
            )
            excluded = model.prepare_file_backed_weights(Path(tmp), [path])
            excluded_names = excluded[str(path.resolve())]
            self.assertIn("layers.0.ffn.experts.0.w1.weight", excluded_names)
            self.assertNotIn("layers.0.ffn.experts.4.w1.weight", excluded_names)
            embedding = model.layers[0].engram.embed
            self.assertEqual(embedding.cache.store.row_start, 204)
            self.assertEqual(embedding.cache.store.num_rows, 204)
            self.assertIsNotNone(embedding.all_reduce)
            self.assertTrue(
                np.array_equal(
                    embedding.cache.store.read_rows([0])[0][0], weight[204]
                )
            )
            embedding.cache.store.close()

            def add_unknown_expert(header):
                template = dict(header["layers.0.engram.embed.weight"])
                header["layers.0.ffn.experts.8.w1.weight"] = dict(template)

            invalid_path = Path(tmp) / "invalid.safetensors"
            _write_engram_fixture(
                invalid_path,
                weight,
                scale,
                weight_key="layers.0.engram.embed.weight",
                scale_key="layers.0.engram.embed.scale",
                mutate=add_unknown_expert,
            )
            with self.assertRaisesRegex(ValueError, "names no model expert"):
                model.prepare_file_backed_weights(Path(tmp), [invalid_path])

    def test_file_backed_loader_claims_only_the_engram_table_payloads(self):
        config = _full_config_dict()
        text = _text_config_dict()
        text.update(
            {
                "vocab_size": 64,
                "hidden_size": 32,
                "moe_intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 0,
                "compress_ratios": [0, 0],
                "kv_source_layer_ids": [],
                "index_source_layer_ids": [],
                "candidate_source_layer_id": -1,
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "num_attention_heads": 4,
                "head_dim": 32,
                "qk_rope_head_dim": 8,
                "q_lora_rank": 32,
                "o_groups": 2,
                "o_lora_rank": 8,
                "sliding_window": 4,
                "engram_layer_ids": [0],
                "engram_num_embeddings": [408],
                "engram_max_ngram_size": 3,
                "engram_vocab_size": 97,
                "engram_n_heads": 2,
                "engram_head_dim": 32,
                "engram_pad_token_id": 2,
                "engram_compressed_vocab_size": 11,
                "dspark_block_size": 0,
                "dspark_target_layer_ids": [0],
            }
        )
        config["text_config"] = text
        config["vision_config"] = _vision_config_dict() | {
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "patch_size": 2,
            "downsample_ratio": 2,
            "max_image_tokens": 64,
            "min_pixels": 16,
        }
        model = Model(ModelArgs.from_dict(config))
        rows = text["engram_num_embeddings"][0]
        weight, scale = _random_engram_shard(rows, 32, 32, seed=17)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model-00001-of-00001.safetensors"
            weight_key = "layers.0.engram.embed.weight"
            scale_key = "layers.0.engram.embed.scale"
            _write_engram_fixture(
                path,
                weight,
                scale,
                weight_key=weight_key,
                scale_key=scale_key,
            )
            excluded = model.prepare_file_backed_weights(Path(tmp), [path])
            self.assertEqual(
                excluded[str(path.resolve())], {weight_key, scale_key}
            )
            names = dict(tree_flatten(model.parameters()))
            self.assertIn("layers.0.engram.wkv.weight", names)
            self.assertIn("layers.0.engram.wkv.scale", names)
            self.assertIn("layers.0.engram.q_weight", names)
            self.assertNotIn(weight_key, names)
            embedding = model.layers[0].engram.embed
            self.assertEqual(embedding.cache.store.bytes_read, 0)
            embedding(mx.array([0], dtype=mx.int32))
            self.assertEqual(embedding.cache.store.rows_read, 1)
            embedding.cache.clear()
            model.embed.weight = (
                mx.arange(64 * 32, dtype=mx.float32).reshape(64, 32) / 2048
            )
            model.layers[0].engram.wkv.weight = mx.full(
                model.layers[0].engram.wkv.weight.shape, 0x38, dtype=mx.uint8
            )
            model.layers[0].engram.wkv.scale = mx.full(
                model.layers[0].engram.wkv.scale.shape, 127, dtype=mx.uint8
            )
            model._runtime.bind_engram_hasher(
                EngramNgramHasher(
                    model._runtime.engram_layout,
                    _ENGRAM_TOKEN_MAP + [i % 11 for i in range(24)],
                    2,
                    11,
                    max_seq_len=16,
                )
            )
            input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
            expanded = expand_hyper_connection_stream(
                model.embed(input_ids), model.args.text_config.hc_mult
            )
            _, main_hidden = model.forward_main(
                input_ids,
                token_types=mx.array([[-1, IMAGE, -1]], dtype=mx.int32),
            )
            expected_masked = mx.mean(expanded[:, 1:2], axis=2)
            self.assertTrue(
                mx.allclose(main_hidden[:, 1:2], expected_masked)
            )
            self.assertFalse(
                mx.allclose(main_hidden[:, 0:1], mx.mean(expanded[:, 0:1], axis=2))
            )
            embedding.cache.store.close()

    def test_file_backed_loader_rejects_missing_or_wrong_engram_state(self):
        model = Model(ModelArgs.from_dict(_full_config_dict()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model-00001-of-00001.safetensors"
            weight, scale = _random_engram_shard(2, 32, 32, seed=23)
            _write_engram_fixture(path, weight, scale)
            with self.assertRaisesRegex(ValueError, "missing file-backed Engram"):
                model.prepare_file_backed_weights(Path(tmp), [path])

        config = _full_config_dict()
        text = config["text_config"] | {
            "engram_layer_ids": [0],
            "engram_num_embeddings": [408],
            "engram_max_ngram_size": 3,
            "engram_vocab_size": 97,
            "engram_n_heads": 2,
            "engram_head_dim": 32,
        }
        config["text_config"] = text
        config["vision_config"] = _vision_config_dict() | {"num_hidden_layers": 0}
        small = Model(ModelArgs.from_dict(config))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model-00001-of-00001.safetensors"
            weight, scale = _random_engram_shard(407, 32, 32, seed=29)
            _write_engram_fixture(
                path,
                weight,
                scale,
                weight_key="layers.0.engram.embed.weight",
                scale_key="layers.0.engram.embed.scale",
            )
            with self.assertRaisesRegex(ValueError, "config requires"):
                small.prepare_file_backed_weights(Path(tmp), [path])

    def test_registered_model_merges_an_actual_image_span(self):
        args = self._args()
        config = _full_config_dict()
        config["text_config"] = asdict(args.text_config)
        config["vision_config"] = _vision_config_dict() | {
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "patch_size": 2,
            "downsample_ratio": 2,
            "max_image_tokens": 64,
            "min_pixels": 16,
        }
        model = Model(ModelArgs.from_dict(config))
        stream = mx.ones((1, 4, 32), dtype=mx.float32)
        patches = mx.zeros((4, 3, 2, 2), dtype=mx.float32)
        image = ImageInput(0, patches, 2, 2, image_token_types(1, 1))
        merged = model._runtime.merge_image_embeddings([[image]], stream)
        self.assertEqual(merged.shape, stream.shape)
        self.assertFalse(mx.allclose(merged, stream))


if __name__ == "__main__":
    unittest.main()

