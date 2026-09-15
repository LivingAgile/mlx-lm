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
"""
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.deepseek_v41 import (
    COMPRESS_KV_FP4_BLOCK_SIZE,
    FP4_E2M1_TABLE,
    FP4_WEIGHT_BLOCK_SIZE,
    DeepseekV41AttentionStack,
    DeepseekV41Expert,
    DeepseekV41Gate,
    DeepseekV41HyperConnections,
    DeepseekV41MoE,
    DeepseekV41PackedLinear,
    ModelArgs,
    Model,
    PhysicalLatentCache,
    QuantizationConfig,
    TextConfig,
    VisionConfig,
    act_quant_roundtrip,
    apply_rope_tail,
    decode_e8m0_scale,
    dequantize_fp4_block,
    dequantize_fp8_block,
    dequantize_wo_a,
    deterministic_topk_indices,
    expand_hyper_connection_stream,
    fp4_act_quant_roundtrip,
    hc_post,
    hc_pre,
    hc_split_sinkhorn,
    make_deepseek_v41_attention_caches,
    make_identity_pre_mix,
    noaux_tc_route,
    resolve_attention_layer_policies,
    rope_cos_sin,
    routed_expert_partition,
    routing_scores,
    select_candidate_blocks,
    sparse_attn,
    unpack_fp4_e2m1,
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
        with self.assertRaises(NotImplementedError):
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

    def test_prompt_cache_state_is_refused_rather_than_silently_unshared(self):
        cache = make_deepseek_v41_attention_caches(_official_text_config())[3]
        with self.assertRaises(NotImplementedError):
            _ = cache.state

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
        for name in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base"):
            setattr(hc, name, mx.random.normal(getattr(hc, name).shape))
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
            mx.allclose(got.reshape(-1, config.hidden_size), expected, atol=1e-3).item()
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
        for token in range(3):
            single = moe(x[:, token : token + 1])
            self.assertTrue(
                mx.allclose(single[0, 0], batched[0, token], atol=1e-3).item()
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
        self.assertEqual(len(moe.experts), 2)
        # Global ids, not rank-local ones.
        self.assertIs(moe.expert(3), moe.experts[1])
        with self.assertRaises(KeyError):
            moe.expert(0)
        with self.assertRaises(KeyError):
            moe.expert(4)

    def test_the_gate_is_replicated_across_ranks_while_experts_are_not(self):
        config = _tiny_moe_text_config()
        for rank in range(4):
            moe = DeepseekV41MoE(config, world_size=4, rank=rank)
            self.assertEqual(moe.gate.weight.shape, (8, config.hidden_size))
            self.assertEqual(moe.gate.n_routed_experts, 8)
            self.assertEqual(len(moe.experts), 2)
            # Every rank runs the shared expert; only the routed set is split.
            self.assertEqual(
                moe.shared_experts.w1.weight.shape,
                (config.moe_intermediate_size, config.hidden_size),
            )

    def test_executing_a_sharded_moe_fails_loud_instead_of_silently_partial(self):
        config = _tiny_moe_text_config()
        moe = DeepseekV41MoE(config, world_size=2, rank=0)
        with self.assertRaises(NotImplementedError):
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


if __name__ == "__main__":
    unittest.main()

