# Copyright (c) 2026 the mlx-lm contributors
"""Plan 0051 M2 slice: config fail-closed behavior, packed FP8(E4M3)/E8M0 and
FP4(E2M1)/E8M0 numerical primitives, and the wo_a exception for deepseek_v41.

The FP8 and FP4 fixtures below are the *real* sampled tensor bytes and
independently-decoded expected values recorded in Plan 0051 ``artifacts/
m1-layout-probes.md`` (Probe 1: ``layers.0.attn.wkv.weight``/``.scale`` from
``model-00003-of-00048.safetensors`` at the pinned revision
``deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277``;
Probe 2: ``layers.0.ffn.experts.0.w1.weight``/``.scale`` from the same shard).
The FP4 nibble order and magnitude table were independently confirmed
byte-for-byte against the official ``inference/convert.py`` reference
converter at that pinned revision (see ``artifacts/m1-gap-closure.md``).
"""
import unittest

import mlx.core as mx

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.deepseek_v41 import (
    FP4_E2M1_TABLE,
    ModelArgs,
    Model,
    QuantizationConfig,
    TextConfig,
    VisionConfig,
    decode_e8m0_scale,
    dequantize_fp4_block,
    dequantize_fp8_block,
    dequantize_wo_a,
    unpack_fp4_e2m1,
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


if __name__ == "__main__":
    unittest.main()

