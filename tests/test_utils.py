# Copyright © 2024 Apple Inc.

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm import convert, utils

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        if not os.path.isdir(cls.test_dir):
            os.mkdir(cls.test_dir_fid.name)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_load(self):
        from mlx_lm.models.qwen2 import Model as Qwen2Model

        model, _ = utils.load(HF_MODEL_PATH)
        self.assertIsInstance(model, Qwen2Model)

        model_lazy, _ = utils.load(HF_MODEL_PATH, lazy=True)

        mx.eval(model_lazy.parameters())

        p1 = model.layers[0].mlp.up_proj.weight
        p2 = model_lazy.layers[0].mlp.up_proj.weight
        self.assertTrue(mx.allclose(p1, p2))

    def test_make_shards(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=2048,
            num_hidden_layers=32,
            intermediate_size=4096,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=30_000,
        )
        model = llama.Model(args)
        weights = tree_flatten(model.parameters())
        gb = sum(p.nbytes for _, p in weights) // 2**30
        shards = utils.make_shards(dict(weights), 1)
        self.assertTrue(gb <= len(shards) <= gb + 1)

    def test_quantize(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = llama.Model(args)
        model, config = utils.quantize_model(model, {}, 64, 4)
        weights = dict(tree_flatten(model.parameters()))
        self.assertTrue("model.layers.2.mlp.up_proj.scales" in weights)
        self.assertTrue("model.layers.2.mlp.up_proj.biases" in weights)
        self.assertEqual(config["quantization"]["group_size"], 64)
        self.assertEqual(config["quantization"]["bits"], 4)

    def test_convert(self):
        mlx_path = os.path.join(self.test_dir, "mlx_model")

        convert(HF_MODEL_PATH, mlx_path=mlx_path, quantize=False)
        model, _ = utils.load(mlx_path)
        self.assertTrue(isinstance(model.layers[0].mlp.up_proj, nn.QuantizedLinear))
        self.assertTrue(isinstance(model.layers[-1].mlp.up_proj, nn.QuantizedLinear))

        # Check model weights have right type
        mlx_path = os.path.join(self.test_dir, "mlx_model_bf16")
        convert(HF_MODEL_PATH, mlx_path=mlx_path, dtype="bfloat16")
        model, _ = utils.load(mlx_path)

        self.assertEqual(model.layers[0].mlp.up_proj.scales.dtype, mx.bfloat16)
        self.assertEqual(model.layers[-1].mlp.up_proj.scales.dtype, mx.bfloat16)

    def test_load_model_with_custom_get_classes(self):
        class CustomQwenModel(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.config = args
                self.custom_attribute = "This is a custom model"

            def load_weights(self, weights, **kwargs):
                self.qwenWeights = weights

        class CustomQwenConfig:
            @classmethod
            def from_dict(cls, config):
                instance = cls()
                for k, v in config.items():
                    setattr(instance, k, v)
                return instance

        def custom_get_classes(config):
            return CustomQwenModel, CustomQwenConfig

        model_path = utils.hf_repo_to_path(HF_MODEL_PATH)
        model, _ = utils.load_model(model_path, get_model_classes=custom_get_classes)

        self.assertIsInstance(model, CustomQwenModel)
        self.assertTrue(hasattr(model, "custom_attribute"))
        self.assertEqual(model.custom_attribute, "This is a custom model")
        self.assertTrue(hasattr(model, "qwenWeights"))

    def test_selective_safetensor_load_never_reads_excluded_payload(self):
        path = Path(self.test_dir) / "selective.safetensors"
        keep = mx.array([3, 5, 7], dtype=mx.uint8)
        header = {
            "keep": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
            "excluded": {
                "dtype": "U8",
                "shape": [1_000_000_000],
                "data_offsets": [3, 1_000_000_003],
            },
        }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 8)
        with open(path, "wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            handle.write(bytes(memoryview(keep)))

        loaded = utils._load_safetensors_with_e8m0(str(path), {"excluded"})
        self.assertEqual(set(loaded), {"keep"})
        self.assertTrue(mx.array_equal(loaded["keep"], keep))

    def test_load_model_honors_file_backed_exclusion_before_mx_load(self):
        class _Args:
            @classmethod
            def from_dict(cls, config):
                return cls()

        class _Model(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.keep = mx.zeros((3,), dtype=mx.uint8)

            def prepare_file_backed_weights(self, model_path, weight_files):
                path = str(Path(weight_files[0]).resolve())
                return {path: {"excluded"}}

        path = Path(self.test_dir) / "model-00001-of-00001.safetensors"
        keep = mx.array([11, 13, 17], dtype=mx.uint8)
        header = {
            "keep": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
            "excluded": {
                "dtype": "U8",
                "shape": [1_000_000_000],
                "data_offsets": [3, 1_000_000_003],
            },
        }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 8)
        with open(path, "wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            handle.write(bytes(memoryview(keep)))
        with open(Path(self.test_dir) / "config.json", "w") as handle:
            json.dump({"model_type": "test"}, handle)

        model, _ = utils.load_model(
            Path(self.test_dir),
            get_model_classes=lambda config: (_Model, _Args),
        )
        self.assertTrue(mx.array_equal(model.keep, keep))

    def test_load_model_prepares_model_specific_sharding_before_file_exclusions(self):
        events = []

        class _Args:
            @classmethod
            def from_dict(cls, config):
                return cls()

        class _Model(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.keep = mx.zeros((3,), dtype=mx.uint8)

            def prepare_sharded_load(self, group):
                events.append(("shard", group))

            def prepare_file_backed_weights(self, model_path, weight_files):
                events.append(("files", events[-1][0]))
                return {}

        path = Path(self.test_dir) / "model-00001-of-00001.safetensors"
        mx.save_safetensors(str(path), {"keep": mx.array([2, 3, 5], dtype=mx.uint8)})
        with open(Path(self.test_dir) / "config.json", "w") as handle:
            json.dump({"model_type": "test"}, handle)
        group = object()

        model, _ = utils.load_model(
            Path(self.test_dir),
            get_model_classes=lambda config: (_Model, _Args),
            shard_group=group,
        )

        self.assertEqual(events, [("shard", group), ("files", "shard")])
        self.assertTrue(mx.array_equal(model.keep, mx.array([2, 3, 5], dtype=mx.uint8)))

    def test_load_model_gemma4_with_per_layer_projection_quantization(self):
        from mlx_lm.models import gemma4

        args = gemma4.ModelArgs.from_dict(
            {
                "model_type": "gemma4",
                "vocab_size": 32,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 16,
                    "global_head_dim": 16,
                    "sliding_window": 8,
                    "sliding_window_pattern": 1,
                    "layer_types": ["full_attention", "full_attention"],
                    "hidden_size_per_layer_input": 32,
                    "vocab_size_per_layer_input": 32,
                    "num_kv_shared_layers": 0,
                    "tie_word_embeddings": True,
                },
            }
        )
        model = gemma4.Model(args)
        model, config = utils.quantize_model(
            model,
            {
                "model_type": "gemma4",
                "vocab_size": args.vocab_size,
                "text_config": args.text_config,
            },
            group_size=32,
            bits=4,
        )

        config["quantization"]["language_model.model.per_layer_model_projection"] = {
            "group_size": 32,
            "bits": 4,
        }

        with tempfile.TemporaryDirectory(dir=self.test_dir) as mlx_path:
            utils.save_model(mlx_path, model)
            utils.save_config(config, os.path.join(mlx_path, "config.json"))

            loaded, loaded_config = utils.load_model(Path(mlx_path))

            self.assertIn(
                "language_model.model.per_layer_model_projection",
                loaded_config["quantization"],
            )

            logits = loaded(mx.array([[1, 2, 3]], dtype=mx.int32))
            mx.eval(logits)
            self.assertEqual(logits.shape, (1, 3, args.vocab_size))

    def test_load_model_honors_nested_mixed_bit_quantization_policy(self):
        from mlx_lm.models.deepseek_v41 import (
            DeepseekV41PackedLinear,
            DeepseekV41QuantizedLinear,
        )

        class _Args:
            @classmethod
            def from_dict(cls, config):
                return cls()

        class _Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = DeepseekV41PackedLinear(64, 32, "fp4")

        class _FFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = [_Expert()]

        class _Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq_a = DeepseekV41PackedLinear(64, 32, "fp8")

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.ffn = _FFN()
                self.attn = _Attention()

        class _Model(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.layers = [_Layer()]

        expert = mx.random.normal((32, 64))
        dense = mx.random.normal((32, 64))
        expert_weight, expert_scales, expert_biases = mx.quantize(
            expert, group_size=64, bits=4
        )
        dense_weight, dense_scales, dense_biases = mx.quantize(
            dense, group_size=64, bits=8
        )
        weights = {
            "layers.0.ffn.experts.0.w1.weight": expert_weight,
            "layers.0.ffn.experts.0.w1.scales": expert_scales,
            "layers.0.ffn.experts.0.w1.biases": expert_biases,
            "layers.0.attn.wq_a.weight": dense_weight,
            "layers.0.attn.wq_a.scales": dense_scales,
            "layers.0.attn.wq_a.biases": dense_biases,
        }
        config = {
            "model_type": "test",
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "expert_bits": 4,
                "modules": {
                    "layers.0.ffn.experts.gate_proj": {
                        "group_size": 64,
                        "bits": 4,
                    },
                    "layers.0.attn.wq_a": {"group_size": 64, "bits": 8},
                },
            },
        }

        with tempfile.TemporaryDirectory(dir=self.test_dir) as model_path:
            mx.save_safetensors(
                str(Path(model_path) / "model-00001-of-00001.safetensors"),
                weights,
            )
            with open(Path(model_path) / "config.json", "w") as handle:
                json.dump(config, handle)

            model, _ = utils.load_model(
                Path(model_path),
                get_model_classes=lambda config: (_Model, _Args),
            )

        routed = model.layers[0].ffn.experts[0].w1
        dense = model.layers[0].attn.wq_a
        self.assertIsInstance(routed, DeepseekV41QuantizedLinear)
        self.assertIsInstance(dense, DeepseekV41QuantizedLinear)
        self.assertEqual(routed.bits, 4)
        self.assertEqual(dense.bits, 8)


if __name__ == "__main__":
    unittest.main()
