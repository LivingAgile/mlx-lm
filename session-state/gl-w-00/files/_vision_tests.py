
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
                [[ImageInput(0, patches, 2, 2, types)]], stream
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
        with self.assertRaises(ValueError):
            tower.merge_image_embeddings([[ImageInput(0, patches, 2, 2, types)]], mx.ones((1, 4, 8, 4)))
        with self.assertRaises(ValueError):
            tower.vision(patches, 3, 2)
        with self.assertRaises(ValueError):
            DeepseekV41Vision(_tiny_vision_config(num_attention_heads=3), 8)


class TestDeepseekV41PublicNames(unittest.TestCase):
    """Official public tensor naming, including remaining gate-bias and wo_a."""

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
        with self.assertRaises(NotImplementedError):
            model(mx.array([[1, 2, 3]]))
