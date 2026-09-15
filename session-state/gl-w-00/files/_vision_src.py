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
_CONVERT_BARE_MARKERS = ("hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_")


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
