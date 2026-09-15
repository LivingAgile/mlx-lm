"""Throwaway numpy-backed harness (NOT production, NOT committed).

MLX cannot import on this machine, so this execs the pure-numeric subset of
mlx_lm/models/deepseek_v41.py against a numpy shim to catch logic bugs before
the Mac run. Anything needing mlx.nn is out of reach here.
"""
import ast, io, math, sys, types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

mx = types.SimpleNamespace()
for name in (
    "arange zeros zeros_like ones full abs minimum maximum max sum where sign floor "
    "clip exp cos sin sort argpartition take_along_axis take repeat pad concatenate "
    "stack einsum broadcast_to bitwise_and right_shift array_equal allclose isnan "
    "isinf any all power expand_dims"
).split():
    setattr(mx, name, getattr(np, name))
mx.float32 = np.float32
mx.uint32 = np.uint32
mx.int32 = np.int32
mx.int8 = np.int8
mx.bool_ = np.bool_
mx.bfloat16 = np.float32
mx.array = lambda x, dtype=None: np.array(x, dtype=dtype)


def _put_along_axis(arr, idx, vals, axis):
    out = np.array(arr, copy=True)
    np.put_along_axis(out, idx, vals, axis=axis)
    return out


mx.put_along_axis = _put_along_axis


def _softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


mx.softmax = _softmax

SRC = io.open("mlx_lm/models/deepseek_v41.py", encoding="utf-8").read()
tree = ast.parse(SRC)
WANTED = {
    "FP8_ACT_BLOCK_SIZE", "FP4_ACT_BLOCK_SIZE", "COMPRESS_KV_FP4_BLOCK_SIZE",
    "_FP8_E4M3_MAX", "_FP4_E2M1_MAX", "_SPARSE_ATTN_NEG_BOUND",
    "SPARSE_ATTN_QUERY_CHUNK", "COMPRESS_CACHE_GROWTH_SLOTS",
    "_binade_exponent", "_round_scale_pow2", "_round_half_even",
    "round_to_fp8_e4m3", "round_to_fp4_e2m1", "_blocked",
    "act_quant_roundtrip", "fp4_act_quant_roundtrip", "yarn_rope_frequencies",
    "rope_cos_sin", "apply_rope_tail", "window_topk_idxs", "sparse_attn",
    "select_candidate_blocks", "AttentionLayerPolicy", "_validate_id_list",
    "resolve_attention_layer_policies", "PhysicalLatentCache",
    "CompressorPoolState", "_latent_positions",
    "DeepseekV41SharedAttentionRuntime", "DeepseekV41AttentionCache",
    "make_deepseek_v41_attention_caches", "DeepseekV41Compressor",
    "DeepseekV41Indexer", "DeepseekV41Attention", "DeepseekV41AttentionStack",
}
keep = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANTED:
        keep.append(node)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in WANTED:
                keep.append(node)
                break
_RNG = np.random.default_rng(1234)


class _Module:
    def __init__(self):
        pass


class _Linear(_Module):
    def __init__(self, in_dims, out_dims, bias=False):
        scale = 1.0 / math.sqrt(in_dims)
        self.weight = (
            _RNG.uniform(-scale, scale, (out_dims, in_dims))
        ).astype(np.float32)

    def __call__(self, x):
        return (x.astype(np.float32) @ self.weight.T).astype(np.float32)


class _RMSNorm(_Module):
    def __init__(self, dims, eps=1e-5):
        self.weight = np.ones((dims,), dtype=np.float32)
        self.eps = eps

    def __call__(self, x):
        xf = x.astype(np.float32)
        rms = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (self.weight * (xf / rms)).astype(x.dtype)


nn = types.SimpleNamespace(Module=_Module, Linear=_Linear, RMSNorm=_RMSNorm)

ns = {
    "mx": mx, "nn": nn, "_BaseCache": object, "math": math, "dataclass": dataclass, "Optional": Optional,
    "List": List, "Dict": Dict, "Tuple": Tuple, "Any": Any, "Union": Union,
    "TextConfig": object, "__name__": "harness",
}
exec(compile(ast.Module(body=keep, type_ignores=[]), "deepseek_v41", "exec"), ns)
print("exec'd", len(keep), "top-level nodes")
globals().update(ns)
