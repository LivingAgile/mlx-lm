"""Throwaway numpy harness for the Engram slice (NOT production, NOT committed)."""
import ast, io, json, math, re, sys, types, unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

mx = types.SimpleNamespace()
for name in (
    "zeros zeros_like ones abs maximum minimum sum mean sqrt where power take "
    "isnan concatenate stack arange"
).split():
    setattr(mx, name, getattr(np, name))
mx.float32 = np.float32
mx.uint32 = np.uint32
mx.uint8 = np.uint8
mx.int32 = np.int32
mx.int64 = np.int64
mx.bfloat16 = np.float32
mx.array = lambda x, dtype=None: np.array(x, dtype=dtype)
mx.rsqrt = lambda x: 1.0 / np.sqrt(x)
mx.sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
mx.take = lambda a, idx, axis=None: np.take(a, idx, axis=axis)


def _from_fp8(codes, dtype=np.float32):
    c = np.asarray(codes).astype(np.uint8)
    sign = np.where((c >> 7) & 1, -1.0, 1.0)
    exp = ((c >> 3) & 0xF).astype(np.int32)
    man = (c & 0x7).astype(np.int32)
    sub = (man.astype(np.float64) / 8.0) * (2.0 ** -6)
    nor = (1.0 + man.astype(np.float64) / 8.0) * (2.0 ** (exp.astype(np.float64) - 7.0))
    val = np.where(exp == 0, sub, nor) * sign
    val = np.where((exp == 15) & (man == 7), np.nan, val)
    return val.astype(dtype)


mx.from_fp8 = _from_fp8


class _Module:
    def __init__(self):
        pass


class _Linear(_Module):
    def __init__(self, in_dims, out_dims, bias=False):
        rng = np.random.default_rng(7)
        scale = 1.0 / math.sqrt(in_dims)
        self.weight = rng.uniform(-scale, scale, (out_dims, in_dims)).astype(np.float32)

    def __call__(self, x):
        return (np.asarray(x).astype(np.float32) @ self.weight.T).astype(np.float32)


nn = types.SimpleNamespace(Module=_Module, Linear=_Linear)

SRC = io.open("mlx_lm/models/deepseek_v41.py", encoding="utf-8").read()
tree = ast.parse(SRC)
WANTED = {
    "decode_e8m0_scale",
    "ENGRAM_FP8_BLOCK_SIZE", "ENGRAM_GATE_CLAMP", "ENGRAM_RNG_LAYER_SEED_STRIDE",
    "ENGRAM_DEAD_TOKEN", "ENGRAM_NORMALIZER_SEQUENCE", "ENGRAM_SPACE_SENTINEL",
    "_UNICODE_REPLACEMENT_CHAR", "_ENGRAM_WHITESPACE_RUN", "_UNICODE_WHITESPACE",
    "_strip_unicode_whitespace", "normalize_engram_token_text",
    "engram_compressed_token_key", "build_engram_compressed_token_map",
    "build_engram_compressed_token_map_from_tokenizer",
    "_is_prime", "find_next_prime", "compute_engram_hash_multipliers",
    "EngramLayout", "validate_engram_config", "EngramNgramHasher",
    "dequantize_engram_rows", "EngramRowStore",
    "ENGRAM_WEIGHT_SAFETENSORS_DTYPES", "ENGRAM_SCALE_SAFETENSORS_DTYPES",
    "_SAFETENSORS_HEADER_LIMIT", "SafetensorsEngramRowStore",
    "EngramCacheStats", "BoundedEngramRowCache",
    "DeepseekV41EngramEmbedding", "engram_signed_sqrt_sigmoid_gate",
    "DeepseekV41Engram",
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

ns = {
    "mx": mx, "nn": nn, "np": np, "math": math, "json": json, "re": re,
    "unicodedata": unicodedata, "OrderedDict": OrderedDict, "dataclass": dataclass,
    "Any": Any, "Callable": Callable, "Dict": Dict, "Iterable": Iterable,
    "List": List, "Optional": Optional, "Sequence": Sequence, "Tuple": Tuple,
    "Union": Union, "TextConfig": object, "__name__": "engram_harness",
}
exec(compile(ast.Module(body=keep, type_ignores=[]), "deepseek_v41", "exec"), ns)
print("exec'd", len(keep), "top-level nodes")
globals().update(ns)
