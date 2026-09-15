"""Session-only NumPy shim runner for tests.test_deepseek_v41. Not production."""
import math
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.getcwd())

_resource = types.ModuleType("resource")
_resource.RLIMIT_NOFILE = 0
_resource.getrlimit = lambda *a: (4096, 8192)
_resource.setrlimit = lambda *a: None
_resource.__getattr__ = lambda name: (lambda *a, **k: None)
sys.modules["resource"] = _resource

core = types.ModuleType("mlx.core")
for name in (
    "zeros zeros_like ones abs maximum minimum min max sum mean sqrt where power take "
    "isnan isinf concatenate stack arange clip exp cos sin sign floor repeat pad "
    "broadcast_to expand_dims sort argpartition take_along_axis einsum allclose "
    "array_equal any all logaddexp full transpose"
).split():
    setattr(core, name, getattr(np, name))
core.float32 = np.float32
core.uint32 = np.uint32
core.uint8 = np.uint8
core.int32 = np.int32
core.int8 = np.int8
core.int64 = np.int64
core.bool_ = np.bool_
core.bfloat16 = np.float32
core.rsqrt = lambda x: 1.0 / np.sqrt(x)
core.sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x)))
core.bitwise_and = np.bitwise_and
core.right_shift = np.right_shift
core.sqrt = np.sqrt
core.log = np.log
core.isfinite = np.isfinite

def _as_array_flag(fn):
    def wrapped(*a, **k):
        return np.array(fn(*a, **k))
    return wrapped

core.allclose = _as_array_flag(np.allclose)
core.array_equal = _as_array_flag(np.array_equal)



def _softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


core.softmax = _softmax


def _put_along_axis(arr, idx, vals, axis):
    out = np.array(arr, copy=True)
    np.put_along_axis(out, idx, vals, axis=axis)
    return out


core.put_along_axis = _put_along_axis


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


core.from_fp8 = _from_fp8
core.load = lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("mx.load"))
core.eval = lambda *a, **k: None

_orig_array = lambda x, dtype=None: np.array(x, dtype=dtype)


class _ArrayMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, np.ndarray)

    def __call__(cls, *args, **kwargs):
        return _orig_array(*args, **kwargs)


class _Array(metaclass=_ArrayMeta):
    pass


core.array = _Array


class _Random:
    def seed(self, s):
        np.random.seed(int(s))

    def normal(self, shape=None, loc=0.0, scale=1.0, dtype=None, key=None):
        dt = np.float32 if dtype is None else dtype
        if key is None:
            return np.random.normal(loc, scale, size=shape).astype(dt)
        return np.random.default_rng(int(key)).normal(loc, scale, size=shape).astype(dt)

    def uniform(self, low=0.0, high=1.0, shape=None, dtype=None, key=None):
        dt = np.float32 if dtype is None else dtype
        if key is None:
            return np.random.uniform(low, high, size=shape).astype(dt)
        return np.random.default_rng(int(key)).uniform(low, high, size=shape).astype(dt)

    def randint(self, low, high=None, shape=None, dtype=None, key=None):
        dt = np.int32 if dtype is None else dtype
        if key is None:
            return np.random.randint(low, high, size=shape).astype(dt)
        return np.random.default_rng(int(key)).integers(low, high, size=shape, dtype=dt)


core.random = _Random()


def _permissive(module):
    def __getattr__(name):
        def _stub(*args, **kwargs):
            raise NotImplementedError("shim stub: " + module.__name__ + "." + name)
        _stub.__name__ = name
        return _stub
    module.__getattr__ = __getattr__
    return module


class _Module:
    def __init__(self):
        pass

    def parameters(self):
        out = {}
        for key, value in vars(self).items():
            if key.startswith("_"):
                continue
            if isinstance(value, np.ndarray):
                out[key] = value
            elif isinstance(value, _Module):
                child = value.parameters()
                if child:
                    out[key] = child
            elif isinstance(value, (list, tuple)):
                child = []
                found = False
                for item in value:
                    if isinstance(item, _Module):
                        found = True
                        child.append(item.parameters())
                    else:
                        child.append({})
                if found:
                    out[key] = child
        return out

    def keys(self):
        return self.parameters().keys()

    def __getitem__(self, key):
        return self.parameters()[key]

    def __iter__(self):
        return iter(self.parameters())


class _Linear(_Module):
    def __init__(self, in_dims, out_dims, bias=True):
        super().__init__()
        scale = 1.0 / math.sqrt(max(in_dims, 1))
        rng = np.random.default_rng(7)
        self.weight = rng.uniform(-scale, scale, (out_dims, in_dims)).astype(np.float32)
        if bias:
            self.bias = np.zeros((out_dims,), dtype=np.float32)

    def __call__(self, x):
        y = np.asarray(x, dtype=np.float32) @ self.weight.T
        if hasattr(self, "bias"):
            y = y + self.bias
        return y


class _RMSNorm(_Module):
    def __init__(self, dims, eps=1e-5):
        super().__init__()
        self.weight = np.ones((dims,), dtype=np.float32)
        self.eps = eps

    def __call__(self, x):
        xf = np.asarray(x, dtype=np.float32)
        rms = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (self.weight * (xf / rms)).astype(xf.dtype)


def _silu(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-x))


nn_mod = types.ModuleType("mlx.nn")
nn_mod.Module = _Module
nn_mod.Linear = _Linear
nn_mod.RMSNorm = _RMSNorm
nn_mod.silu = _silu

utils = types.ModuleType("mlx.utils")


def tree_flatten(tree, prefix=""):
    if isinstance(tree, dict):
        out = []
        for key, value in tree.items():
            out.extend(tree_flatten(value, prefix + "." + str(key)))
        return out
    if isinstance(tree, (list, tuple)):
        out = []
        for index, value in enumerate(tree):
            out.extend(tree_flatten(value, prefix + "." + str(index)))
        return out
    return [(prefix[1:], tree)]


utils.tree_flatten = tree_flatten
utils.tree_map = lambda f, tree: f(tree)
utils.tree_unflatten = lambda pairs: dict(pairs)

_permissive(core)
_permissive(nn_mod)
_permissive(utils)
mlx = types.ModuleType("mlx")
mlx.core = core
mlx.nn = nn_mod
mlx.utils = utils
sys.modules["mlx"] = mlx
sys.modules["mlx.core"] = core
sys.modules["mlx.nn"] = nn_mod
sys.modules["mlx.utils"] = utils

_pkg = types.ModuleType("mlx_lm")
_pkg.__path__ = [os.path.join(os.getcwd(), "mlx_lm")]
_models = types.ModuleType("mlx_lm.models")
_models.__path__ = [os.path.join(os.getcwd(), "mlx_lm", "models")]
sys.modules["mlx_lm"] = _pkg
sys.modules["mlx_lm.models"] = _models

v4 = types.ModuleType("mlx_lm.models.deepseek_v4")

class _V4Model:
    pass

class _V4Args:
    pass

v4.Model = _V4Model
v4.ModelArgs = _V4Args
sys.modules["mlx_lm.models.deepseek_v4"] = v4

import importlib
import importlib.util
prod = importlib.import_module("mlx_lm.models.deepseek_v41")
print("production module imported under shim")

loader = unittest.TestLoader()
spec = importlib.util.spec_from_file_location("test_deepseek_v41", "tests/test_deepseek_v41.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
suite = unittest.TestSuite()
for name in dir(module):
    obj = getattr(module, name)
    if isinstance(obj, type) and name.startswith("TestDeepseekV41"):
        suite.addTests(loader.loadTestsFromTestCase(obj))
print("collected", suite.countTestCases(), "tests")
result = unittest.TextTestRunner(verbosity=1).run(suite)
result_s = "RAN %s FAIL %s ERR %s SKIP %s" % (result.testsRun, len(result.failures), len(result.errors), len(result.skipped))
print(result_s)
from pathlib import Path as _Path
errp = _Path(r"session-state/gl-w-00/files/_suite_errors.txt")
chunks=[]
for item in result.failures + result.errors:
    chunks.append(str(item[0]))
    chunks.append(chr(10).join(item[1].splitlines()[-8:]))
    chunks.append("----")
errp.write_text(chr(10).join(chunks), encoding="utf-8")
print("wrote errors", errp.stat().st_size)
sys.exit(0 if result.wasSuccessful() else 1)
