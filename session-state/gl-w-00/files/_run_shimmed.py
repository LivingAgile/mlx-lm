"""Best-effort: run the Engram test classes against a numpy shim for mlx."""
import os, runpy, sys, types
sys.path.insert(0, os.getcwd())
_resource = types.ModuleType("resource")
_resource.RLIMIT_NOFILE = 0
_resource.getrlimit = lambda *a: (4096, 8192)
_resource.setrlimit = lambda *a: None
_resource.__getattr__ = lambda name: (lambda *a, **k: None)
sys.modules["resource"] = _resource
import numpy as np

g = runpy.run_path("session-state/gl-w-00/files/_engram_harness.py")
shim_mx = g["mx"]
shim_nn = g["nn"]

core = types.ModuleType("mlx.core")
for name in dir(shim_mx):
    if not name.startswith("__"):
        setattr(core, name, getattr(shim_mx, name))
nn_mod = types.ModuleType("mlx.nn")
for name in dir(shim_nn):
    if not name.startswith("__"):
        setattr(nn_mod, name, getattr(shim_nn, name))
utils = types.ModuleType("mlx.utils")


def tree_flatten(tree, prefix="", is_leaf=None):
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
utils.tree_unflatten = lambda pairs: dict(pairs)


def _permissive(module):
    def __getattr__(name):
        def _stub(*args, **kwargs):
            raise NotImplementedError("shim stub: " + module.__name__ + "." + name)

        _stub.__name__ = name
        return _stub

    module.__getattr__ = __getattr__
    return module


# mx.array is a real type in MLX; make the shim one behave like a type so
# isinstance checks in the tests exercise the same thing they will on Metal.
_orig_array = core.array


class _ArrayMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, np.ndarray)

    def __call__(cls, *args, **kwargs):
        return _orig_array(*args, **kwargs)


class _Array(metaclass=_ArrayMeta):
    pass


core.array = _Array


def _parameters(self):
    out = {}
    for key, value in vars(self).items():
        if isinstance(value, np.ndarray):
            out[key] = value
        elif isinstance(value, nn_mod.Module):
            child = value.parameters()
            if child:
                out[key] = child
    return out


nn_mod.Module.parameters = _parameters

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

# Register stub parent packages so mlx_lm/__init__.py (transformers, tokenizers)
# never runs: only the model module under test is loaded.
_pkg = types.ModuleType("mlx_lm")
_pkg.__path__ = [os.path.join(os.getcwd(), "mlx_lm")]
_models = types.ModuleType("mlx_lm.models")
_models.__path__ = [os.path.join(os.getcwd(), "mlx_lm", "models")]
sys.modules["mlx_lm"] = _pkg
sys.modules["mlx_lm.models"] = _models

try:
    import importlib

    prod = importlib.import_module("mlx_lm.models.deepseek_v41")
    print("production module imported under shim")
except Exception as exc:
    print("SHIM IMPORT FAILED:", type(exc).__name__, exc)
    raise SystemExit(0)

import unittest

spec = __import__("importlib").util.spec_from_file_location(
    "engram_tests", "tests/test_deepseek_v41.py"
)
module = __import__("importlib").util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except Exception as exc:
    print("TEST MODULE IMPORT FAILED:", type(exc).__name__, exc)
    raise SystemExit(0)

loader = unittest.TestLoader()
suite = unittest.TestSuite()
for name in dir(module):
    if name.startswith("TestDeepseekV41Engram"):
        suite.addTests(loader.loadTestsFromTestCase(getattr(module, name)))
print("collected", suite.countTestCases(), "engram tests")
unittest.TextTestRunner(verbosity=1).run(suite)
