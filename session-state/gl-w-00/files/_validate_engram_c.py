"""Numpy validation part C: dequant, file-backed store, bounded cache, embedding, gate."""
import json, os, runpy, tempfile
import numpy as np

g = runpy.run_path("session-state/gl-w-00/files/_engram_harness.py")
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


dequant_rows = g["dequantize_engram_rows"]
Store = g["SafetensorsEngramRowStore"]
Cache = g["BoundedEngramRowCache"]
Embed = g["DeepseekV41EngramEmbedding"]
gate_fn = g["engram_signed_sqrt_sigmoid_gate"]
from_fp8 = g["mx"].from_fp8


def write_fixture(path, weight, scale, weight_dtype="F8_E4M3", scale_dtype="F8_E8M0", corrupt=None):
    w = np.ascontiguousarray(weight, dtype=np.uint8)
    s = np.ascontiguousarray(scale, dtype=np.uint8)
    header = {
        "weight": {"dtype": weight_dtype, "shape": list(w.shape), "data_offsets": [0, w.nbytes]},
        "scale": {"dtype": scale_dtype, "shape": list(s.shape), "data_offsets": [w.nbytes, w.nbytes + s.nbytes]},
    }
    if corrupt:
        corrupt(header)
    blob = json.dumps(header).encode("utf-8")
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with open(path, "wb") as fh:
        fh.write(len(blob).to_bytes(8, "little"))
        fh.write(blob)
        fh.write(w.tobytes())
        fh.write(s.tobytes())
    return path


BLOCK = 4
DIM = 8
ROWS = 12
rng = np.random.default_rng(11)
weight = rng.integers(0, 200, size=(ROWS, DIM), dtype=np.uint8)
scale = rng.integers(120, 132, size=(ROWS, DIM // BLOCK), dtype=np.uint8)
tmpdir = tempfile.mkdtemp()
path = write_fixture(os.path.join(tmpdir, "engram.safetensors"), weight, scale)

print("---- row dequant ----")
rows = dequant_rows(weight[[1, 5]], scale[[1, 5]], BLOCK, np.float32)
expected = from_fp8(weight[[1, 5]], np.float32).reshape(2, DIM // BLOCK, BLOCK) * (
    2.0 ** (scale[[1, 5]].astype(np.float64) - 127.0)
)[:, :, None]
check("row dequant matches manual", np.allclose(rows, expected.reshape(2, DIM), rtol=0, atol=0, equal_nan=True))
check("empty gather is empty", dequant_rows(np.empty((0, DIM), np.uint8), np.empty((0, 2), np.uint8), BLOCK, np.float32).shape == (0, DIM))
for name, args in [
    ("scale grid mismatch", (weight, scale[:, :1], BLOCK)),
    ("row count mismatch", (weight, scale[:3], BLOCK)),
    ("indivisible width", (weight, scale, 3)),
]:
    try:
        dequant_rows(*args)
        check("rejects " + name, False)
    except ValueError:
        check("rejects " + name, True)

print()
print("---- file-backed store ----")
store = Store(path, block_size=BLOCK)
check("metadata parsed", (store.num_rows, store.dim, store.scale_dim) == (ROWS, DIM, DIM // BLOCK))
w, s = store.read_rows([7, 0, 7])
check("exact bytes by row", np.array_equal(w, weight[[7, 0, 7]]) and np.array_equal(s, scale[[7, 0, 7]]))
check("reads only what was asked", store.rows_read == 3 and store.bytes_read == 3 * (DIM + DIM // BLOCK))
try:
    store.read_rows([ROWS])
    check("rejects out-of-range row", False)
except IndexError:
    check("rejects out-of-range row", True)
store.close()

bad = write_fixture(os.path.join(tmpdir, "bad_dtype.safetensors"), weight, scale, weight_dtype="F32")
try:
    Store(bad, block_size=BLOCK)
    check("rejects wrong dtype", False)
except ValueError:
    check("rejects wrong dtype", True)
bad2 = write_fixture(os.path.join(tmpdir, "bad_scale.safetensors"), weight, scale[:, :1])
try:
    Store(bad2, block_size=BLOCK)
    check("rejects scale grid mismatch", False)
except ValueError:
    check("rejects scale grid mismatch", True)
bad3 = write_fixture(os.path.join(tmpdir, "bad_off.safetensors"), weight, scale, corrupt=lambda h: h["weight"].__setitem__("data_offsets", [0, 3]))
try:
    Store(bad3, block_size=BLOCK)
    check("rejects bad data_offsets", False)
except ValueError:
    check("rejects bad data_offsets", True)
bad4 = write_fixture(os.path.join(tmpdir, "bad_key.safetensors"), weight, scale, corrupt=lambda h: h.__setitem__("scale", h.pop("scale")) or h.__setitem__("other", h.pop("weight")))
try:
    Store(bad4, block_size=BLOCK)
    check("rejects missing weight key", False)
except ValueError:
    check("rejects missing weight key", True)
with open(os.path.join(tmpdir, "short.safetensors"), "wb") as fh:
    fh.write(b"\x01\x02")
try:
    Store(os.path.join(tmpdir, "short.safetensors"), block_size=BLOCK)
    check("rejects truncated file", False)
except ValueError:
    check("rejects truncated file", True)

print()
print("---- bounded cache ----")
store = Store(path, block_size=BLOCK)
cache = Cache(store, max_rows=4)
check("empty cache holds nothing", cache.nbytes() == 0 and cache.resident_rows == 0)
out = cache.gather_rows([3, 3, 1, 3, 1], np.float32)
check("cold gather shape/order", out.shape == (5, DIM) and np.array_equal(out[0], out[1]) and np.array_equal(out[0], out[3]))
check("cold gather dedups the fetch", store.rows_read == 2 and cache.stats.misses == 2 and cache.stats.hits == 0)
check("cold gather counts request vs unique", cache.stats.requested_rows == 5 and cache.stats.unique_rows == 2)
warm = cache.gather_rows([1, 3], np.float32)
check("warm gather reads nothing", store.rows_read == 2 and cache.stats.hits == 2 and cache.stats.misses == 2)
check("warm values equal cold values", np.array_equal(warm[0], out[2]) and np.array_equal(warm[1], out[0]))
check("residency bytes", cache.nbytes() == 2 * (DIM + DIM // BLOCK) and cache.resident_rows == 2)
cache.gather_rows([0, 2, 4, 5, 6], np.float32)
check("bound respected", cache.resident_rows == 4 and cache.nbytes() == 4 * (DIM + DIM // BLOCK))
check("evictions counted", cache.stats.evictions == 3)
before = store.rows_read
cache.gather_rows([3], np.float32)
check("evicted row refetched", store.rows_read == before + 1)
check("gather of nothing is empty", cache.gather_rows([], np.float32).shape == (0, DIM))
big = Cache(Store(path, block_size=BLOCK), max_rows=2)
out_big = big.gather_rows(list(range(ROWS)), np.float32)
check("oversized request still correct", out_big.shape == (ROWS, DIM) and big.resident_rows == 2)
try:
    Cache(store, max_rows=0)
    check("rejects zero bound", False)
except ValueError:
    check("rejects zero bound", True)

print()
print("fails:", fails)
