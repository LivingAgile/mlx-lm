"""Numpy validation part D: sharded embedding, all-reduce boundary, gate, module."""
import json, os, runpy, tempfile
import numpy as np

g = runpy.run_path("session-state/gl-w-00/files/_engram_harness.py")
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


Store = g["SafetensorsEngramRowStore"]
Cache = g["BoundedEngramRowCache"]
Embed = g["DeepseekV41EngramEmbedding"]
gate_fn = g["engram_signed_sqrt_sigmoid_gate"]
Engram = g["DeepseekV41Engram"]
EngramLayout = g["EngramLayout"]
dequant_rows = g["dequantize_engram_rows"]


def write_fixture(path, weight, scale):
    w = np.ascontiguousarray(weight, dtype=np.uint8)
    s = np.ascontiguousarray(scale, dtype=np.uint8)
    header = {
        "weight": {"dtype": "F8_E4M3", "shape": list(w.shape), "data_offsets": [0, w.nbytes]},
        "scale": {"dtype": "F8_E8M0", "shape": list(s.shape), "data_offsets": [w.nbytes, w.nbytes + s.nbytes]},
    }
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as fh:
        fh.write(len(blob).to_bytes(8, "little"))
        fh.write(blob)
        fh.write(w.tobytes())
        fh.write(s.tobytes())
    return path


BLOCK, DIM, ROWS = 4, 8, 11
rng = np.random.default_rng(5)
weight = rng.integers(0, 200, size=(ROWS, DIM), dtype=np.uint8)
scale = rng.integers(122, 130, size=(ROWS, DIM // BLOCK), dtype=np.uint8)
tmpdir = tempfile.mkdtemp()
full = write_fixture(os.path.join(tmpdir, "full.safetensors"), weight, scale)

print("---- embedding, world_size == 1 ----")
emb = Embed(ROWS, DIM, Cache(Store(full, block_size=BLOCK), max_rows=8), block_size=BLOCK)
check("no dense table attribute", not any(isinstance(v, np.ndarray) and v.shape == (ROWS, DIM) for v in emb.__dict__.values()))
ids = np.array([[0, 3, 3], [10, 0, 7]], dtype=np.int64)
out = emb(ids, np.float32)
ref = dequant_rows(weight[ids.reshape(-1)], scale[ids.reshape(-1)], BLOCK, np.float32).reshape(2, 3, DIM)
check("lookup matches direct dequant", np.allclose(out, ref, equal_nan=True))
check("shard span at ws=1", (emb.vocab_start_idx, emb.vocab_end_idx) == (0, ROWS))
check("nbytes tracks the cache bound", emb.nbytes() == emb.cache.nbytes() and emb.nbytes() <= 8 * (DIM + DIM // BLOCK))
for name, bad in [("negative id", np.array([[-1]], np.int64)), ("id past table", np.array([[ROWS]], np.int64))]:
    try:
        emb(bad)
        check("rejects " + name, False)
    except IndexError:
        check("rejects " + name, True)
check("empty request", emb(np.zeros((0, 3), np.int64), np.float32).shape == (0, 3, DIM))

print()
print("---- embedding, world_size == 2 ----")
part = -(-ROWS // 2)
padded = np.zeros((2 * part, DIM), dtype=np.uint8)
padded[:ROWS] = weight
padded_scale = np.zeros((2 * part, DIM // BLOCK), dtype=np.uint8)
padded_scale[:ROWS] = scale
shards = []
for r in range(2):
    p = write_fixture(
        os.path.join(tmpdir, "shard%d.safetensors" % r),
        padded[r * part : (r + 1) * part],
        padded_scale[r * part : (r + 1) * part],
    )
    shards.append(p)

partials = {}


def make(rank, reducer):
    return Embed(ROWS, DIM, Cache(Store(shards[rank], block_size=BLOCK), max_rows=8), rank=rank, world_size=2, all_reduce=reducer, block_size=BLOCK)


try:
    make(0, None)(ids)
    check("missing reducer fails loud", False)
except RuntimeError as exc:
    check("missing reducer fails loud", "all_reduce" in str(exc))

seen = []


def capture(rank):
    def reducer(arr):
        seen.append((rank, np.array(arr)))
        return arr
    return reducer


r0 = make(0, capture(0))(ids, np.float32)
r1 = make(1, capture(1))(ids, np.float32)
check("each shard zeroes the remote rows", float(np.abs(seen[0][1][1, 0]).sum()) == 0.0 and float(np.abs(seen[1][1][0, 0]).sum()) == 0.0)
check("summed shards equal the unsharded lookup", np.allclose(seen[0][1] + seen[1][1], ref, equal_nan=True))
check("reducer actually invoked per rank", len(seen) == 2)
try:
    make(0, lambda a: None)(ids)
    check("reducer returning None fails loud", False)
except RuntimeError:
    check("reducer returning None fails loud", True)
try:
    Embed(ROWS, DIM, Cache(Store(full, block_size=BLOCK), max_rows=4), rank=0, world_size=2, all_reduce=lambda a: a, block_size=BLOCK)
    check("wrong shard size fails loud", False)
except ValueError:
    check("wrong shard size fails loud", True)

print()
print("---- gate ----")


def reference_gate(x, key, weight, eps, clamp=1e-6):
    h = x.astype(np.float64)
    k = key.astype(np.float64)
    w = weight.astype(np.float64)
    dim = h.shape[-1]
    rstd = (1.0 / np.sqrt((h * h).mean(-1) + eps)) * (1.0 / np.sqrt((k * k).mean(-1) + eps))
    dot = (h * w * k).sum(-1) * rstd * dim ** -0.5
    mag = np.sqrt(np.maximum(np.abs(dot), clamp))
    return 1.0 / (1.0 + np.exp(-np.copysign(mag, dot)))


rng = np.random.default_rng(17)
x = rng.normal(size=(2, 3, 2, 8)).astype(np.float32)
key = rng.normal(size=(2, 3, 2, 8)).astype(np.float32)
w = rng.normal(size=(2, 8)).astype(np.float32)
check("gate matches reference", np.allclose(gate_fn(x, key, w, 1e-6), reference_gate(x, key, w, 1e-6), atol=1e-5))
check("gate is in (0, 1)", bool((gate_fn(x, key, w, 1e-6) > 0).all() and (gate_fn(x, key, w, 1e-6) < 1).all()))
zero = np.zeros((1, 1, 2, 8), np.float32)
check("all-zero stream hits the clamp plateau", np.allclose(gate_fn(zero, zero, w, 1e-6), 1.0 / (1.0 + np.exp(-(1e-6 ** 0.5))), atol=1e-6))
neg = gate_fn(x, -key, w, 1e-6)
pos = gate_fn(x, key, w, 1e-6)
check("sign of the dot flips the gate across 0.5", bool((((pos - 0.5) * (neg - 0.5)) < 0).mean() > 0.8))
scaled = gate_fn(x * 1000.0, key, w, 1e-6)
check("gate is scale-normalized in the stream", np.allclose(scaled, pos, atol=2e-3))
for name, args in [
    ("shape mismatch", (x, key[:, :2], w)),
    ("weight shape", (x, key, w[:1])),
]:
    try:
        gate_fn(*args, 1e-6)
        check("rejects " + name, False)
    except ValueError:
        check("rejects " + name, True)

print()
print("fails:", fails)
