import io, json, os, runpy, tempfile, types
import numpy as np

g = runpy.run_path("session-state/gl-w-00/files/_engram_harness.py")
mx = g["mx"]
Store = g["SafetensorsEngramRowStore"]
Cache = g["BoundedEngramRowCache"]
dequant = g["dequantize_engram_rows"]
gate_fn = g["engram_signed_sqrt_sigmoid_gate"]
Embed = g["DeepseekV41EngramEmbedding"]
BLOCK = g["ENGRAM_FP8_BLOCK_SIZE"]
CLAMP = g["ENGRAM_GATE_CLAMP"]
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def write_fixture(path, weight, scale, weight_dtype="F8_E4M3", scale_dtype="F8_E8M0", mutate=None):
    weight = np.ascontiguousarray(weight, dtype=np.uint8)
    scale = np.ascontiguousarray(scale, dtype=np.uint8)
    header = {
        "weight": {"dtype": weight_dtype, "shape": list(weight.shape), "data_offsets": [0, weight.nbytes]},
        "scale": {"dtype": scale_dtype, "shape": list(scale.shape), "data_offsets": [weight.nbytes, weight.nbytes + scale.nbytes]},
    }
    if mutate is not None:
        mutate(header)
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as handle:
        handle.write(len(blob).to_bytes(8, "little"))
        handle.write(blob)
        handle.write(weight.tobytes())
        handle.write(scale.tobytes())
    return path


def shard(n_rows, dim, block_size, seed):
    rng = np.random.default_rng(seed)
    magnitude = rng.integers(0, 0x78, size=(n_rows, dim), dtype=np.uint8)
    sign = rng.integers(0, 2, size=(n_rows, dim), dtype=np.uint8) << 7
    return (magnitude | sign).astype(np.uint8), rng.integers(120, 132, size=(n_rows, dim // block_size), dtype=np.uint8)


def decode_e4m3fn(codes):
    codes = np.asarray(codes, dtype=np.uint8).astype(np.int64)
    sign = np.where(codes >> 7 == 1, -1.0, 1.0)
    exponent = (codes >> 3) & 0xF
    mantissa = (codes & 0x7).astype(np.float64)
    normal = (1.0 + mantissa / 8.0) * np.power(2.0, exponent.astype(np.float64) - 7.0)
    return sign * np.where(exponent == 0, mantissa * (2.0**-9), normal)


def ref_gate(stream, key, weight, eps, clamp=1e-6):
    h = np.asarray(stream, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    dim = h.shape[-1]
    hr = 1.0 / np.sqrt(np.mean(h * h, axis=-1) + eps)
    kr = 1.0 / np.sqrt(np.mean(k * k, axis=-1) + eps)
    dot = np.sum(h * w * k, axis=-1) * hr * kr / np.sqrt(dim)
    return 1.0 / (1.0 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(dot), clamp)), dot)))


tmp = tempfile.mkdtemp()
W, S = shard(64, 32, 32, 7)
P = write_fixture(os.path.join(tmp, "e.safetensors"), W, S)

print("---- store ----")
st = Store(P)
check("geometry", (st.num_rows, st.dim, st.block_size, st.scale_dim, st.row_nbytes) == (64, 32, 32, 1, 33))
wanted = [5, 0, 63, 5]
w2, s2 = st.read_rows(wanted)
check("exact rows", np.array_equal(w2, W[wanted]) and np.array_equal(s2, S[wanted]))
check("counters", st.rows_read == 4 and st.bytes_read == 4 * 33)
check("file bigger than read", os.path.getsize(P) > st.bytes_read)
st2 = Store(P)
we, se = st2.read_rows([])
check("empty read", we.shape == (0, 32) and se.shape == (0, 1) and st2.bytes_read == 0)
for nm, rid in (("hi", [64]), ("neg", [-1])):
    try:
        st2.read_rows(rid); check("reject row " + nm, False)
    except IndexError:
        check("reject row " + nm, True)
with Store(P) as ctx:
    first = ctx.read_rows([4])[0]
    ctx.close()
    check("reopen", np.array_equal(ctx.read_rows([4])[0], first))


def reject(name, frag, **kw):
    path = write_fixture(os.path.join(tmp, name), W, S, **kw)
    try:
        Store(path); check("reject " + name, False)
    except ValueError as exc:
        check("reject " + name + " [" + frag + "]", frag in str(exc))


reject("bf16.st", "dtype", weight_dtype="BF16")
reject("f32s.st", "dtype", scale_dtype="F32")
reject("scale.st", "requires", mutate=lambda h: h["scale"].__setitem__("shape", [32, 2]))
reject("offs.st", "bytes", mutate=lambda h: h["weight"].__setitem__("data_offsets", [0, 16]))


def overrun(h):
    size = h["scale"]["data_offsets"][1]
    h["scale"]["shape"] = [64, 1]
    h["scale"]["data_offsets"] = [size, size + 64]


reject("over.st", "bytes long", mutate=overrun)
reject("miss.st", "no tensor named", mutate=lambda h: h.__setitem__("quant_scale", h.pop("scale")))


def rename(h):
    h["engram.weight"] = h.pop("weight")
    h["engram.scale"] = h.pop("scale")


named = write_fixture(os.path.join(tmp, "named.st"), W, S, mutate=rename)
st3 = Store(named, weight_key="engram.weight", scale_key="engram.scale")
check("custom keys", np.array_equal(st3.read_rows([2])[0][0], W[2]))
short = os.path.join(tmp, "short.st")
blob = io.open(P, "rb").read()
io.open(short, "wb").write(blob[: len(blob) - 40])
try:
    Store(short); check("reject truncated", False)
except ValueError:
    check("reject truncated", True)
junk = os.path.join(tmp, "junk.bin")
io.open(junk, "wb").write(b"not a safetensors file")
try:
    Store(junk); check("reject junk", False)
except ValueError:
    check("reject junk", True)

print("---- cache ----")
W2, S2 = shard(64, 32, 32, 11)
P2 = write_fixture(os.path.join(tmp, "c.safetensors"), W2, S2)
stc = Store(P2); c = Cache(stc, max_rows=8)
u, w, s, inv = c.gather_packed([3, 3, 7, 3, 7, 1])
check("dedup order", list(u) == [3, 7, 1] and list(inv) == [0, 0, 1, 0, 1, 2])
check("dedup shapes/stats", w.shape == (3, 32) and s.shape == (3, 1) and stc.rows_read == 3 and c.stats.requested_rows == 6 and c.stats.unique_rows == 3 and c.stats.misses == 3 and c.stats.hits == 0)
stc2 = Store(P2); c2 = Cache(stc2, max_rows=8)
c2.gather_packed([3, 7, 1]); cold = stc2.bytes_read
c2.gather_packed([1, 7, 3, 3])
check("warm reads nothing", stc2.bytes_read == cold and c2.stats.hits == 3 and c2.stats.rows_fetched == 3)
stc3 = Store(P2); c3 = Cache(stc3, max_rows=8)
req = [9, 2, 9, 40, 2]
_, w3, _, inv3 = c3.gather_packed(req)
check("restore order", np.array_equal(w3[inv3], W2[req]))
rows = c3.gather_rows(req)
exp = dequant(W2[req], S2[req])
check("gather_rows equals direct", rows.shape == (5, 32) and np.array_equal(np.asarray(rows), np.asarray(exp)))
stc4 = Store(P2); c4 = Cache(stc4, max_rows=8)
check("nbytes zero", c4.nbytes() == 0)
c4.gather_packed([1, 2, 3])
check("nbytes 3", c4.resident_rows == 3 and c4.nbytes() == 3 * 33)
c4.clear()
check("clear", c4.resident_rows == 0 and c4.nbytes() == 0)
stc5 = Store(P2); c5 = Cache(stc5, max_rows=4)
for r in range(12):
    c5.gather_packed([r])
check("bound", c5.resident_rows == 4 and c5.nbytes() == 4 * 33 and c5.stats.evictions == 8 and stc5.rows_read == 12)
stc6 = Store(P2); c6 = Cache(stc6, max_rows=2)
c6.gather_packed([1]); c6.gather_packed([2]); c6.gather_packed([1]); c6.gather_packed([3])
before = stc6.rows_read
c6.gather_packed([1])
ok = stc6.rows_read == before
c6.gather_packed([2])
check("lru", ok and stc6.rows_read == before + 1)
stc7 = Store(P2); c7 = Cache(stc7, max_rows=1)
f = c7.gather_packed([5])[1].copy(); c7.gather_packed([6]); again = c7.gather_packed([5])[1]
check("refetch identical", np.array_equal(f, again) and np.array_equal(again[0], W2[5]))
stc8 = Store(P2); c8 = Cache(stc8, max_rows=4)
req = list(range(16))
_, w8, _, inv8 = c8.gather_packed(req)
check("oversized served then trimmed", np.array_equal(w8[inv8], W2[req]) and c8.resident_rows == 4 and c8.nbytes() == 4 * 33)
stc9 = Store(P2); c9 = Cache(stc9, max_rows=8)
u9, w9, s9, i9 = c9.gather_packed([])
check("empty gather", w9.shape == (0, 32) and s9.shape == (0, 1) and u9.size == 0 and i9.size == 0 and stc9.bytes_read == 0 and c9.gather_rows([]).shape == (0, 32))
try:
    Cache(stc9, max_rows=0); check("reject max_rows 0", False)
except ValueError:
    check("reject max_rows 0", True)

print("---- dequant ----")
wd, sd = shard(6, 64, BLOCK, 3)
rows = dequant(wd, sd, BLOCK, mx.float32)
expected = decode_e4m3fn(wd).reshape(6, 2, BLOCK) * np.power(2.0, sd.astype(np.float64) - 127.0)[..., None]
check("dequant matches e4m3 spec", rows.shape == (6, 64) and np.allclose(np.asarray(rows), expected.reshape(6, 64), rtol=1e-6, atol=0.0))
r2 = np.asarray(dequant(np.full((1, 64), 0x38, np.uint8), np.array([[127, 130]], np.uint8), BLOCK, mx.float32))
check("block tiling", np.all(r2[0, :32] == 1.0) and np.all(r2[0, 32:] == 8.0))
check("empty dequant", dequant(np.zeros((0, 32), np.uint8), np.zeros((0, 1), np.uint8)).shape == (0, 32))
wr, sr = shard(4, 32, 32, 4)
for nm, args in (("rows", (wr, sr[:2])), ("grid", (wr, np.zeros((4, 2), np.uint8))), ("1d", (wr[0], sr[0])), ("width", (np.zeros((4, 20), np.uint8), sr))):
    try:
        dequant(*args); check("reject dequant " + nm, False)
    except ValueError:
        check("reject dequant " + nm, True)

print("---- gate ----")
rng = np.random.default_rng(5)
stream = rng.normal(size=(2, 3, 2, 16)).astype(np.float32)
key = rng.normal(size=(2, 3, 2, 16)).astype(np.float32)
weight = rng.normal(size=(2, 16)).astype(np.float32)
gate = gate_fn(mx.array(stream), mx.array(key), mx.array(weight), 1e-6)
check("gate matches reference", gate.shape == (2, 3, 2) and np.allclose(np.asarray(gate), ref_gate(stream, key, weight, 1e-6), atol=1e-6))
rng = np.random.default_rng(6)
loud_s = (rng.normal(size=(2, 3, 2, 16)).astype(np.float32)) * 50.0
loud_k = rng.normal(size=(2, 3, 2, 16)).astype(np.float32)
loud_w = rng.normal(size=(2, 16)).astype(np.float32)
gl = np.asarray(gate_fn(mx.array(loud_s), mx.array(loud_k), mx.array(loud_w), 1e-6))
check("gate in (0,1)", bool(np.all(gl > 0.0) and np.all(gl < 1.0)))
rng = np.random.default_rng(7)
bs = rng.normal(size=(2, 3, 2, 16)).astype(np.float32)
bk = rng.normal(size=(2, 3, 2, 16)).astype(np.float32)
bw = rng.normal(size=(2, 16)).astype(np.float32)
base = np.asarray(gate_fn(mx.array(bs), mx.array(bk), mx.array(bw), 1e-6))
loud = np.asarray(gate_fn(mx.array(bs * 100.0), mx.array(bk), mx.array(bw), 1e-6))
check("scale invariance", np.allclose(base, loud, atol=1e-4))
rng = np.random.default_rng(8)
zk = rng.normal(size=(2, 3, 2, 16)).astype(np.float32)
zw = rng.normal(size=(2, 16)).astype(np.float32)
zg = np.asarray(gate_fn(mx.array(np.zeros((1, 1, 2, 16), np.float32)), mx.array(zk[:1, :1]), mx.array(zw), 1e-6))
check("clamp plateau", np.allclose(zg, 1.0 / (1.0 + np.exp(-np.sqrt(CLAMP))), atol=1e-6))
rng = np.random.default_rng(9)
ps = rng.normal(size=(2, 3, 2, 16)).astype(np.float32)
_ = rng.normal(size=(2, 3, 2, 16))
pw = np.abs(rng.normal(size=(2, 16)).astype(np.float32))
pos = np.asarray(gate_fn(mx.array(ps), mx.array(ps), mx.array(pw), 1e-6))
neg = np.asarray(gate_fn(mx.array(ps), mx.array(-ps), mx.array(pw), 1e-6))
check("sign flip", bool(np.all(pos > 0.5) and np.all(neg < 0.5)))
for nm, args in (("key", (mx.array(stream), mx.array(key[:1]), mx.array(weight), 1e-6)), ("weight", (mx.array(stream), mx.array(key), mx.array(weight[:1]), 1e-6))):
    try:
        gate_fn(*args); check("reject gate " + nm, False)
    except ValueError:
        check("reject gate " + nm, True)
try:
    gate_fn(mx.array(stream), mx.array(key), mx.array(weight), 1e-6, clamp_value=0.0)
    check("reject clamp", False)
except ValueError:
    check("reject clamp", True)

print("\nfails:", fails)
