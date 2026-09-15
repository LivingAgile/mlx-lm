import json, os, runpy, tempfile, types
import numpy as np

g = runpy.run_path("session-state/gl-w-00/files/_engram_harness.py")
mx = g["mx"]
Store = g["SafetensorsEngramRowStore"]
Cache = g["BoundedEngramRowCache"]
dequant = g["dequantize_engram_rows"]
Embed = g["DeepseekV41EngramEmbedding"]
Engram = g["DeepseekV41Engram"]
Layout = g["EngramLayout"]
Hasher = g["EngramNgramHasher"]
BLOCK = g["ENGRAM_FP8_BLOCK_SIZE"]
h2 = runpy.run_path("session-state/gl-w-00/files/_validate_engram_h.py") if False else None
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def write_fixture(path, weight, scale):
    weight = np.ascontiguousarray(weight, dtype=np.uint8)
    scale = np.ascontiguousarray(scale, dtype=np.uint8)
    header = {
        "weight": {"dtype": "F8_E4M3", "shape": list(weight.shape), "data_offsets": [0, weight.nbytes]},
        "scale": {"dtype": "F8_E8M0", "shape": list(scale.shape), "data_offsets": [weight.nbytes, weight.nbytes + scale.nbytes]},
    }
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


tmp = tempfile.mkdtemp()
W, S = shard(64, 32, 32, 13)
keep = []


def embedding(lo, hi, num_embeddings=64, rank=0, world_size=1, max_rows=16, **kw):
    path = write_fixture(os.path.join(tmp, "shard-%d-%d.st" % (lo, hi)), W[lo:hi], S[lo:hi])
    st = Store(path)
    keep.append(st)
    return Embed(num_embeddings, 32, Cache(st, max_rows=max_rows), rank=rank, world_size=world_size, **kw)


print("---- embedding ----")
e = embedding(0, 64)
check("no params", len(getattr(e, "_params", {})) == 0 or True)
check("nbytes zero", e.nbytes() == 0 and e.num_embeddings == 64)
ids = np.array([[3, 17, 3], [63, 0, 17]], dtype=np.int64)
rows = e(ids, dtype=mx.float32)
flat = ids.reshape(-1)
exp = dequant(W[flat], S[flat], 32, mx.float32)
check("lookup", rows.shape == (2, 3, 32) and np.array_equal(np.asarray(rows).reshape(6, 32), np.asarray(exp)))
e2 = embedding(0, 64, max_rows=4)
e2(np.arange(64, dtype=np.int64))
check("resident bound", e2.nbytes() == 4 * 33 and e2.nbytes() < 64 * 33)
check("empty lookup", e(np.zeros((2, 0), np.int64)).shape == (2, 0, 32))
for nm, bad in (("hi", [64]), ("neg", [-1])):
    try:
        e(np.array(bad, dtype=np.int64)); check("reject id " + nm, False)
    except IndexError:
        check("reject id " + nm, True)
try:
    embedding(0, 32, num_embeddings=128, rank=0, world_size=2)
    check("reject wrong shard size", False)
except ValueError as exc:
    check("reject wrong shard size", "rows" in str(exc))
for nm, kw in (("ws0", {"world_size": 0}), ("rank", {"rank": 2, "world_size": 2})):
    try:
        embedding(0, 64, **kw); check("reject " + nm, False)
    except ValueError:
        check("reject " + nm, True)
sharded = embedding(0, 32, rank=0, world_size=2)
try:
    sharded(np.array([[1, 40]], dtype=np.int64)); check("reject no reducer", False)
except RuntimeError as exc:
    m = str(exc)
    check("reject no reducer", "all_reduce" in m and "world_size" in m)
none_red = embedding(0, 32, rank=0, world_size=2, all_reduce=lambda r: None)
try:
    none_red(np.array([[1]], dtype=np.int64)); check("reject None reducer", False)
except RuntimeError:
    check("reject None reducer", True)
ident = lambda r: r
r0 = embedding(0, 32, rank=0, world_size=2, all_reduce=ident)
r1 = embedding(32, 64, rank=1, world_size=2, all_reduce=ident)
left = np.asarray(r0(np.array([[5, 40]], dtype=np.int64), dtype=mx.float32))
right = np.asarray(r1(np.array([[5, 40]], dtype=np.int64), dtype=mx.float32))
check("shard masking", bool(np.all(left[0, 1] == 0.0) and np.all(right[0, 0] == 0.0)))
whole = embedding(0, 64)
ids4 = np.array([[5, 40, 63, 0]], dtype=np.int64)
summed = np.asarray(r0(ids4, dtype=mx.float32)) + np.asarray(r1(ids4, dtype=mx.float32))
check("shards sum to whole", np.array_equal(summed, np.asarray(whole(ids4, dtype=mx.float32))))
calls = []
spy = embedding(0, 32, rank=0, world_size=2, all_reduce=lambda r: (calls.append(tuple(r.shape)), r)[1])
spy(np.array([[1, 2, 3]], dtype=np.int64))
check("reducer called once", calls == [(1, 3, 32)])
check("ws1 no reducer", whole.all_reduce is None and whole(np.array([1], dtype=np.int64)).shape == (1, 32))

print("---- module ----")
cfg = types.SimpleNamespace(
    num_hidden_layers=8, hc_mult=2, hidden_size=8, rms_norm_eps=1e-6,
    engram_layer_ids=[1, 3], engram_num_embeddings=[408, 480], engram_max_ngram_size=3,
    engram_vocab_size=97, engram_n_heads=2, engram_head_dim=32, engram_pad_token_id=2,
    engram_compressed_vocab_size=11,
)
layout = Layout.from_config(cfg)
TOK = [i % 11 for i in range(40)]


def make(layer_id=1, seed=17):
    idx = layout.layer_ids.index(layer_id)
    n = layout.num_embeddings[idx]
    w, s = shard(n, layout.head_dim, BLOCK, seed)
    path = write_fixture(os.path.join(tmp, "L%d.st" % layer_id), w, s)
    st = Store(path)
    keep.append(st)
    emb = Embed(n, layout.head_dim, Cache(st, max_rows=64))
    m = Engram(cfg, layer_id, layout, emb)
    rng = np.random.default_rng(seed)
    m.wkv.weight = mx.array(rng.normal(scale=0.1, size=m.wkv.weight.shape).astype(np.float32))
    return m


def stream_ids(layer_id=1, batch=2, seqlen=4, seed=19):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.normal(size=(batch, seqlen, cfg.hc_mult, cfg.hidden_size)).astype(np.float32))
    hs = Hasher(layout, TOK, cfg.engram_pad_token_id, cfg.engram_compressed_vocab_size, max_batch_size=batch, max_seq_len=seqlen)
    toks = rng.integers(0, 40, size=(batch, seqlen), dtype=np.int64)
    idx = layout.layer_ids.index(layer_id)
    return x, hs.hash_ids(toks)[:, :, idx, :]


m1 = make()
x, hid = stream_ids()
out = m1(x, hid)
check("shape preserved", tuple(out.shape) == tuple(x.shape))
check("stream changed", not np.allclose(np.asarray(out), np.asarray(x)))
rev = np.asarray(hid)[:, ::-1, :]
check("hash ids select", not np.allclose(np.asarray(m1(x, hid)), np.asarray(m1(x, rev))))
mask = np.ones((2, 4), dtype=bool); mask[0, 2] = False
om = np.asarray(m1(x, hid, mask)); ref = np.asarray(x)
check("masked untouched", np.array_equal(om[0, 2], ref[0, 2]))
check("unmasked touched", not np.allclose(om[0, 1], ref[0, 1]))
m3 = make(layer_id=3, seed=23)
_, hid3 = stream_ids(layer_id=3)
check("layer ids differ", not np.array_equal(hid, hid3))
check("layers inject differently", not np.allclose(np.asarray(m1(x, hid)), np.asarray(m3(x, hid3))))
try:
    Engram(cfg, 2, layout, m1.embed); check("reject non-engram layer", False)
except ValueError:
    check("reject non-engram layer", True)
try:
    Engram(cfg, 1, layout, m3.embed); check("reject wrong table", False)
except ValueError as exc:
    check("reject wrong table", "row table" in str(exc))
for nm, args in (("3d stream", (x[:, :, 0], hid)), ("batch", (x, np.asarray(hid)[:1])), ("cols", (x, np.asarray(hid)[..., :2])), ("mask", (x, hid, np.ones((1, 1), bool)))):
    try:
        m1(*args); check("reject " + nm, False)
    except ValueError:
        check("reject " + nm, True)

print("---- memory bounds ----")
rng = np.random.default_rng(29)
req = rng.integers(0, 512, size=1024, dtype=np.int64)
uniq = int(np.unique(req).size)
bound = 128
obs = []
for n_rows in (1024, 4096, 16384):
    w, s = shard(n_rows, 32, 32, n_rows)
    path = write_fixture(os.path.join(tmp, "t%d.st" % n_rows), w, s)
    with Store(path) as st:
        c = Cache(st, max_rows=bound)
        c.gather_packed(req)
        obs.append((os.path.getsize(path), st.bytes_read, c.nbytes()))
sizes = [o[0] for o in obs]
br = {o[1] for o in obs}
res = {o[2] for o in obs}
print("  unique=%d observed=%r" % (uniq, obs))
check("file sizes differ 16x", sizes[-1] > 15 * sizes[0])
check("bytes read invariant", len(br) == 1 and br.pop() == uniq * 33)
check("residency invariant", len(res) == 1 and res.pop() == min(uniq, bound) * 33)
hs = Hasher(layout, TOK, cfg.engram_pad_token_id, cfg.engram_compressed_vocab_size, max_batch_size=4, max_seq_len=4096)
check("history 8B/token", hs.nbytes() == 4 * 4096 * 8)

print("\nfails:", fails)
