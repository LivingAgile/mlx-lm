"""Numpy validation part E: end-to-end Engram module and bounded-memory property."""
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
Engram = g["DeepseekV41Engram"]
EngramLayout = g["EngramLayout"]
Hasher = g["EngramNgramHasher"]


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


class Cfg:
    def __init__(self, **kw):
        self.vocab_size = 40
        self.hidden_size = 8
        self.num_hidden_layers = 4
        self.hc_mult = 2
        self.rms_norm_eps = 1e-6
        self.engram_layer_ids = [1, 3]
        self.engram_num_embeddings = [10 ** 9, 10 ** 9]
        self.engram_max_ngram_size = 3
        self.engram_vocab_size = 97
        self.engram_n_heads = 2
        self.engram_head_dim = 32
        self.engram_pad_token_id = 2
        self.engram_compressed_vocab_size = 11
        self.__dict__.update(kw)


probe = EngramLayout.from_config(Cfg())
rows = [probe.bucket_span(0), probe.bucket_span(1)]
cfg = Cfg(engram_num_embeddings=rows)
layout = EngramLayout.from_config(cfg)
HEAD, BLOCK = layout.head_dim, 32
tmpdir = tempfile.mkdtemp()
rng = np.random.default_rng(23)

embs = []
stores = []
for i, n in enumerate(rows):
    w = rng.integers(0, 200, size=(n, HEAD), dtype=np.uint8)
    s = np.full((n, HEAD // BLOCK), 127, dtype=np.uint8)
    p = write_fixture(os.path.join(tmpdir, "t%d.safetensors" % i), w, s)
    st = Store(p, block_size=BLOCK)
    stores.append(st)
    embs.append(Embed(n, HEAD, Cache(st, max_rows=64), block_size=BLOCK))

token_map = [i % 11 for i in range(40)]
hasher = Hasher(layout, token_map, 2, 11, max_batch_size=2, max_seq_len=64)

engrams = [Engram(cfg, lid, layout, embs[i]) for i, lid in enumerate(layout.layer_ids)]
check("module exposes no dense table", all(not any(getattr(v, "shape", None) == (rows[i], HEAD) for v in e.embed.__dict__.values()) for i, e in enumerate(engrams)))

B, L = 2, 6
ids = rng.integers(0, 40, size=(B, L), dtype=np.int64)
mask = np.ones((B, L), dtype=bool)
mask[1, 2:4] = False
hashes = hasher.hash_ids(ids, 0, mask)
check("hash tensor shape", hashes.shape == (B, L, 2, layout.n_hash_cols))

x = rng.normal(size=(B, L, cfg.hc_mult, cfg.hidden_size)).astype(np.float32)
out0 = engrams[0](x, hashes[:, :, 0], mask)
check("output shape preserved", out0.shape == x.shape)
check("masked positions pass through untouched", np.allclose(out0[1, 2:4], x[1, 2:4], atol=0))
check("unmasked positions are modified", not np.allclose(out0[0], x[0]))
out_unmasked = engrams[0](x, hashes[:, :, 0])
check("mask is what shuts the gate", not np.allclose(out_unmasked[1, 2:4], x[1, 2:4]))
out1 = engrams[1](x, hashes[:, :, 1], mask)
check("the two engram layers differ", not np.allclose(out0, out1))

for name, fn in [
    ("wrong stream rank", lambda: engrams[0](x[0], hashes[:, :, 0])),
    ("wrong hash cols", lambda: engrams[0](x, hashes[:, :, 0, :2])),
    ("mismatched batch", lambda: engrams[0](x, hashes[:1, :, 0])),
    ("bad mask shape", lambda: engrams[0](x, hashes[:, :, 0], np.ones((B, L + 1), bool))),
]:
    try:
        fn()
        check("rejects " + name, False)
    except ValueError:
        check("rejects " + name, True)

print()
print("---- bounded memory ----")
requested = B * L * layout.n_hash_cols
unique = len(set(hashes[:, :, 0].reshape(-1).tolist()))
row_bytes = HEAD + HEAD // BLOCK
cache0 = embs[0].cache
check("requested vs unique dedup", cache0.stats.requested_rows >= requested and cache0.stats.unique_rows < cache0.stats.requested_rows)
check("store read only unique rows", stores[0].rows_read <= unique)
check("bytes read bounded by unique rows", stores[0].bytes_read <= unique * row_bytes)
check("residency bounded by the cache bound", cache0.nbytes() <= 64 * row_bytes)
table_bytes = rows[0] * row_bytes
# Scale invariance, not a ratio, is the real proof: _validate_engram_f.py.
check("resident is exactly the bound", cache0.nbytes() == min(unique, 64) * row_bytes)
print("  table bytes %d, resident %d, read %d, unique %d of %d requested" % (table_bytes, cache0.nbytes(), stores[0].bytes_read, unique, requested))

print()
print("fails:", fails)
