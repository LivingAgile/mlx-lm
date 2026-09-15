import runpy, types
import numpy as np

g = runpy.run_path("session-state/gl-w-00/files/_engram_harness.py")
EngramLayout = g["EngramLayout"]
Hasher = g["EngramNgramHasher"]
fails = []

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

BASE = dict(
    num_hidden_layers=8,
    hc_mult=2,
    engram_layer_ids=[1, 3],
    engram_num_embeddings=[10**9, 10**9],
    engram_max_ngram_size=3,
    engram_vocab_size=97,
    engram_n_heads=2,
    engram_head_dim=32,
    engram_pad_token_id=2,
    engram_compressed_vocab_size=11,
)

def cfg(**ov):
    d = dict(BASE)
    d.update(ov)
    c = types.SimpleNamespace(**d)
    if "engram_num_embeddings" not in ov and d["engram_layer_ids"]:
        lay = EngramLayout.from_config(c)
        d["engram_num_embeddings"] = [lay.bucket_span(i) for i in range(len(lay.layer_ids))]
        c = types.SimpleNamespace(**d)
    return c

TOK = [i % 11 for i in range(40)]
layout = EngramLayout.from_config(cfg())
print("tiny primes L0", layout.flat_primes(0), "L1", layout.flat_primes(1))
print("tiny spans", layout.bucket_span(0), layout.bucket_span(1))
print("n_hash_cols", layout.n_hash_cols)

def mk(max_seq_len=32, max_batch_size=2, **ov):
    c = cfg(**ov)
    lay = EngramLayout.from_config(c)
    return lay, Hasher(lay, TOK, c.engram_pad_token_id, c.engram_compressed_vocab_size,
                       max_batch_size=max_batch_size, max_seq_len=max_seq_len)

_, h = mk()
ids = h.hash_ids(np.arange(12, dtype=np.int64).reshape(2, 6))
check("shape", ids.shape == (2, 6, 2, 4) and ids.dtype == np.int64)

lay, h = mk()
rng = np.random.default_rng(1)
ids = h.hash_ids(rng.integers(0, 40, size=(2, 10), dtype=np.int64))
off = lay.bucket_offsets()
ok = True
for L in range(2):
    fl = lay.flat_primes(L)
    for c_ in range(lay.n_hash_cols):
        lo = int(off[L][c_])
        col = ids[:, :, L, c_]
        ok = ok and int(col.min()) >= lo and int(col.max()) < lo + fl[c_]
check("in bucket range", ok)

_, whole = mk(); _, split = mk()
rng = np.random.default_rng(2)
toks = rng.integers(0, 40, size=(2, 9), dtype=np.int64)
one = whole.hash_ids(toks, 0)
pieces = [split.hash_ids(toks[:, :5], 0)]
for s in range(5, 9):
    pieces.append(split.hash_ids(toks[:, s:s+1], s))
check("prefill==split", np.array_equal(one, np.concatenate(pieces, axis=1)))

_, h = mk()
ids = h.hash_ids(np.array([[7, 8, 9, 1, 7, 8, 9]], dtype=np.int64))
check("repeat ngram", np.array_equal(ids[0, 2], ids[0, 6]) and not np.array_equal(ids[0, 2], ids[0, 3]))

_, h = mk()
ids = h.hash_ids(np.array([[3, 4, 5], [14, 15, 16]], dtype=np.int64))
check("normalize alike", np.array_equal(ids[0], ids[1]))

toks = np.array([[5, 6, 9, 5, 6]], dtype=np.int64)
mask = np.ones((1, 5), dtype=bool); mask[0, 2] = False
_, h = mk(); masked = h.hash_ids(toks, 0, mask)
_, h2 = mk(); plain = h2.hash_ids(toks, 0)
check("dead blocks lookback", np.array_equal(masked[0, 3], masked[0, 0]))
check("unmasked differs", not np.array_equal(plain[0, 3], plain[0, 0]))

_, h = mk()
ids = h.hash_ids(np.array([[3, 4, 5, 6]], dtype=np.int64))
check("layers differ", not np.array_equal(ids[:, :, 0], ids[:, :, 1]))

_, h = mk(max_seq_len=64, max_batch_size=4)
check("nbytes", h.nbytes() == 4 * 64 * 8 and h.cache.shape == (4, 64))

_, h = mk()
h.hash_ids(np.array([[5, 6, 7]], dtype=np.int64), 0)
after = h.hash_ids(np.array([[8]], dtype=np.int64), 3)
h.reset()
fresh = h.hash_ids(np.array([[8]], dtype=np.int64), 3)
_, h3 = mk()
start = h3.hash_ids(np.array([[8]], dtype=np.int64), 0)
check("reset changes", not np.array_equal(after, fresh))
check("reset == fresh sequence", np.array_equal(fresh, start))

lay = EngramLayout.from_config(cfg())
for name, fn in [
    ("map too wide", lambda: Hasher(lay, [0, 1, 2, 99], 2, 11)),
    ("pad oob", lambda: Hasher(lay, TOK, 999, 11)),
    ("seq0", lambda: Hasher(lay, TOK, 2, 11, max_seq_len=0)),
]:
    try:
        fn(); check(name, False)
    except ValueError as e:
        check(name, "compressed" in str(e) if name == "map too wide" else True)

_, h = mk(max_seq_len=8, max_batch_size=2)
cases = [
    ("batch", lambda: h.hash_ids(np.zeros((3, 2), dtype=np.int64))),
    ("seq", lambda: h.hash_ids(np.zeros((1, 4), dtype=np.int64), 6)),
    ("tok hi", lambda: h.hash_ids(np.array([[99]], dtype=np.int64))),
    ("tok neg", lambda: h.hash_ids(np.array([[-1]], dtype=np.int64))),
    ("1d", lambda: h.hash_ids(np.zeros((4,), dtype=np.int64))),
    ("mask", lambda: h.hash_ids(np.zeros((1, 2), dtype=np.int64), 0, np.ones((1, 3), bool))),
]
for name, fn in cases:
    try:
        fn(); check("reject " + name, False)
    except ValueError:
        check("reject " + name, True)

print("\nfails:", fails)
