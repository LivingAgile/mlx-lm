"""Numpy validation part B: hasher, store, cache, dequant, embedding, gate."""
import json, os, runpy, tempfile
import numpy as np

g = runpy.run_path("session-state/gl-w-00/files/_engram_harness.py")
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


EngramLayout = g["EngramLayout"]
EngramNgramHasher = g["EngramNgramHasher"]
compute_mult = g["compute_engram_hash_multipliers"]
dequant_rows = g["dequantize_engram_rows"]
Store = g["SafetensorsEngramRowStore"]
Cache = g["BoundedEngramRowCache"]
Embed = g["DeepseekV41EngramEmbedding"]
gate_fn = g["engram_signed_sqrt_sigmoid_gate"]
DEAD = g["ENGRAM_DEAD_TOKEN"]


class Cfg:
    def __init__(self, **kw):
        self.vocab_size = 40
        self.hidden_size = 8
        self.num_hidden_layers = 4
        self.hc_mult = 2
        self.rms_norm_eps = 1e-6
        self.engram_layer_ids = [1, 3]
        self.engram_num_embeddings = [0, 0]
        self.engram_max_ngram_size = 3
        self.engram_vocab_size = 97
        self.engram_n_heads = 2
        self.engram_head_dim = 32
        self.engram_pad_token_id = 2
        self.engram_compressed_vocab_size = 11
        self.__dict__.update(kw)


probe = Cfg()
probe.engram_num_embeddings = [10 ** 9, 10 ** 9]
tmp = EngramLayout.from_config(probe)
rows = [tmp.bucket_span(0), tmp.bucket_span(1)]
cfg = Cfg(engram_num_embeddings=rows)
layout = EngramLayout.from_config(cfg)
print("tiny layout rows", rows, "cols", layout.n_hash_cols)

token_map = [i % 11 for i in range(40)]
hasher = EngramNgramHasher(layout, token_map, 2, 11, max_batch_size=2, max_seq_len=16)


def reference_hashes(layout, token_map, pad_token_id, compressed_vocab, batch_ids_seq, masks_seq, max_b, max_s):
    """Independent transcription of inference/engram.py NgramHashState.forward."""
    tm = np.asarray(token_map, dtype=np.int64)
    pad_id = int(tm[pad_token_id])
    flat = [[p for per in layer for p in per] for layer in layout.primes]
    offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64)
    primes = np.array(layout.primes, dtype=np.int64)
    mult = compute_mult(layout.layer_ids, layout.max_ngram_size, compressed_vocab)
    cache = np.zeros((max_b, max_s), dtype=np.int64)
    out = []
    start = 0
    for ids, mask in zip(batch_ids_seq, masks_seq):
        ids = np.asarray(ids, dtype=np.int64)
        b, l = ids.shape
        comp = tm[ids]
        if mask is not None:
            comp = np.where(np.asarray(mask, dtype=bool), comp, -1)
        cache[:b, start : start + l] = comp
        pos = np.broadcast_to(np.arange(start, start + l, dtype=np.int64), (b, l))
        blocked = np.zeros_like(pos, dtype=bool)
        toks = []
        for shift in range(layout.max_ngram_size):
            src = np.take_along_axis(cache[:b], np.clip(pos - shift, 0, None), axis=1)
            blocked = blocked | (pos < shift) | (src == -1)
            toks.append(np.where(blocked, pad_id, src))
        stacked = np.stack(toks, axis=-1)
        prod = stacked[:, :, None, :] * mult
        rolling = prod[..., 0]
        hs = []
        for i in range(1, layout.max_ngram_size):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            hs.append(rolling[..., None] % primes[:, i - 1])
        out.append(np.concatenate(hs, axis=-1) + offsets)
        start += l
    return out


rng = np.random.default_rng(3)
prefill = rng.integers(0, 40, size=(2, 5), dtype=np.int64)
decode1 = rng.integers(0, 40, size=(2, 1), dtype=np.int64)
decode2 = rng.integers(0, 40, size=(2, 1), dtype=np.int64)
pmask = np.ones((2, 5), dtype=bool)
pmask[0, 2] = False
seqs = [prefill, decode1, decode2]
masks = [pmask, None, None]
ref = reference_hashes(layout, token_map, 2, 11, seqs, masks, 2, 16)

got = []
start = 0
for ids, mask in zip(seqs, masks):
    got.append(hasher.hash_ids(ids, start, mask))
    start += ids.shape[1]
check("hasher matches reference (prefill+2 decode)", all(np.array_equal(a, b) for a, b in zip(ref, got)))
check("hash shape", got[0].shape == (2, 5, 2, layout.n_hash_cols))
check("hash ids within table", all(int(h.min()) >= 0 for h in got) and int(got[0][:, :, 0].max()) < rows[0])
check("layers hash differently", not np.array_equal(got[0][:, :, 0], got[0][:, :, 1]))
check("dead token blocks lookback", np.array_equal(got[0][0, 3, 0, :2], got[0][0, 3, 0, :2]))
check("history cache is 8 bytes/token/batch", hasher.nbytes() == 2 * 16 * 8)

h2 = EngramNgramHasher(layout, token_map, 2, 11, max_batch_size=2, max_seq_len=16)
one_shot = h2.hash_ids(np.concatenate([prefill, decode1, decode2], axis=1), 0, np.concatenate([pmask, np.ones((2, 2), bool)], axis=1))
check("prefill-in-one equals split prefill/decode", np.array_equal(one_shot[:, 5:6], got[1]) and np.array_equal(one_shot[:, 6:7], got[2]))

for name, fn in [
    ("batch over bound", lambda: hasher.hash_ids(np.zeros((3, 2), np.int64), 0)),
    ("seq past cache", lambda: hasher.hash_ids(np.zeros((1, 2), np.int64), 15)),
    ("token id out of vocab", lambda: hasher.hash_ids(np.array([[99]], np.int64), 0)),
    ("1-D input", lambda: hasher.hash_ids(np.zeros((4,), np.int64), 0)),
    ("mask shape", lambda: hasher.hash_ids(np.zeros((1, 2), np.int64), 0, np.ones((1, 3), bool))),
]:
    try:
        fn()
        check("rejects " + name, False)
    except ValueError:
        check("rejects " + name, True)

print()
print("fails:", fails)
