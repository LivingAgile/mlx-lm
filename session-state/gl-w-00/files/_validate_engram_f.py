"""Numpy validation part F: gather cost is invariant to table size."""
import json, os, runpy, tempfile
import numpy as np

g = runpy.run_path("session-state/gl-w-00/files/_engram_harness.py")
Store = g["SafetensorsEngramRowStore"]
Cache = g["BoundedEngramRowCache"]
Embed = g["DeepseekV41EngramEmbedding"]
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def write_fixture(path, n_rows, dim, block, seed):
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 200, size=(n_rows, dim), dtype=np.uint8)
    s = np.full((n_rows, dim // block), 127, dtype=np.uint8)
    header = {
        "weight": {"dtype": "F8_E4M3", "shape": [n_rows, dim], "data_offsets": [0, w.nbytes]},
        "scale": {"dtype": "F8_E8M0", "shape": [n_rows, dim // block], "data_offsets": [w.nbytes, w.nbytes + s.nbytes]},
    }
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as fh:
        fh.write(len(blob).to_bytes(8, "little"))
        fh.write(blob)
        fh.write(w.tobytes())
        fh.write(s.tobytes())
    return path


DIM, BLOCK = 32, 32
tmp = tempfile.mkdtemp()
rng = np.random.default_rng(99)
ids = rng.integers(0, 512, size=(4, 12, 24), dtype=np.int64)
unique = len(set(ids.reshape(-1).tolist()))
row_bytes = DIM + DIM // BLOCK

results = []
for n_rows in (1024, 4096, 16384):
    p = write_fixture(os.path.join(tmp, "t%d.safetensors" % n_rows), n_rows, DIM, BLOCK, 4)
    store = Store(p, block_size=BLOCK)
    emb = Embed(n_rows, DIM, Cache(store, max_rows=256), block_size=BLOCK)
    out = emb(ids, np.float32)
    results.append((n_rows, os.path.getsize(p), store.bytes_read, emb.nbytes(), out.shape))
    print("  rows=%6d file=%8d read=%6d resident=%6d" % results[-1][:4])

check("read bytes identical across table sizes", len({r[2] for r in results}) == 1)
check("resident bytes identical across table sizes", len({r[3] for r in results}) == 1)
check("read bytes == unique rows exactly", results[0][2] == unique * row_bytes)
check("resident == min(unique, bound) exactly", results[0][3] == min(unique, 256) * row_bytes)
check("file sizes really did differ 16x", results[-1][1] > 15 * results[0][1])
check("output shape", results[0][4] == (4, 12, 24, DIM))
check("dedup was real", unique < ids.size)
print("  unique %d of %d requested" % (unique, ids.size))
print()
print("fails:", fails)
