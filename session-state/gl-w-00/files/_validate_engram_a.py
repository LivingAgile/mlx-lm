"""Numpy validation of the Engram slice against an independent transcription."""
import json, os, runpy, struct, tempfile
import numpy as np

g = runpy.run_path("session-state/gl-w-00/files/_engram_harness.py")
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


norm = g["normalize_engram_token_text"]
build_map = g["build_engram_compressed_token_map"]
find_next_prime = g["find_next_prime"]
EngramLayout = g["EngramLayout"]
validate_engram_config = g["validate_engram_config"]
EngramNgramHasher = g["EngramNgramHasher"]
compute_mult = g["compute_engram_hash_multipliers"]
dequant_rows = g["dequantize_engram_rows"]
Store = g["SafetensorsEngramRowStore"]
Cache = g["BoundedEngramRowCache"]
Embed = g["DeepseekV41EngramEmbedding"]
gate_fn = g["engram_signed_sqrt_sigmoid_gate"]
Engram = g["DeepseekV41Engram"]

print("---- normalization ----")
check("case/space fold", norm(" The") == "the" and norm("the") == "the" and norm("THE") == "the")
check("accents stripped", norm("Cafe\u0301") == "cafe" and norm("\u00c9\u00c0") == "ea")
check("nfkc applied", norm("\uff21\uff22") == "ab")
check("lone space survives", norm(" ") == " ")
check("whitespace run folds to lone space", norm("\n\t ") == " ")
check("inner run collapses", norm("a \t b") == "a b")
check("nbsp trimmed by strip", norm("\u00a0x\u00a0") == "x")
check("empty stays empty", norm("") == "")

decoded = [" The", "the", "THE", " ", "\n", "caf\u00e9", "cafe", "\ufffd", "\ufffd", "x"]
raw = [None, None, None, None, None, None, None, "<0xC3>", "<0xA9>", None]
lookup, size = build_map(decoded, raw)
check("the-family collapses", lookup[0] == lookup[1] == lookup[2] == 0)
check("space and newline collapse", lookup[3] == lookup[4] and lookup[3] != lookup[0])
check("cafe collapses", lookup[5] == lookup[6])
check("byte tokens keyed raw and distinct", lookup[7] != lookup[8])
check("compressed size", size == 6 and max(lookup) + 1 == 6)
check("first-seen ids", lookup == [0, 0, 0, 1, 1, 2, 2, 3, 4, 5][: len(lookup)] or lookup[:7] == [0, 0, 0, 1, 1, 2, 2])

print()
print("---- prime layout ----")


class Cfg:
    def __init__(self, **kw):
        self.model_type = "deepseek_v41_text"
        self.vocab_size = 129280
        self.hidden_size = 5120
        self.num_hidden_layers = 40
        self.hc_mult = 4
        self.rms_norm_eps = 1e-20
        self.engram_layer_ids = [1, 14]
        self.engram_num_embeddings = [384006168, 384016682]
        self.engram_max_ngram_size = 4
        self.engram_vocab_size = 16000000
        self.engram_n_heads = 8
        self.engram_head_dim = 256
        self.engram_pad_token_id = 2
        self.engram_compressed_vocab_size = 99092
        self.__dict__.update(kw)


official = EngramLayout.from_config(Cfg())
check("24 hash cols", official.n_hash_cols == 24)
check("span layer1 exact", official.bucket_span(0) == 384006168)
check("span layer14 exact", official.bucket_span(1) == 384016682)
flat0 = official.flat_primes(0)
flat1 = official.flat_primes(1)
check("primes strictly increasing and disjoint", list(flat0 + flat1) == sorted(set(flat0 + flat1)))
check("first prime", flat0[0] == 16000057)
off = official.bucket_offsets()
check("offsets shape", off.shape == (2, 24) and off[0][0] == 0 and off[0][1] == flat0[0])
check("offsets cumsum", int(off[0][-1]) + flat0[-1] == 384006168)
check("prime_array shape", official.prime_array().shape == (2, 3, 8))

mult = compute_mult((1, 14), 4, 99092)
check("multipliers odd", bool((mult % 2 == 1).all()))
check("multiplier row 0", mult[0].tolist() == [76632096046245, 4839876093313, 35959672319349, 73987337458391])
check("multiplier row 1", mult[1].tolist() == [67716810739261, 51510806800915, 30921347202721, 82619226485591])
check("no overflow bound", int(mult.max()) * 99091 < np.iinfo(np.int64).max)

for name, kw in [
    ("span wider than table", dict(engram_num_embeddings=[10, 10])),
    ("ngram size 1", dict(engram_max_ngram_size=1)),
    ("head_dim not block multiple", dict(engram_head_dim=100)),
    ("row count cardinality", dict(engram_num_embeddings=[384006168])),
    ("unsorted layer ids", dict(engram_layer_ids=[14, 1])),
    ("layer id out of range", dict(engram_layer_ids=[1, 400])),
    ("zero heads", dict(engram_n_heads=0)),
]:
    try:
        EngramLayout.from_config(Cfg(**kw))
        check("rejects " + name, False)
    except ValueError:
        check("rejects " + name, True)

check("disabled engram returns None", EngramLayout.from_config(Cfg(engram_layer_ids=[])) is None)
check("validate returns layout", validate_engram_config(Cfg()) is not None)
try:
    validate_engram_config(Cfg(engram_pad_token_id=999999))
    check("rejects pad id out of vocab", False)
except ValueError:
    check("rejects pad id out of vocab", True)

print()
print("fails:", fails)
