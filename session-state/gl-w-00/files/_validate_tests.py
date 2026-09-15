"""Supplemental numpy check of the *new tests* expectations."""
import runpy

import numpy as np

ns = runpy.run_path("session-state/gl-w-00/files/_validate_hc_moe.py")
g = ns
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


routing_scores = g["routing_scores"]
noaux_tc_route = g["noaux_tc_route"]
routed_expert_partition = g["routed_expert_partition"]
validate_moe_routing_config = g["validate_moe_routing_config"]
DeepseekV41Gate = g["DeepseekV41Gate"]
DeepseekV41Expert = g["DeepseekV41Expert"]
DeepseekV41PackedLinear = g["DeepseekV41PackedLinear"]
DeepseekV41MoE = g["DeepseekV41MoE"]
_MoeCfg = g["_MoeCfg"]
_Cfg = g["_Cfg"]

print()
print("---- new-test expectations ----")

tails = routing_scores(np.array([[-200.0, 0.0, 200.0]], np.float32), "sqrtsoftplus")
check("sqrtsoftplus finite in both tails", bool(np.isfinite(tails).all()))
check("sqrtsoftplus large tail ~ sqrt(x)", abs(float(tails[0, 2]) - 200.0 ** 0.5) < 1e-2)
check("sqrtsoftplus is not normalized", not np.allclose(routing_scores(np.array([[0.1, 0.2, 0.3]], np.float32), "sqrtsoftplus").sum(-1), 1.0, atol=1e-2))

sc = np.array([[0.5, 0.4, 0.3, 0.2]], np.float32)
zero4 = np.zeros((4,), np.float32)
w, i = noaux_tc_route(sc, zero4, 3, True, 1.5)
check("norm_topk sums to route scale (k=3)", np.allclose(w.sum(-1), 1.5, atol=1e-5))
w, i = noaux_tc_route(sc, zero4, 3, False, 2.0)
check("unnormalized k=3 just scales", np.allclose(w, np.array([[1.0, 0.8, 0.6]], np.float32), atol=1e-6))
w, i = noaux_tc_route(np.array([[0.25, 0.1]], np.float32), np.zeros((2,), np.float32), 1, True, 2.0)
check("k=1 keeps the raw score", i.tolist() == [[0]] and np.allclose(w, [[0.5]], atol=1e-6))

gate = DeepseekV41Gate(_MoeCfg())
w, i = gate(np.ones((3, 32), np.float32))
check("identical logits -> lowest indices", i.tolist() == [[0, 1, 2]] * 3)
check("identical logits -> scale/topk each", np.allclose(w, np.full((3, 3), 1.5 / 3.0), atol=1e-5))
check("gate is replicated, full width", gate.weight.shape == (8, 32) and gate.n_routed_experts == 8)

vl = DeepseekV41Gate(_MoeCfg(), vision_enabled=True)
vl.bias_vl = np.concatenate([np.zeros((7,), np.float32), np.array([10.0], np.float32)])
_, ivl = vl(np.ones((2, 32), np.float32), np.array([False, True]))
check("vl bias only inside image spans", ivl.tolist()[0] == [0, 1, 2] and ivl.tolist()[1][0] == 7)
try:
    DeepseekV41Gate(_MoeCfg())(np.ones((2, 32), np.float32), np.array([False, True]))
    check("vl mask on text-only gate fails", False)
except ValueError:
    check("vl mask on text-only gate fails", True)

check("partition 384/2 rank1", routed_expert_partition(384, 2, 1) == (192, 384))
check("partition 384/8 rank0", routed_expert_partition(384, 8, 0) == (0, 48))
check("partition 128/4 rank3", routed_expert_partition(128, 4, 3) == (96, 128))
ok = True
for ws in (1, 2, 3, 4, 6, 8, 12, 16):
    cov = []
    for r in range(ws):
        s, e = routed_expert_partition(384, ws, r)
        cov.extend(range(s, e))
    ok = ok and cov == list(range(384))
check("all listed world sizes cover exactly once", ok)
for ws in (5, 7, 9, 10):
    try:
        routed_expert_partition(384, ws, 0)
        check(f"world size {ws} refused", False)
    except ValueError:
        check(f"world size {ws} refused", True)

lin = DeepseekV41PackedLinear(64, 32, "fp4")
check("fp4 packed shapes", lin.weight.shape == (32, 32) and lin.scale.shape == (32, 2))
lin8 = DeepseekV41PackedLinear(64, 48, "fp8")
check("fp8 tiled scale grid", lin8.weight.shape == (48, 64) and lin8.scale.shape == (2, 2))
dense = DeepseekV41PackedLinear(64, 32, None)
check("dense carries no scale", not hasattr(dense, "scale"))
for args in ((33, 32, "fp4"), (33, 32, "fp8"), (0, 32, "fp4"), (32, 0, "fp4"), (64, 32, "int4")):
    try:
        DeepseekV41PackedLinear(*args)
        check(f"packed rejects {args}", False)
    except ValueError:
        check(f"packed rejects {args}", True)


def silu(v):
    return v / (1.0 + np.exp(-v))


def build(limit, sign):
    e = DeepseekV41Expert(32, 64, None, limit)
    e.w1.weight = np.full((64, 32), sign * 10.0 / 32, np.float32)
    e.w3.weight = np.full((64, 32), -10.0 / 32, np.float32)
    e.w2.weight = np.full((32, 64), 1.0 / 64, np.float32)
    return e


xo = np.ones((1, 32), np.float32)
check("swiglu clamp on", np.allclose(build(1.0, 1.0)(xo), silu(1.0) * -1.0, atol=1e-3))
check("swiglu clamp off", np.allclose(build(0.0, 1.0)(xo), silu(10.0) * -10.0, atol=1e-2))
check("swiglu clamp off is wide", float(np.abs(build(0.0, 1.0)(xo)).max()) > 50.0)
gn = build(1.0, -1.0)(xo)
check("gate branch clamped from above only", np.allclose(gn, silu(-10.0) * -1.0, atol=1e-4))
check("gate branch stays tiny", float(np.abs(gn).max()) < 1e-2)

try:
    validate_moe_routing_config(_MoeCfg(), 8, 8)
    check("topk == n_routed allowed", True)
except ValueError:
    check("topk == n_routed allowed", False)
for args in ((8, 9), (0, 1)):
    try:
        validate_moe_routing_config(_MoeCfg(), *args)
        check(f"validate rejects {args}", False)
    except ValueError:
        check(f"validate rejects {args}", True)


def _cfg(**kw):
    c = _MoeCfg()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


for kw, needle in (
    ({"topk_method": "group_limited_greedy"}, "noaux_tc"),
    ({"scoring_func": "gelu"}, "sqrtsoftplus"),
    ({"n_shared_experts": 3}, "shared"),
):
    try:
        validate_moe_routing_config(_cfg(**kw), 8, 3)
        check(f"validate rejects {kw}", False)
    except ValueError as exc:
        check(f"validate rejects {kw} naming {needle}", needle in str(exc).lower())

moe = DeepseekV41MoE(_MoeCfg())
rng = np.random.default_rng(3)
for e in moe.experts:
    for l in (e.w1, e.w2, e.w3):
        l.weight = rng.integers(0, 256, size=l.weight.shape, dtype=np.uint8)
        l.scale = np.full(l.scale.shape, 127, np.uint8)
for l in (moe.shared_experts.w1, moe.shared_experts.w2, moe.shared_experts.w3):
    l.weight = rng.integers(0, 120, size=l.weight.shape, dtype=np.uint8)
    l.scale = np.full(l.scale.shape, 127, np.uint8)
moe.gate.weight = rng.normal(size=moe.gate.weight.shape).astype(np.float32) * 0.3
moe.gate.bias = rng.normal(size=moe.gate.bias.shape).astype(np.float32)

xt = rng.normal(size=(1, 1, 32)).astype(np.float32)
_, ids = moe.gate(xt.reshape(-1, 32))
routed = set(int(v) for v in ids[0])
check("one token routes to exactly topk distinct experts", len(routed) == 3 and len(routed) < 8)
calls = {"n": 0}
exec_ns = g["ns"]
real = exec_ns["dequantize_fp4_block"]


def counting(*a, **k):
    calls["n"] += 1
    return real(*a, **k)


exec_ns["dequantize_fp4_block"] = counting
moe(xt)
exec_ns["dequantize_fp4_block"] = real
print("   unpack calls:", calls["n"], "routed:", len(routed))
check("exactly 3 unpacks per routed expert, none otherwise", calls["n"] == 3 * len(routed))

zero_moe = DeepseekV41MoE(_MoeCfg())
for l in (zero_moe.shared_experts.w1, zero_moe.shared_experts.w2, zero_moe.shared_experts.w3):
    l.weight = rng.integers(0, 120, size=l.weight.shape, dtype=np.uint8)
    l.scale = np.full(l.scale.shape, 127, np.uint8)
zero_moe.gate.weight = moe.gate.weight
zero_moe.gate.bias = moe.gate.bias
xb = rng.normal(size=(2, 3, 32)).astype(np.float32)
check(
    "zero routed experts leave only the shared output",
    np.allclose(zero_moe(xb).reshape(-1, 32), zero_moe.shared_experts(xb.reshape(-1, 32)), atol=1e-4)
    and float(np.abs(zero_moe(xb)).max()) > 0.0,
)

print()
print("NEW-TEST FAILURES:", fails if fails else "none")
