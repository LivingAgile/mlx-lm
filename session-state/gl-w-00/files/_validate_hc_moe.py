"""Throwaway numpy-backed validation of the new HC/MoE numeric logic."""
import ast, io, math, types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

mx = types.SimpleNamespace()
for name in (
    "arange zeros zeros_like ones abs minimum maximum max min sum mean where sign "
    "clip exp sqrt power logaddexp sort argpartition take_along_axis take repeat pad "
    "concatenate stack einsum broadcast_to bitwise_and right_shift expand_dims"
).split():
    setattr(mx, name, getattr(np, name))
mx.float32 = np.float32
mx.uint32 = np.uint32
mx.uint8 = np.uint8
mx.int32 = np.int32
mx.bfloat16 = np.float32
mx.array = lambda x, dtype=None: np.array(x, dtype=dtype)
mx.rsqrt = lambda x: 1.0 / np.sqrt(x)
mx.sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


mx.softmax = _softmax


def _from_fp8(codes, dtype=np.float32):
    c = np.asarray(codes).astype(np.uint8)
    sign = np.where((c >> 7) & 1, -1.0, 1.0)
    exp = ((c >> 3) & 0x0F).astype(np.float32)
    man = (c & 0x07).astype(np.float32)
    sub = man / 8.0 * (2.0 ** -6)
    nor = (1.0 + man / 8.0) * (2.0 ** (exp - 7.0))
    return (sign * np.where(exp == 0, sub, nor)).astype(np.float32)


mx.from_fp8 = _from_fp8


class _Module:
    def __init__(self):
        pass


nn = types.SimpleNamespace(Module=_Module, silu=lambda x: x * mx.sigmoid(x))

SRC = io.open("mlx_lm/models/deepseek_v41.py", encoding="utf-8").read()
tree = ast.parse(SRC)
WANTED = {
    "FP4_E2M1_TABLE", "FP4_WEIGHT_BLOCK_SIZE", "FP8_WEIGHT_BLOCK_SIZE",
    "GATE_TEMP", "ROUTING_NORM_EPS", "SUPPORTED_SCORING_FUNCS", "NOAUX_TC",
    "decode_e8m0_scale", "dequantize_fp8_block", "unpack_fp4_e2m1",
    "decode_fp4_e2m1_codes", "dequantize_fp4_block",
    "hc_split_sinkhorn", "expand_hyper_connection_stream",
    "make_identity_pre_mix", "hc_pre", "hc_post",
    "DeepseekV41HyperConnections",
    "routing_scores", "deterministic_topk_indices", "noaux_tc_route",
    "routed_expert_partition", "validate_moe_routing_config",
    "DeepseekV41PackedLinear", "DeepseekV41Expert", "DeepseekV41Gate",
    "DeepseekV41MoE",
}
keep = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANTED:
        keep.append(node)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in WANTED:
                keep.append(node)
                break

ns = {
    "mx": mx, "nn": nn, "math": math, "dataclass": dataclass, "Optional": Optional,
    "List": List, "Dict": Dict, "Tuple": Tuple, "Any": Any, "Union": Union,
    "TextConfig": object, "__name__": "harness",
}
exec(compile(ast.Module(body=keep, type_ignores=[]), "deepseek_v41", "exec"), ns)
print("exec'd", len(keep), "top-level nodes")
globals().update(ns)

rng = np.random.default_rng(7)
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# ---- Sinkhorn ----
HC, ITERS, EPS = 4, 20, 1e-6
MIX = (2 + HC) * HC
mixes = rng.normal(size=(2, 3, MIX)).astype(np.float32)
scale = np.array([0.7, 1.3, 0.9], dtype=np.float32)
base = rng.normal(size=(MIX,)).astype(np.float32)
pre, post, comb = hc_split_sinkhorn(mixes, scale, base, HC, ITERS, EPS)
check("sinkhorn shapes", pre.shape == (2, 3, HC) and post.shape == (2, 3, HC) and comb.shape == (2, 3, HC, HC))
check("comb rows ~ 1", np.allclose(comb.sum(-1), 1.0, atol=2e-3))
check("comb cols ~ 1", np.allclose(comb.sum(-2), 1.0, atol=2e-3))
check("comb positive", bool((comb > 0).all()))
check("pre in (eps, 1+eps)", bool((pre > EPS).all() and (pre < 1.0 + EPS + 1e-6).all()))
check("post in (0, 2)", bool((post > 0).all() and (post < 2.0).all()))

z = np.zeros((1, 1, MIX), dtype=np.float32)
zb = np.zeros((MIX,), dtype=np.float32)
p0, q0, c0 = hc_split_sinkhorn(z, scale, zb, HC, ITERS, EPS)
check("zero mixes -> uniform comb", np.allclose(c0, 1.0 / HC, atol=1e-5))
check("zero mixes -> pre 0.5+eps", np.allclose(p0, 0.5 + EPS, atol=1e-6))
check("zero mixes -> post 1.0", np.allclose(q0, 1.0, atol=1e-6))

# segment layout: scale[0]=scale[1]=0 isolates base
base2 = np.arange(MIX, dtype=np.float32)
p1, q1, c1 = hc_split_sinkhorn(z, np.array([0.0, 0.0, 1.0], np.float32), base2, HC, 1, EPS)
check("pre segment = base[:hc]", np.allclose(p1[0, 0], 1 / (1 + np.exp(-base2[:HC])) + EPS, atol=1e-6))
check("post segment = 2*sigmoid(base[hc:2hc])", np.allclose(q1[0, 0], 2 / (1 + np.exp(-base2[HC:2 * HC])), atol=1e-6))
exp_comb = _softmax(base2[2 * HC:].reshape(HC, HC), -1) + EPS
exp_comb = exp_comb / (exp_comb.sum(0, keepdims=True) + EPS)
check("comb segment row-major", np.allclose(c1[0, 0], exp_comb, atol=1e-6))

for bad in [
    lambda: hc_split_sinkhorn(mixes[..., :-1], scale, base, HC, ITERS, EPS),
    lambda: hc_split_sinkhorn(mixes, scale[:2], base, HC, ITERS, EPS),
    lambda: hc_split_sinkhorn(mixes, scale, base[:-1], HC, ITERS, EPS),
    lambda: hc_split_sinkhorn(mixes, scale, base, 0, ITERS, EPS),
    lambda: hc_split_sinkhorn(mixes, scale, base, HC, 0, EPS),
    lambda: hc_split_sinkhorn(mixes, scale, base, HC, ITERS, 0.0),
]:
    try:
        bad()
        check("malformed sinkhorn rejected", False)
    except ValueError:
        check("malformed sinkhorn rejected", True)

# ---- residual stream ----
h = rng.normal(size=(2, 3, 5)).astype(np.float32)
stream = expand_hyper_connection_stream(h, HC)
check("expand shape", stream.shape == (2, 3, HC, 5))
check("expand copies identical", np.allclose(stream[:, :, 0], stream[:, :, 3]))
idm = make_identity_pre_mix(2, 3, HC)
check("identity pre_mix is one-hot", np.allclose(idm[..., 0], 1.0) and np.allclose(idm[..., 1:], 0.0))
check("identity collapse recovers embedding", np.allclose(hc_pre(stream, idm), h, atol=1e-6))

residual = rng.normal(size=(2, 3, HC, 5)).astype(np.float32)
sub = rng.normal(size=(2, 3, 5)).astype(np.float32)
pm = rng.normal(size=(2, 3, HC)).astype(np.float32)
cb = rng.normal(size=(2, 3, HC, HC)).astype(np.float32)
got = hc_post(sub, residual, pm, cb)
exp = np.einsum("bsk,bsd->bskd", pm, sub) + np.einsum("bsjk,bsjd->bskd", cb, residual)
check("hc_post mixes comb over source axis", np.allclose(got, exp, atol=1e-5))


class _Cfg:
    hidden_size = 8
    hc_mult = HC
    hc_sinkhorn_iters = ITERS
    hc_eps = EPS
    rms_norm_eps = 1e-20
    n_routed_experts = 8
    num_experts_per_tok = 3
    n_shared_experts = 1
    scoring_func = "sqrtsoftplus"
    topk_method = "noaux_tc"
    norm_topk_prob = True
    routed_scaling_factor = 1.5
    moe_intermediate_size = 64
    swiglu_limit = 10.0


hcm = DeepseekV41HyperConnections(_Cfg())
hcm.hc_attn_fn = rng.normal(size=hcm.hc_attn_fn.shape).astype(np.float32)
hcm.hc_attn_base = rng.normal(size=hcm.hc_attn_base.shape).astype(np.float32)
hcm.hc_attn_scale = np.array([0.5, 0.5, 0.5], dtype=np.float32)
hcm.hc_ffn_fn = rng.normal(size=hcm.hc_ffn_fn.shape).astype(np.float32)
hcm.hc_ffn_base = rng.normal(size=hcm.hc_ffn_base.shape).astype(np.float32)
hcm.hc_ffn_scale = np.array([0.5, 0.5, 0.5], dtype=np.float32)
x = rng.normal(size=(2, 3, HC, 8)).astype(np.float32)
a1 = hcm.attn_mixes(x)
a2 = hcm.attn_mixes(x * 4.0)
check("hc_mixes are rms-scale invariant", np.allclose(a1[0], a2[0], atol=1e-4) and np.allclose(a1[2], a2[2], atol=1e-4))
check("hc_mixes deterministic", np.allclose(hcm.attn_mixes(x)[1], a1[1]))

seen = {}


def _attn(v):
    seen["attn_in"] = v
    return v * 2.0


def _ffn(v):
    seen["ffn_in"] = v
    return v * 3.0


out, nxt = hcm.block_step(x, idm[:, :, :HC], _attn, _ffn)
check("block_step shape", out.shape == x.shape and nxt.shape == (2, 3, HC))
attn_pre, attn_post, attn_comb = hcm.attn_mixes(x)
mid = hc_post(_attn(hc_pre(x, idm)), x, attn_post, attn_comb)
ffn_pre, ffn_post, ffn_comb = hcm.ffn_mixes(mid)
exp_out = hc_post(_ffn(hc_pre(mid, attn_pre)), mid, ffn_post, ffn_comb)
check("block_step ordering: ffn collapses with attn's pre", np.allclose(out, exp_out, atol=1e-5))
check("block_step hands on ffn pre-mix", np.allclose(nxt, ffn_pre, atol=1e-6))

# ---- routing ----
log = rng.normal(size=(5, 8)).astype(np.float32)
check("sqrtsoftplus", np.allclose(routing_scores(log, "sqrtsoftplus"), np.sqrt(np.log1p(np.exp(log))), atol=1e-5))
check("sigmoid scoring", np.allclose(routing_scores(log, "sigmoid"), 1 / (1 + np.exp(-log)), atol=1e-6))
check("softmax scoring", np.allclose(routing_scores(log, "softmax").sum(-1), 1.0, atol=1e-6))
try:
    routing_scores(log, "relu")
    check("unknown scoring rejected", False)
except ValueError:
    check("unknown scoring rejected", True)

sc = np.array([[3.0, 1.0, 3.0, 2.0, 3.0]], dtype=np.float32)
idx = deterministic_topk_indices(sc, 3)
check("topk ties -> lowest index, descending", idx.tolist() == [[0, 2, 4]])
sc2 = np.array([[0.1, 0.9, 0.5, 0.4]], dtype=np.float32)
check("topk ordering", deterministic_topk_indices(sc2, 3).tolist() == [[1, 2, 3]])
check("topk k=n is a full permutation", sorted(deterministic_topk_indices(sc2, 4).tolist()[0]) == [0, 1, 2, 3])
for k in (0, 5):
    try:
        deterministic_topk_indices(sc2, k)
        check("topk range rejected", False)
    except ValueError:
        check("topk range rejected", True)

scores = np.array([[0.5, 0.4, 0.3, 0.2]], dtype=np.float32)
no_bias = np.zeros((4,), dtype=np.float32)
bias = np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float32)
w0, i0 = noaux_tc_route(scores, no_bias, 2, True, 1.5)
w1, i1 = noaux_tc_route(scores, bias, 2, True, 1.5)
check("bias changes selection", i0.tolist() == [[0, 1]] and i1.tolist() == [[3, 0]])
check("weights come from unbiased scores", np.allclose(w1, np.array([[0.2, 0.5]]) / 0.7 * 1.5, atol=1e-5))
check("norm_topk_prob sums to route scale", np.allclose(w1.sum(-1), 1.5, atol=1e-5))
w2, _ = noaux_tc_route(scores, no_bias, 2, False, 2.0)
check("unnormalized path just scales", np.allclose(w2, np.array([[1.0, 0.8]]), atol=1e-6))

# ---- partition ----
check("partition 384/1", routed_expert_partition(384, 1, 0) == (0, 384))
check("partition 384/4 rank 2", routed_expert_partition(384, 4, 2) == (192, 288))
cover = []
for r in range(8):
    s, e = routed_expert_partition(384, 8, r)
    cover.extend(range(s, e))
check("partition covers exactly once", cover == list(range(384)))
for args in [(384, 5, 0), (384, 7, 0), (384, 4, 4), (384, 0, 0), (0, 1, 0), (384, 2, -1)]:
    try:
        routed_expert_partition(*args)
        check(f"partition rejects {args}", False)
    except ValueError:
        check(f"partition rejects {args}", True)

# ---- MoE composition ----
def _fill_fp4(lin, r):
    lin.weight = r.integers(0, 256, size=lin.weight.shape, dtype=np.uint8)
    lin.scale = np.full(lin.scale.shape, 127, dtype=np.uint8)


def _fill_fp8(lin, r):
    lin.weight = r.integers(0, 120, size=lin.weight.shape, dtype=np.uint8)
    lin.scale = np.full(lin.scale.shape, 127, dtype=np.uint8)


class _MoeCfg(_Cfg):
    hidden_size = 32
    moe_intermediate_size = 64


moe = DeepseekV41MoE(_MoeCfg(), expert_quant="fp4", shared_expert_quant="fp8")
check("all experts local at world_size 1", moe.local_expert_ids == list(range(8)))
for e in moe.experts:
    for lin in (e.w1, e.w2, e.w3):
        _fill_fp4(lin, rng)
for lin in (moe.shared_experts.w1, moe.shared_experts.w2, moe.shared_experts.w3):
    _fill_fp8(lin, rng)
moe.gate.weight = rng.normal(size=moe.gate.weight.shape).astype(np.float32) * 0.3
moe.gate.bias = rng.normal(size=moe.gate.bias.shape).astype(np.float32)

xm = rng.normal(size=(2, 3, 32)).astype(np.float32)
y = moe(xm)
check("moe output shape", y.shape == xm.shape)
check("moe output finite", bool(np.isfinite(y).all()))

flat = xm.reshape(-1, 32)
ww, ii = moe.gate(flat)
exp_y = np.zeros_like(flat, dtype=np.float32)
for t in range(flat.shape[0]):
    for s in range(moe.topk):
        e = int(ii[t, s])
        exp_y[t] += moe.experts[e](flat[t:t + 1], np.array([[ww[t, s]]], np.float32))[0]
exp_y += moe.shared_experts(flat)
check("moe matches explicit per-token composition", np.allclose(y.reshape(-1, 32), exp_y, atol=1e-3))
check("experts stay packed after forward", all(e.w1.weight.dtype == np.uint8 and e.w1.weight.shape == (64, 16) for e in moe.experts))

calls = {"n": 0}
_real = ns["dequantize_fp4_block"]


def _counting(*a, **k):
    calls["n"] += 1
    return _real(*a, **k)


ns["dequantize_fp4_block"] = _counting
moe(xm[:1, :1])
ns["dequantize_fp4_block"] = _real
check("only routed experts are unpacked", calls["n"] <= 3 * moe.topk)

sharded = DeepseekV41MoE(_MoeCfg(), world_size=4, rank=1)
check("rank owns its slice only", sharded.local_expert_ids == [2, 3] and len(sharded.experts) == 2)
check("global lookup works", sharded.expert(3) is sharded.experts[1])
try:
    sharded.expert(0)
    check("foreign expert refused", False)
except KeyError:
    check("foreign expert refused", True)
try:
    sharded(xm)
    check("world_size>1 execution fails loud", False)
except NotImplementedError:
    check("world_size>1 execution fails loud", True)

print()
print("FAILURES:", fails if fails else "none")
