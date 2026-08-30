#!/usr/bin/env python3
"""Backward-CROWN-lite per-ReLU sensitivity on a REFERENCE extracted form, keyed the
same way VerifCost keys nodes (by output weight-name set) so it injects as
--sensitivity_file. Each ewmax/ewmin node's ReLU sensitivity |lambda| = the node's
OUTPUT sensitivity, propagated top-down from the root through the max/min relaxation
slopes (alpha = pu/(pu-pl) on the pre-activation interval; the dominant branch keeps
the sensitivity, the off-path branch decays to ~0). This is the critical-path weight
the unweighted gap-area cost lacks.

Usage: gen_sensitivity.py <ref.model> <weight_names.json> <intervals.json> <out.json>
"""
import sys, json, numpy as np

model, wnjson, ivjson, out = sys.argv[1:5]
# taso op ints (as exported in the .model)
OP_INPUT, OP_WEIGHT, OP_RELU, OP_EW_ADD, OP_MATMUL, OP_EW_SUB, OP_EW_MAX, OP_EW_MIN = \
    0, 1, 8, 16, 18, 28, 33, 34

L = open(model).read().splitlines()
op, deps, order = {}, {}, []
for i in range(0, len(L) - 3, 4):
    g = L[i].strip(); op[g] = int(L[i + 1])
    # Input/Weight leaves carry a spurious "10:0"/"11:0" dep that reconstruct ignores.
    deps[g] = [] if int(L[i + 1]) in (OP_INPUT, OP_WEIGHT) \
        else [d.split(":")[0] for d in L[i + 2].split(",") if d.strip()]
    order.append(g)
wn_raw = json.load(open(wnjson))                       # guid(str) -> [weight names]
ivs = {tuple(sorted(k.split(","))): (np.array(v["lo"]), np.array(v["hi"]))
       for k, v in json.load(open(ivjson)).items()}

# weight_names per node, bottom-up (matches tensat: Weight leaf -> {name}; union else)
wnames = {}
for g in order:
    if op[g] == OP_WEIGHT:
        wnames[g] = set(wn_raw.get(g, []))
    elif op[g] == OP_INPUT:
        wnames[g] = set()
    else:
        s = set()
        for c in deps[g]:
            s |= wnames.get(c, set())
        wnames[g] = s

def key(g):
    return tuple(sorted(wnames[g]))

# intervals bottom-up: affine leaf (weight-name set is an injected key) -> injected;
# else interval arithmetic; matmul/input below a leaf are never reached.
iv = {}
def interval(g):
    if g in iv:
        return iv[g]
    k = key(g)
    if k in ivs:
        iv[g] = ivs[k]; return iv[g]
    o, d = op[g], deps[g]
    if o == OP_EW_ADD:
        (la, ua), (lb, ub) = interval(d[0]), interval(d[1]); r = (la + lb, ua + ub)
    elif o == OP_EW_SUB:
        (la, ua), (lb, ub) = interval(d[0]), interval(d[1]); r = (la - ub, ua - lb)
    elif o == OP_EW_MAX:
        (la, ua), (lb, ub) = interval(d[0]), interval(d[1]); r = (np.maximum(la, lb), np.maximum(ua, ub))
    elif o == OP_EW_MIN:
        (la, ua), (lb, ub) = interval(d[0]), interval(d[1]); r = (np.minimum(la, lb), np.minimum(ua, ub))
    elif o == OP_RELU:
        l, u = interval(d[0]); r = (np.maximum(l, 0), np.maximum(u, 0))
    else:
        r = None
    iv[g] = r; return r

for g in order:
    interval(g)

root = next(g for g in order if all(g not in deps[h] for h in order))
sens = {g: 0.0 for g in order}; sens[root] = 1.0
node_weight = {}   # key(g) -> ReLU sensitivity for ewmax/ewmin g

def alpha(pl, pu):
    # scalar upper-relaxation slope on the LIVE (component-0) dim of the pre-activation
    pl0, pu0 = float(pl[0]), float(pu[0])
    if pl0 >= 0: return 1.0
    if pu0 <= 0: return 0.0
    return pu0 / (pu0 - pl0)

for g in reversed(order):                              # root -> leaves
    s = sens[g]
    if s == 0.0 or op[g] not in (OP_EW_MAX, OP_EW_MIN, OP_EW_ADD, OP_EW_SUB):
        # non-relu nodes still pass sensitivity through to children (mag 1)
        for c in deps[g]:
            sens[c] += s
        continue
    u, v = deps[g][0], deps[g][1]
    if op[g] in (OP_EW_MAX, OP_EW_MIN):
        (lu, uu), (lv, uv) = interval(u), interval(v)
        pl, pu = (lv - uu, uv - lu) if op[g] == OP_EW_MAX else (lu - uv, uu - lv)
        a = alpha(pl, pu)                              # relu backward slope
        node_weight[key(g)] = max(node_weight.get(key(g), 0.0), abs(s))
        sens[u] += s * (1.0 - a); sens[v] += s * a
    else:                                              # ewadd/ewsub: pass through, no relu
        sens[u] += s; sens[v] += s

json.dump({",".join(k): w for k, w in node_weight.items()}, open(out, "w"))
mm = sorted(node_weight.values())
print(f"wrote {len(node_weight)} node sensitivities to {out}; "
      f"range [{mm[0]:.3f}, {mm[-1]:.3f}], root nodes ~1.0")
