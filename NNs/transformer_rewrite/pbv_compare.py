"""Paired stock-vs-gauged comparison of PBVerification result files (their res.json format).

    python pbv_compare.py results/pbv_s6_origin_stock.json results/pbv_s6_origin_gauged.json [more pairs...]

Instances are matched by (sample index, position); the same --seed gives the same sentences for both weights, which is
checked via the token strings.  Prints: n, mean radius stock -> gauged (ratio of means, as in the AAAI-26 table), per-instance
larger / smaller / equal counts (their "#Win" convention counts strict wins only), and the mean per-instance ratio.
"""
import json, sys
import numpy as np

def load(p):
    d = json.load(open(p)); out = {}
    for si, ex in enumerate(d["examples"]):
        for b in ex["bounds"]: out[(si, b["position"])] = (b["eps"], " ".join(ex["tokens"]))
    return out

args = sys.argv[1:]
assert len(args) % 2 == 0, "give stock/gauged pairs"
for ps, pg in zip(args[::2], args[1::2]):
    S, G = load(ps), load(pg); keys = sorted(set(S) & set(G))
    assert all(S[k][1] == G[k][1] for k in keys), "sentence mismatch between stock and gauged runs"
    s = np.array([S[k][0] for k in keys]); g = np.array([G[k][0] for k in keys]); d = g - s
    tag = ps.split("/")[-1].replace("_stock.json", "")
    print(f"{tag}: n={len(keys)} (of {len(S)}/{len(G)}); mean radius {s.mean():.5f} -> {g.mean():.5f} ({100*(g.mean()/s.mean()-1):+.1f}% ratio of means; "
          f"mean per-instance ratio {100*np.mean(g/np.maximum(s,1e-12)-1):+.1f}%); larger on {(d>0).sum()}, smaller on {(d<0).sum()}, equal {(d==0).sum()}; "
          f"median rel {100*np.median(g/np.maximum(s,1e-12)-1):+.1f}%; min/max rel {100*(g/np.maximum(s,1e-12)-1).min():+.1f}% / {100*(g/np.maximum(s,1e-12)-1).max():+.1f}%")
