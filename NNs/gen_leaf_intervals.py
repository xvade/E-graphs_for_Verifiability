#!/usr/bin/env python3
"""Generate the affine-leaf IBP intervals for --verif_cost. Each leaf k is
Linear(d,d) with weight w_{2k}, bias w_{2k+1} (npz order); its output interval over
the L-inf box [x0-eps, x0+eps] is exact IBP (affine over a box): center = W x0 + b,
radius = eps*|W|.1. Keyed by the leaf's SORTED weight-name set (lexical, to match
tensat's Vec<String> sort). Unit-checks each interval against brute-force sampling.

Usage: gen_leaf_intervals.py <wN.npz> <wbx.npz> <out.json>
"""
import numpy as np, json, sys

wN = np.load(sys.argv[1]); wbx = np.load(sys.argv[2]); out = sys.argv[3]
x0 = wbx["x0"].astype(np.float64); eps = float(wbx["eps"])
M = sum(1 for k in wN.files if k.startswith("w_")) // 2
rng = np.random.default_rng(0)
xs = x0 + rng.uniform(-eps, eps, size=(4000, x0.shape[0]))
intervals = {}
for k in range(M):
    W = wN[f"w_{2*k}"].astype(np.float64)      # (d,d)
    b = wN[f"w_{2*k+1}"].astype(np.float64)    # (d,)
    center = W @ x0 + b
    radius = eps * np.abs(W).sum(axis=1)
    lo, hi = center - radius, center + radius
    vals = xs @ W.T + b                        # brute-force over the box
    assert (vals.min(0) >= lo - 1e-4).all() and (vals.max(0) <= hi + 1e-4).all(), \
        f"leaf {k}: IBP interval not sound vs sampling"
    # exactness (affine IBP is exact): sampled range should ~fill [lo,hi] on the live dim
    key = ",".join(sorted([f"w_{2*k}", f"w_{2*k+1}"]))
    intervals[key] = {"lo": lo.astype(float).tolist(), "hi": hi.astype(float).tolist()}
json.dump(intervals, open(out, "w"))
print(f"wrote {len(intervals)} leaf intervals to {out}; keys e.g. {list(intervals)[:3]}")
