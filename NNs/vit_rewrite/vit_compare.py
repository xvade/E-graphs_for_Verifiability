#!/usr/bin/env python
"""Paired per-instance comparison of two harness result files (results/*.npz): fraction of instances with a
tighter min-spec lower bound, mean/median/worst delta, flips (unverified->verified and the reverse), width.

  python vit_compare.py results/A.npz results/B.npz [--method CROWN]
"""
import sys, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--method", default="CROWN")
a = ap.parse_args()
A = np.load(a.a, allow_pickle=True); B = np.load(a.b, allow_pickle=True)
assert (A["ids"] == B["ids"]).all(), "instance sets differ"
la, lb = A[f"lb_{a.method}"], B[f"lb_{a.method}"]
ma, mb = la.min(1), lb.min(1); d = mb - ma
va, vb = (la > 0).all(1), (lb > 0).all(1)
print(f"# {a.method}: A={a.a.split('/')[-1]}  B={a.b.split('/')[-1]}  n={len(d)}")
print(f"  verified all-9: A {va.sum()} -> B {vb.sum()}   flips unver->ver {int((~va & vb).sum())}, ver->unver {int((va & ~vb).sum())}")
print(f"  min-spec lb: mean A {ma.mean():+.4f} -> B {mb.mean():+.4f}  (delta mean {d.mean():+.4f}, median {np.median(d):+.4f}, worst {d.min():+.4f}, best {d.max():+.4f})")
print(f"  paired: B tighter on {int((d > 0).sum())}/{len(d)} instances, equal {int((d == 0).sum())}, looser {int((d < 0).sum())}")
print(f"  all-spec lb: mean A {la.mean():+.4f} -> B {lb.mean():+.4f};  per-spec paired tighter {int((lb > la).sum())}/{la.size}, looser {int((lb < la).sum())}")
if f"ub_{a.method}" in A and not np.isnan(A[f"ub_{a.method}"]).all():
    wa = (A[f"ub_{a.method}"] - la).mean(1); wb = (B[f"ub_{a.method}"] - lb).mean(1)
    print(f"  width: mean A {wa.mean():.4f} -> B {wb.mean():.4f} ({100*(wb.mean()/wa.mean()-1):+.1f}%), narrower on {int((wb < wa).sum())}/{len(wa)}")
