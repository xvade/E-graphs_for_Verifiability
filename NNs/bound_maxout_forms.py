#!/usr/bin/env python3
"""Measure each tensat-reassociated maxout form: numeric-gate (component 0 == true
max), certified upper bound on component 0 (alpha-CROWN), unstable-ReLU count, and
tree depth. Report the spread vs the hand-built chain/balanced envelope. Run in the
alpha-beta-CROWN venv on the GPU."""
import sys, os, glob, numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "NNs"))
sys.path.insert(0, os.path.join(REPO, "NNs", "reassoc_results"))
sys.path.insert(0, os.path.join(REPO, "alpha-beta-CROWN", "complete_verifier"))
import maxtree_bounds as mtb
import structural_signature as ss
import onnx, onnx2pytorch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

d = np.load(os.path.join(REPO, "NNs/reassoc_results/maxout_wbx.npz"))
W, b, x0, eps = d["W"], d["b"], d["x0"], float(d["eps"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x0t = torch.tensor(x0, dtype=torch.float32, device=dev).view(1, -1)
rng = np.random.default_rng(3)
xs = (x0 + rng.uniform(-eps, eps, size=(64, W.shape[1]))).astype(np.float32)
true_max = (xs @ W.T + b).max(axis=1)

def load_pt(onnx_path):
    m = onnx2pytorch.ConvertModel(onnx.load(onnx_path), experimental=True)
    return m.to(dev).eval()

def measure(onnx_path):
    model = load_pt(onnx_path)
    with torch.no_grad():
        out = model(torch.tensor(xs, device=dev)).cpu().numpy()
    dmax = float(np.abs(out[:, 0] - true_max).max())          # component-0 numeric gate
    bm = BoundedModule(model, x0t, device=str(dev), bound_opts={'conv_mode': 'matrix'})
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    lb, ub = bm.compute_bounds(x=(BoundedTensor(x0t, ptb),), method='CROWN-Optimized')
    ub0 = float(ub.flatten()[0])
    nr, nu = mtb.count_unstable_relus(bm)
    return dmax, ub0, nr, nu

manifest = [l.split() for l in open(os.path.join(REPO, "NNs/reassoc_results/maxout_forms/manifest.txt")) if "onnx" in l]
rows, bad = [], []
for parts in manifest:
    i = parts[0]; onnx_path = parts[-1]
    depth = ss.analyze(os.path.join(REPO, f"tensat/tmp/maxout_out_diverse{i}.model"))["max_depth"]
    dmax, ub0, nr, nu = measure(os.path.join(REPO, onnx_path))
    if dmax >= 1e-3:
        bad.append((i, dmax)); continue          # numeric gate (bidir-burn rule)
    rows.append((int(i), depth, ub0, nu, nr, dmax))

rows.sort(key=lambda r: r[1])
print("form depth cert_ub  unstable  numchk")
for i, depth, ub0, nu, nr, dmax in rows:
    print(f"{i:4d} {depth:5d} {ub0:8.4f}  {nu:2d}/{nr}   {dmax:.1e}")
ubs = [r[2] for r in rows]
depths = [r[1] for r in rows]
import numpy as _np
corr = _np.corrcoef(depths, ubs)[0, 1] if len(set(depths)) > 1 else float('nan')
print(f"\nnumeric-gate failures: {len(bad)} {bad[:5]}")
print(f"ENVELOPE  chain(tight)=11.7978  balanced(input,loose)=12.0257")
print(f"tensat forms: ub range [{min(ubs):.4f}, {max(ubs):.4f}]  (n={len(ubs)})")
print(f"best (tightest) tensat form = {min(ubs):.4f}  vs balanced input {12.0257:.4f}  -> improvement {12.0257-min(ubs):+.4f}")
print(f"depth vs cert_ub correlation = {corr:+.3f}  (negative => deeper is tighter, as predicted)")
