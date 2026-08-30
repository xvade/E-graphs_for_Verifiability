#!/usr/bin/env python3
"""Hand-built chain/balanced envelope on the SAME W,b as the maxout model, using the
PoC's MaxTree + certified_ub. Two purposes: (1) the chain/balanced certified bounds
bracket where the tensat-reassociated forms should land; (2) the balanced bound must
~match the reconstructed-input bound -- a cross-validation gate on the whole
ingest->reconstruct->bound path. Run in the alpha-beta-CROWN venv (auto_LiRPA)."""
import numpy as np, torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "alpha-beta-CROWN", "complete_verifier"))
import maxtree_bounds as mtb

d = np.load("NNs/reassoc_results/maxout_wbx.npz")
W, b, x0, eps = d["W"], d["b"], d["x0"], float(d["eps"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x0t = torch.tensor(x0, dtype=torch.float32, device=dev).unsqueeze(0)
method = "CROWN-Optimized"
print(f"# envelope on maxout W,b (M={W.shape[0]}, d={W.shape[1]}, eps={eps}, {method}, {dev})")
for shape in ["chain", "balanced"]:
    m = mtb.MaxTree(W, b, shape).to(dev).eval()
    lb, ub, nr, nu = mtb.certified_ub(m, x0t, eps, method)
    print(f"{shape:9s}: certified_ub = {ub:.4f}   unstable_relus = {nu}/{nr}")
# dense-sample true max for reference
big = x0 + np.random.default_rng(2).uniform(-eps, eps, size=(50000, W.shape[1])).astype(np.float32)
print(f"true max over box (sampled): {float((big @ W.T + b).max(axis=1).max()):.4f}")
