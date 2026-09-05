#!/usr/bin/env python
"""Sampling diagnostic (not a bound): how input-dependent are the attention probabilities, and how many MLP ReLUs
change sign, over uniform samples inside the benchmark eps-boxes?  Compares models."""
import sys, os, re, numpy as np, torch
REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; sys.path.insert(0, os.path.join(REPO, "NNs/vit_rewrite"))
from vit_model import ViT; from vit_bounds import parse_vnnlib, instance_files, BENCH
model, n_box, n_samp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
net = ViT(os.path.join(BENCH, "onnx", model + ".onnx")).eval()
rec = {"A": [], "pre": []}
_sm, _relu = torch.softmax, torch.relu
def sm(x, dim=-1, **k): y = _sm(x, dim=dim, **k); rec["A"].append(y.detach()); return y
def relu(x): rec["pre"].append(x.detach()); return _relu(x)
torch.softmax, torch.relu = sm, relu
rng = np.random.default_rng(0); A_dev, A_ent, unst, tot = [], [], [], 0
files = instance_files(model, "all")[:n_box]
for f in files:
    xl, xu, _ = parse_vnnlib(f); xl = torch.tensor(xl).reshape(1, 3, 32, 32); xu = torch.tensor(xu).reshape(1, 3, 32, 32)
    u = torch.tensor(rng.random((n_samp, 3, 32, 32), dtype=np.float32)); X = xl + u * (xu - xl)
    rec["A"].clear(); rec["pre"].clear()
    with torch.no_grad(): net(X)
    for A in rec["A"]:   # (n_samp, H, T, T)
        A_dev.append((A.max(0).values - A.min(0).values).max(-1).values.mean().item())     # mean over rows of the largest prob range
        A_ent.append((-(A * (A + 1e-12).log()).sum(-1)).mean().item() / np.log(A.shape[-1]))  # normalized entropy (1 = uniform)
    for P in rec["pre"]:  # (n_samp, T, hidden)
        s = (P > 0).float().mean(0); unst.append(((s > 0) & (s < 1)).float().sum().item()); tot += s.numel()
T = rec["A"][0].shape[-1]
print(f"# {model}: {len(files)} boxes x {n_samp} samples; tokens={T}")
print(f"  attention probs: mean per-row max range across samples = {np.mean(A_dev):.4f} (uniform prob = {1/T:.3f}); normalized entropy = {np.mean(A_ent):.3f} (1.0 = uniform attention)")
print(f"  MLP ReLUs whose sign varies inside the box: {np.sum(unst)/len(files):.0f} of {tot/len(files):.0f} per box ({100*np.sum(unst)/tot:.1f}%)")
