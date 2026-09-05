#!/usr/bin/env python
"""Recover the exact input normalization / box construction used by the VNN-COMP'23 ViT vnnlibs, so that
eps-boxes for OTHER CIFAR images (disjoint from the benchmark) can be generated for gauge tuning.
Hypothesis: x = (pixel/255 - mean)/std, box = clip(pixel/255 +- 1/255, 0, 1) normalized."""
import sys, os, re, pickle, numpy as np
REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
sys.path.insert(0, os.path.join(REPO, "NNs/vit_rewrite")); from vit_bounds import parse_vnnlib, BENCH
CIF = os.path.join(REPO, "alpha-beta-CROWN/complete_verifier/datasets/cifar-10-batches-py")
tb = pickle.load(open(os.path.join(CIF, "test_batch"), "rb"), encoding="bytes")
X = tb[b"data"].reshape(-1, 3, 32, 32).astype(np.float64) / 255.0; Y = np.array(tb[b"labels"])
mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1); std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
for f in sorted(os.listdir(os.path.join(BENCH, "vnnlib")))[:6]:
    idx = int(re.search(r"_(\d+)\.vnnlib$", f).group(1)); l, u, lab = parse_vnnlib(os.path.join(BENCH, "vnnlib", f))
    l = l.reshape(3, 32, 32); u = u.reshape(3, 32, 32)
    c = (l + u) / 2; pix = X[idx]
    cn = (pix - mean) / std
    ln = (np.clip(pix - 1 / 255, 0, 1) - mean) / std; un = (np.clip(pix + 1 / 255, 0, 1) - mean) / std
    print(f"{f}: label vnnlib={lab} cifar={Y[idx]} | center err={np.abs(c - cn).max():.2e} | lb err={np.abs(l - ln).max():.2e} ub err={np.abs(u - un).max():.2e}"
          f" | unclipped lb err={np.abs(l - (pix - 1/255 - mean)/std).max():.2e}")
