#!/usr/bin/env python
"""Sup over the eps-BOX (not just the center) of |stock(x) - rewritten(x)| via onnxruntime, uniform samples per box.
Turns the fp32-storage caveat into a box statement for the certificate-transfer argument."""
import sys, os, re, numpy as np, onnxruntime as ort
REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; sys.path.insert(0, os.path.join(REPO, "NNs/vit_rewrite"))
from vit_bounds import parse_vnnlib, instance_files
BENCH = os.path.join(REPO, "vnncomp2023_benchmarks/benchmarks")
model, other, n_samp = sys.argv[1], sys.argv[2], int(sys.argv[3]); ids = [int(i) for i in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
so = ort.SessionOptions(); so.intra_op_num_threads = 2
s0 = ort.InferenceSession(os.path.join(BENCH, "vit/onnx", model + ".onnx"), so, providers=["CPUExecutionProvider"])
s1 = ort.InferenceSession(os.path.join(BENCH, other, "onnx", model + ".onnx"), so, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0); worst = []
for path in instance_files(model, "all"):
    iid = int(re.search(r"_(\d+)\.vnnlib$", path).group(1))
    if ids is not None and iid not in ids: continue
    xl, xu, _ = parse_vnnlib(path); xl = np.asarray(xl, np.float32).reshape(1, 3, 32, 32); xu = np.asarray(xu, np.float32).reshape(1, 3, 32, 32)
    u = rng.random((n_samp, 3, 32, 32), dtype=np.float32); X = xl + u * (xu - xl)
    X = np.concatenate([X, xl, xu, 0.5 * (xl + xu)], 0)   # include corners-ish + center
    d = 0.0
    for i in range(0, len(X), 64):
        a = s0.run(None, {s0.get_inputs()[0].name: X[i:i + 64]})[0]; b = s1.run(None, {s1.get_inputs()[0].name: X[i:i + 64]})[0]
        d = max(d, float(np.abs(a - b).max()))
    worst.append((d, iid))
worst.sort(reverse=True)
print(f"# {other} vs stock, {len(worst)} boxes x {n_samp}+3 points: sup|diff| = {worst[0][0]:.3e} (instance {worst[0][1]}), median per-box sup {np.median([w for w, _ in worst]):.3e}")
print("# worst 5:", ", ".join(f"{i}:{d:.2e}" for d, i in worst[:5]))
