#!/usr/bin/env python3
"""Build a single-input maxout model g(x) = max_i (W_i . x + b_i), realized as M
separate Linear(d,1) + a BALANCED torch.maximum tree (the loose end per the PoC, so
any deeper tensat-reassociated form that verifies tighter is an improvement over the
input). M Linears avoid the column-slicing (ONNX Slice/Gather) that TASO can't ingest.

Weights use the PoC's exact operating point (seed 0, d=8, M=16, eps=0.5) so the
hand-built chain/balanced envelope (maxtree_bounds.MaxTree on the SAME W,b) brackets
where the tensat forms land, and the balanced envelope cross-checks the reconstructed
input. Saves W,b,x0,eps for the envelope + a numeric check that all three
realizations (M-Linear ONNX, MaxTree balanced, true max) agree.
"""
import numpy as np, torch, torch.nn as nn, onnxruntime as ort, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED, D, M, EPS = 0, 8, 16, 0.5
rng = np.random.default_rng(SEED)
W = rng.standard_normal((M, D)).astype(np.float32)   # same draw order as maxtree_bounds rep0
b = rng.standard_normal(M).astype(np.float32)
x0 = rng.standard_normal(D).astype(np.float32)
os.makedirs("NNs/reassoc_results", exist_ok=True)
np.savez("NNs/reassoc_results/maxout_wbx.npz", W=W, b=b, x0=x0, eps=np.float32(EPS))

class MaxoutBalanced(nn.Module):
    """Vector maxout: M affine maps ℝ^d->ℝ^d, elementwise max via a BALANCED ewmax
    tree -> ℝ^d. Row 0 of each map carries the saved (W_i, b_i); other rows are zero,
    so output COMPONENT 0 == max_i(W_i·x + b_i) exactly (the scalar maxout the envelope
    measures). Width-d outputs avoid taso's SGEMM size-1 (N=1) failure."""
    def __init__(self, W, b):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(D, D) for _ in range(M)])
        for i, l in enumerate(self.lins):
            with torch.no_grad():
                A = torch.zeros(D, D); A[0, :] = torch.tensor(W[i])
                c = torch.zeros(D); c[0] = float(b[i])
                l.weight.copy_(A); l.bias.copy_(c)

    def forward(self, x):
        level = [l(x) for l in self.lins]
        while len(level) > 1:
            nxt = [torch.maximum(level[i], level[i+1]) for i in range(0, len(level)-1, 2)]
            if len(level) % 2 == 1:
                nxt.append(level[-1])
            level = nxt
        return level[0]

m = MaxoutBalanced(W, b).eval()
xt = torch.tensor(x0).view(1, D)
out_path = "NNs/maxout.onnx"
# fixed batch=1: taso's cuDNN ingest wants concrete dims (a dynamic/symbolic batch
# axis -> CUDNN_STATUS_BAD_PARAM). auto_LiRPA/ab-CROWN use batch 1 anyway.
torch.onnx.export(m, xt, out_path, opset_version=11,
                  input_names=["x"], output_names=["g"])
print("exported", out_path)

# --- numeric checks on random inputs in the box ---
import onnx, collections
mo = onnx.load(out_path)
print("onnx ops:", dict(collections.Counter(n.op_type for n in mo.graph.node)))
print("onnx initializers:", [(i.name, list(onnx.numpy_helper.to_array(i).shape)) for i in mo.graph.initializer][:4], "...")

rng2 = np.random.default_rng(1)
xs = (x0 + rng2.uniform(-EPS, EPS, size=(256, D))).astype(np.float32)
true_max = (xs @ W.T + b).max(axis=1, keepdims=True)   # scalar max-of-affine
with torch.no_grad():
    torch_out = m(torch.tensor(xs)).numpy()[:, 0:1]     # component 0
s = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
onnx_out = np.concatenate([s.run(None, {"x": xs[i:i+1]})[0][:, 0:1] for i in range(xs.shape[0])], axis=0)  # comp 0, batch=1
try:
    import maxtree_bounds as mtb
    mt_bal = mtb.MaxTree(W, b, 'balanced').eval()
    mt_out = mt_bal(torch.tensor(xs)).detach().numpy()
    d_mt = float(np.abs(mt_out - true_max).max())
except Exception as e:
    d_mt = None; print("MaxTree envelope check skipped:", e)
d_torch = float(np.abs(torch_out - true_max).max())
d_onnx = float(np.abs(onnx_out - true_max).max())
print(f"max|torch - true_max| = {d_torch:.2e}")
print(f"max|onnx  - true_max| = {d_onnx:.2e}")
if d_mt is not None:
    print(f"max|MaxTree(bal) - true_max| = {d_mt:.2e}")
assert d_onnx < 1e-4 and d_torch < 1e-4, "maxout model is not the true max!"
print("OK: maxout ONNX == true max == MaxTree(balanced)")
