#!/usr/bin/env python3
"""SEMANTIC LIFT of the VNN-COMP tll (Two-Level Lattice) net back to an EXPLICIT
max-of-min ewmax/ewmin lattice, using tll's REAL weights.

The tllBench ONNX is TLL-compiled to a sequential MatMul/Add/Relu MLP -- min/max are
gone as ops. But the architecture is recoverable from the named weights (verified
numerically to 5e-7, max-of-min):
  linearLayer [2,16]  -> 16 local affine functions z_i(x) = W_i.x + b_i
  selectionLayer [16,256] one-hot -> member[256]: which local fn each slot picks
  -> 16 groups x 16 members; output = max_g min_{k in group g} z_{member[g,k]}(x)

We rebuild that as an explicit lattice (vector trick: each local fn is a Linear(2,WIDTH)
with row 0 carrying W_i,b_i and component 0 == the scalar; WIDTH>=8 keeps every matmul's
N>=8 so taso's SGEMM cost-measurement -- which aborts on small N -- is never hit). The
ewmin/ewmax trees are then reassociable by pwl_rules_ac.txt, and BOTH levels (min-16 and
max-16) have reassociation freedom (unlike the G=2 synthetic lattice).

Usage: build_tll_lattice.py [tll_onnx] [out_prefix]
"""
import sys, os, numpy as np, torch, torch.nn as nn, onnx, onnxruntime as ort
from onnx import numpy_helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TLL = sys.argv[1] if len(sys.argv) > 1 else "NNs/candidate_models/exotic2023/tll/tllBench_N16_instance_1_0.onnx"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "NNs/reassoc_results/tll"
WIDTH, DIN, EPS = 8, 2, 1.0
X0 = np.zeros(DIN, dtype=np.float32)

# --- read tll weights ---
I = {i.name: numpy_helper.to_array(i) for i in onnx.load(TLL).graph.initializer}
def get(tag, kind):
    for k, v in I.items():
        if tag in k and (('MatMul' in k) == (kind == 'W')) and (('BiasAdd' in k) == (kind == 'b')):
            return v
    raise KeyError(tag + "/" + kind)
linW = get('linearLayer', 'W').astype(np.float32)      # [2,16]
linb = get('linearLayer', 'b').astype(np.float32)      # [16]
selW = get('selectionLayer', 'W')                      # [16,256] one-hot
member = selW.argmax(0)                                # [256]
groups = member.reshape(16, 16)                        # [G=16, K=16] local-fn indices
M, G, K = 16, 16, 16
W = linW.T.copy()                                      # [16,2] : row i = W_i
b = linb.copy()

def tree(level, op):
    while len(level) > 1:
        nxt = [op(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]

class TLLLattice(nn.Module):
    def __init__(self):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(DIN, WIDTH) for _ in range(M)])
        for i, l in enumerate(self.lins):
            with torch.no_grad():
                A = torch.zeros(WIDTH, DIN); A[0, :] = torch.tensor(W[i])
                c = torch.zeros(WIDTH); c[0] = float(b[i])
                l.weight.copy_(A); l.bias.copy_(c)

    def forward(self, x):
        z = [l(x) for l in self.lins]                                  # 16 local fns
        gmin = [tree([z[groups[g, k]] for k in range(K)], torch.minimum) for g in range(G)]
        return tree(gmin, torch.maximum)                               # max over 16 groups

m = TLLLattice().eval()
xt = torch.tensor(X0).view(1, DIN)
out_path = PREFIX + "_lattice.onnx"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.onnx.export(m, xt, out_path, opset_version=11, input_names=["x"], output_names=["g"])

# --- numeric gate vs original tll (component 0) ---
so = ort.SessionOptions(); so.log_severity_level = 3; so.intra_op_num_threads = 1
ref = ort.InferenceSession(TLL, so, providers=["CPUExecutionProvider"])
rn = ref.get_inputs()[0].name
rng = np.random.default_rng(1)
xs = (X0 + rng.uniform(-EPS, EPS, size=(256, DIN))).astype(np.float32)
r = np.concatenate([ref.run(None, {rn: xs[i:i+1]})[0].ravel() for i in range(256)])
with torch.no_grad():
    o = m(torch.tensor(xs)).numpy()[:, 0]
dmax = float(np.abs(o - r).max())
print(f"max|lifted[:,0] - tll_onnx| = {dmax:.3e}  (GATE {'PASS' if dmax < 1e-3 else 'FAIL'})")
assert dmax < 1e-3, "lift does not match tll!"

# --- sidecars for the pipeline (mirror build_lattice.py's outputs) ---
np.savez(PREFIX + "_wbx.npz", W=W, b=b, x0=X0, eps=np.float32(EPS), G=G, K=K, groups=groups)
print(f"OK: wrote {out_path} and {PREFIX}_wbx.npz  (M={M} G={G} K={K}, WIDTH={WIDTH}, eps={EPS})")
