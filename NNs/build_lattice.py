#!/usr/bin/env python3
"""Single-input min-of-max lattice g(x) = min_g max_k (W_{g,k}.x + b_{g,k}) -- the
tll-shaped structure, and the model where the FULL 621-rule PWL corpus (bridges,
min-reassociation, cross min/max) can actually bite (unlike the pure-max maxout).

Built as M=G*K vector Linear(d,d) maps (row 0 carries W_i,b_i; component 0 == the
scalar lattice; width-d avoids taso's SGEMM N=1 failure), a BALANCED ewmax tree per
group (K leaves), then a BALANCED ewmin tree across the G group-maxes. Same W,b as the
maxout run (seed 0, M=16) so it's the same affine set rearranged as a lattice; the
MinMaxLattice envelope brackets the forms.
"""
import numpy as np, torch, torch.nn as nn, onnxruntime as ort, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reassoc_results"))

SEED, D, G, K, EPS = 0, 8, 2, 8, 0.5
M = G * K
rng = np.random.default_rng(SEED)
W = rng.standard_normal((M, D)).astype(np.float32)
b = rng.standard_normal(M).astype(np.float32)
x0 = rng.standard_normal(D).astype(np.float32)
os.makedirs("NNs/reassoc_results", exist_ok=True)
np.savez("NNs/reassoc_results/lattice_wbx.npz", W=W, b=b, x0=x0, eps=np.float32(EPS), G=G, K=K)

def tree(level, op):
    while len(level) > 1:
        nxt = [op(level[i], level[i+1]) for i in range(0, len(level)-1, 2)]
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]

class Lattice(nn.Module):
    def __init__(self, W, b):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(D, D) for _ in range(M)])
        for i, l in enumerate(self.lins):
            with torch.no_grad():
                A = torch.zeros(D, D); A[0, :] = torch.tensor(W[i])
                c = torch.zeros(D); c[0] = float(b[i])
                l.weight.copy_(A); l.bias.copy_(c)

    def forward(self, x):
        z = [l(x) for l in self.lins]
        group_max = [tree(z[g*K:(g+1)*K], torch.maximum) for g in range(G)]
        return tree(group_max, torch.minimum)

m = Lattice(W, b).eval()
xt = torch.tensor(x0).view(1, D)
out_path = "NNs/lattice.onnx"
torch.onnx.export(m, xt, out_path, opset_version=11, input_names=["x"], output_names=["g"])
import onnx, collections
mo = onnx.load(out_path)
print("onnx ops:", dict(collections.Counter(n.op_type for n in mo.graph.node)))

# numeric check component 0 == scalar min-of-max
rng2 = np.random.default_rng(1)
xs = (x0 + rng2.uniform(-EPS, EPS, size=(256, D))).astype(np.float32)
zt = (xs @ W.T + b).reshape(-1, G, K)
true_lat = zt.max(axis=2).min(axis=1)                       # min_g max_k
with torch.no_grad():
    torch_out = m(torch.tensor(xs)).numpy()[:, 0]
s = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
onnx_out = np.concatenate([s.run(None, {"x": xs[i:i+1]})[0][:, 0] for i in range(xs.shape[0])])
try:
    import maxtree_bounds as mtb
    mt = mtb.MinMaxLattice(W, b, G, K, 'balanced').eval()
    d_mt = float(np.abs(mt(torch.tensor(xs)).detach().numpy().ravel() - true_lat).max())
except Exception as e:
    d_mt = None; print("MinMaxLattice check skipped:", e)
print(f"max|onnx - true_lattice| = {float(np.abs(onnx_out - true_lat).max()):.2e}")
if d_mt is not None:
    print(f"max|MinMaxLattice(bal) - true_lattice| = {d_mt:.2e}")
assert float(np.abs(onnx_out - true_lat).max()) < 1e-4
print(f"OK: lattice ONNX == min-of-max (G={G},K={K})")
