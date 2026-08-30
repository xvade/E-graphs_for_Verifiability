#!/usr/bin/env python3
"""Model-agnostic per-form measurement: numeric-gate each reconstructed tensat form
against the INPUT model's output (semantics preservation), certified upper bound on
component 0 (alpha-CROWN), unstable-ReLU count, tree depth. Report spread.
Usage: bound_forms.py <forms_subdir> <ref_onnx> <wbx_npz> [envelope_chain envelope_bal]
Run in the alpha-beta-CROWN venv on GPU."""
import sys, os, numpy as np, torch, onnx, onnx2pytorch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "NNs"))
sys.path.insert(0, os.path.join(REPO, "NNs", "reassoc_results"))
sys.path.insert(0, os.path.join(REPO, "alpha-beta-CROWN", "complete_verifier"))
import maxtree_bounds as mtb, structural_signature as ss
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

sub, ref_onnx, wbx = sys.argv[1], sys.argv[2], sys.argv[3]
env = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else None
d = np.load(os.path.join(REPO, wbx)); x0, eps = d["x0"], float(d["eps"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
x0t = torch.tensor(x0, dtype=torch.float32, device=dev).view(1, -1)
xs = (x0 + np.random.default_rng(3).uniform(-eps, eps, size=(64, x0.shape[0]))).astype(np.float32)

def load(p):
    return onnx2pytorch.ConvertModel(onnx.load(p), experimental=True).to(dev).eval()

# reference (input model) component-0 outputs
ref = load(os.path.join(REPO, ref_onnx))
with torch.no_grad():
    ref_out = ref(torch.tensor(xs, device=dev)).cpu().numpy()[:, 0]

def measure(p):
    m = load(p)
    with torch.no_grad():
        out = m(torch.tensor(xs, device=dev)).cpu().numpy()[:, 0]
    dmax = float(np.abs(out - ref_out).max())
    bm = BoundedModule(m, x0t, device=dev, bound_opts={'conv_mode': 'matrix'})
    lb, ub = bm.compute_bounds(x=(BoundedTensor(x0t, PerturbationLpNorm(norm=np.inf, eps=eps)),),
                               method='CROWN-Optimized')
    nr, nu = mtb.count_unstable_relus(bm)
    return dmax, float(ub.flatten()[0]), nr, nu

man = [l.split() for l in open(os.path.join(REPO, "NNs/reassoc_results", sub, "manifest.txt")) if "onnx" in l]
rows, bad = [], []
for parts in man:
    i, depth, onnx_path = parts[0], int(parts[1]), parts[-1]
    dmax, ub0, nr, nu = measure(os.path.join(REPO, onnx_path))
    (bad if dmax >= 1e-3 else rows).append((int(i), depth, ub0, nu, nr, dmax))
rows.sort(key=lambda r: r[1])
print("form depth cert_ub  unstable  numchk")
for i, depth, ub0, nu, nr, dmax in rows:
    print(f"{i:4d} {depth:5d} {ub0:8.4f}  {nu:2d}/{nr}   {dmax:.1e}")
ubs = [r[2] for r in rows]; depths = [r[1] for r in rows]
corr = float(np.corrcoef(depths, ubs)[0, 1]) if len(set(depths)) > 1 else float('nan')
print(f"\nnumeric-gate failures: {len(bad)} {[(b[0],f'{b[5]:.1e}') for b in bad[:5]]}")
if env: print(f"ENVELOPE chain={env[0]:.4f} balanced(input)={env[1]:.4f}")
print(f"tensat forms: ub range [{min(ubs):.4f}, {max(ubs):.4f}] (n={len(ubs)})")
ref_bound = env[1] if env else None
if ref_bound: print(f"best form {min(ubs):.4f} vs input {ref_bound:.4f} -> improvement {ref_bound-min(ubs):+.4f}")
print(f"depth vs cert_ub corr = {corr:+.3f}")
