"""GOAL DELIVERABLE: manually rewrite a plain-ReLU (no min/max) network into an
EQUIVALENT but MORE VERIFIABLE form, and measure the improvement with the project's
real verifier (auto_LiRPA).

The network (resnet-flavored: a residual skip, two ReLU stages, no min/max):

    x --A--> u --B--> h=(B(A x)) --(+ skip W_s x)--> ReLU --C--> ReLU --D--> out
             \_____________ fold site ____________/
                 (two input-dependent linear ops, NO nonlinearity between)

The REWRITE (a sound e-graph equality): fold  B(A x) -> (BA) x. Function-identical.

Why it helps a re-boxing verifier (IBP): the intermediate u=Ax is re-boxed, so the
pre-activation radius of the first ReLU is |B||A|r unfolded vs |BA|r folded, and
|BA| <= |B||A| (strict here: A,B are MIXED-SIGN, so BA cancels). We pick weights and
an input box so that folding pulls the first ReLU's pre-activation interval OFF zero
-> neurons flip UNSTABLE->STABLE (the crisp verifiability gain), and the certified
output radius shrinks.

Scope, reported honestly: under CROWN (back-substitution, no re-boxing) BOTH forms
already get the tight |BA| interval, so the fold is NEUTRAL there -- the rule is
useful for IBP/hybrid verifiers, not for pure CROWN. This is exactly what makes it a
verifier-CONDITIONAL rule, and it is the honest positive answer to the goal.

Run: alpha-beta-CROWN/.venv/bin/python NNs/reassoc_results/plain_relu_more_verifiable.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "alpha-beta-CROWN", "complete_verifier"))
import numpy as np, torch, torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# --- mixed-sign fold site: BA cancels hard (|BA| << |B||A|) ---
A  = torch.tensor([[1., -1.], [1., 1.]])
B  = torch.tensor([[1.,  1.], [-1., 1.]])
BA = B @ A                       # = [[2,0],[0,2]] ; |BA|@1=[2,2] vs |B||A|@1=[4,4]
Ws = torch.tensor([[0., 0.], [0., 0.]])   # zero skip on the fold-fed ReLU (keep the
                                          # flip clean); a live skip is added deeper
C  = torch.tensor([[0.7, -0.4], [0.3, 0.6]])
D  = torch.tensor([[1.0,  0.5], [-0.5, 1.0]])

# input box: center chosen so folded pre-act is comfortably >0 but unfolded straddles
c0 = torch.tensor([[1.5, 1.5]]); EPS = 1.0    # l_inf box radius 1

class Net(nn.Module):
    """Two forms selected by `fold`. Residual skip added at the SECOND stage so the
    net is genuinely residual, while the first ReLU is the one the fold sharpens."""
    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        def lin(W):
            m = nn.Linear(W.shape[1], W.shape[0], bias=False); m.weight.data = W.clone(); return m
        if fold:
            self.f1 = lin(BA)                 # (BA) x   -- one linear op
        else:
            self.a = lin(A); self.b = lin(B)  # B(A x)   -- two linear ops, re-boxed by IBP
        self.c = lin(C); self.d = lin(D)
    def forward(self, x):
        h = self.f1(x) if self.fold else self.b(self.a(x))
        z1 = torch.relu(h)                    # <- the ReLU the fold sharpens
        z2 = torch.relu(self.c(z1) + z1)      # residual skip (z1) around the C-block
        return self.d(z2)

unfolded, folded = Net(False), Net(True)

# (0) sound equivalence: identical function
xs = torch.randn(200, 2)
assert torch.allclose(unfolded(xs), folded(xs), atol=1e-5), "NOT equivalent!"
print("equivalence: max|f_unfolded - f_folded| = "
      f"{(unfolded(xs)-folded(xs)).abs().max().item():.2e}  (sound rewrite)\n")

ptb = PerturbationLpNorm(norm=np.inf, eps=EPS)
def certify(net, method):
    bm = BoundedModule(net, c0)
    lb, ub = bm.compute_bounds(x=(BoundedTensor(c0, ptb),), method=method)
    lb, ub = lb.detach().numpy().ravel(), ub.detach().numpy().ravel()
    return lb, ub

# (1) the mechanism: first-ReLU pre-activation interval, hand-computed IBP, both forms
def preact_ibp(fold):
    c, r = c0.numpy().ravel(), np.ones(2) * EPS
    if fold:
        M = BA.numpy(); lo, hi = M@c - np.abs(M)@r, M@c + np.abs(M)@r
    else:
        u_c, u_r = A.numpy()@c, np.abs(A.numpy())@r          # re-box u=Ax
        lo = B.numpy()@u_c - np.abs(B.numpy())@u_r
        hi = B.numpy()@u_c + np.abs(B.numpy())@u_r
    return lo, hi
for name, fold in [("unfolded B(Ax)", False), ("folded (BA)x", True)]:
    lo, hi = preact_ibp(fold)
    unstable = int(np.sum((lo < 0) & (hi > 0)))
    print(f"  IBP first-ReLU pre-act [{name:15s}] lo={lo}  hi={hi}  "
          f"unstable={unstable}  ({'STABLE' if unstable==0 else 'UNSTABLE'})")
print()

# (2) certified output radius under the real verifier, both methods
for method in ["IBP", "CROWN", "CROWN-Optimized"]:
    lu, uu = certify(unfolded, method); lf, uf = certify(folded, method)
    ru, rf = (uu - lu) / 2, (uf - lf) / 2
    tag = ("folded TIGHTER" if np.all(rf <= ru + 1e-6) and np.any(rf < ru - 1e-6)
           else "identical" if np.allclose(ru, rf, atol=1e-5) else "mixed")
    print(f"  {method:16s} out-radius unfolded={ru}  folded={rf}   -> {tag}")

print("\nRead: under IBP the fold flips the first ReLU unstable->stable and shrinks the")
print("certified output radius -> a MORE VERIFIABLE equivalent form. Under CROWN both")
print("forms are already tight (fold neutral): the rule is verifier-conditional.")
