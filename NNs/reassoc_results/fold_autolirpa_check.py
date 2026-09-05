"""Decisive check with the REAL verifier (auto_LiRPA), not my numpy model.

Two exactly-equivalent torch nets:
  unfolded:  Linear A -> Linear B -> ReLU -> Linear C
  folded:    Linear (BA)          -> ReLU -> Linear C
A,B mixed-sign so |BA| < |B||A| (sign cancellation). Bound the SAME output radius
with method IBP, CROWN, CROWN-Optimized. Expect: IBP differs (folded tighter),
CROWN / CROWN-Optimized IDENTICAL (back-substitution composes B*A exactly).

Also: relu-decomposition equivalence relu(x) = x + relu(-x) -- a structure-CHANGING
exact identity (advisor's point). Show CROWN bound is identical (relaxation-neutral),
i.e. adding/re-signing ReLUs this way does not tighten.

Run in the abcrown venv: alpha-beta-CROWN/.venv/bin/python .../fold_autolirpa_check.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "alpha-beta-CROWN", "complete_verifier"))
import numpy as np, torch, torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
torch.manual_seed(0)

A = torch.tensor([[1.,-1.],[1.,1.]])
B = torch.tensor([[1.,1.],[-1.,1.]])
BA = B @ A
C = torch.randn(2,2)

def seq(*mats):
    m=[]
    for i,W in enumerate(mats):
        lin=nn.Linear(W.shape[1],W.shape[0],bias=False); lin.weight.data=W.clone()
        m.append(lin)
        if i==len(mats)-2:  # ReLU before the last map (the output head C)
            m.append(nn.ReLU())
    return nn.Sequential(*m)

unfolded = seq(A,B,C)          # A -> B -> ReLU -> C
folded   = seq(BA,C)           # BA -> ReLU -> C

x0 = torch.zeros(1,2)
ptb = PerturbationLpNorm(norm=np.inf, eps=1.0)
def bounds(net, method):
    bm = BoundedModule(net, x0)
    lb,ub = bm.compute_bounds(x=(BoundedTensor(x0,ptb),), method=method)
    return lb.detach().numpy().ravel(), ub.detach().numpy().ravel()

# sanity: identical function
xs=torch.randn(50,2)
assert torch.allclose(unfolded(xs), folded(xs), atol=1e-5), "not equivalent!"
print("nets function-identical on 50 samples: OK\n")

for method in ["IBP","CROWN","CROWN-Optimized"]:
    lu,uu = bounds(unfolded,method); lf,uf = bounds(folded,method)
    ru,rf = (uu-lu)/2, (uf-lf)/2
    same = np.allclose(ru,rf,atol=1e-5)
    print(f"{method:16s} unfolded out-radius={ru}  folded={rf}  identical={same}")

print("\n--- relu(x)=x+relu(-x): structure-changing exact identity ---")
class ReluStd(nn.Module):
    def forward(self,x): return torch.relu(x)
class ReluDecomp(nn.Module):
    # x + relu(-x) == relu(x); implemented so auto_LiRPA sees a DIFFERENT ReLU
    def forward(self,x): return x + torch.relu(-x)
def head(relumod):
    lin=nn.Linear(2,2,bias=False); lin.weight.data=BA.clone()
    out=nn.Linear(2,2,bias=False); out.weight.data=C.clone()
    return nn.Sequential(lin, relumod, out)
for name,mod in [("relu(x)",ReluStd()),("x+relu(-x)",ReluDecomp())]:
    lb,ub=bounds(head(mod),"CROWN")
    print(f"  {name:12s} CROWN out-radius={(ub-lb)/2}")
