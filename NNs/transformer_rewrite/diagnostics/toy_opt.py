# Toy: per-head gauge optimised under (a) the l1 width-product surrogate, (b) the exact CROWN score-box inflation, then scored by CROWN.
import sys, torch, numpy as np
import os; sys.path.insert(0, os.path.join(os.environ.get("REPO", "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"), "alpha-beta-CROWN/complete_verifier"))
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
torch.set_default_dtype(torch.float64)
def run(seed, n=32, dh=8, eps=0.05, steps=400):
    torch.manual_seed(seed)
    Wq = torch.randn(dh, n) / n**0.5; Wk = torch.randn(dh, n) / n**0.5; bq = torch.randn(dh); bk = torch.randn(dh); xc = torch.randn(n)
    def parts(G):
        A = G.T @ Wq; B = torch.linalg.inv(G) @ Wk; cq = A @ xc + G.T @ bq; ck = B @ xc + torch.linalg.inv(G) @ bk
        aq = eps * A.abs().sum(1); ak = eps * B.abs().sum(1); Js = ck @ A + cq @ B
        D = (ak[:, None] * A + aq[:, None] * B).sum(0); Dp = (ak[:, None] * A - aq[:, None] * B).sum(0)
        prod = 2 * (aq * ak).sum(); infl = prod + eps * ((Js - D).abs().sum() + (Js + Dp).abs().sum()) - 2 * eps * Js.abs().sum()
        return prod, infl
    def crown_width(G):
        class Head(torch.nn.Module):
            def __init__(s):
                super().__init__(); s.q = torch.nn.Linear(n, dh); s.k = torch.nn.Linear(n, dh)
                with torch.no_grad(): s.q.weight.copy_(G.T @ Wq); s.q.bias.copy_(G.T @ bq); s.k.weight.copy_(torch.linalg.inv(G) @ Wk); s.k.bias.copy_(torch.linalg.inv(G) @ bk)
            def forward(s, x): return torch.matmul(s.q(x), s.k(x).transpose(-1, -2))
        lp = BoundedModule(Head().eval(), torch.zeros(1, 1, n), bound_opts={"sparse_intermediate_bounds": False})
        x0 = xc.view(1, 1, n); bx = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, x_L=x0 - eps, x_U=x0 + eps))
        lb, ub = lp.compute_bounds(x=(bx,), method="CROWN"); return (ub - lb).item()
    def opt(obj):
        P = torch.nn.Parameter(torch.zeros(dh, dh)); o = torch.optim.Adam([P], lr=0.03); best = (float("inf"), None)
        for _ in range(steps):
            G = torch.linalg.matrix_exp(P); prod, infl = parts(G); v = prod if obj == "prod" else infl
            if v.item() < best[0]: best = (v.item(), G.detach().clone())
            o.zero_grad(); (v + 1e-4 * ((G**2).sum() + (torch.linalg.inv(G)**2).sum())).backward(); o.step()
        return best[1]
    I = torch.eye(dh); Gp = opt("prod"); Gi = opt("infl")
    out = []
    for nm, G in (("identity", I), ("min product", Gp), ("min exact inflation", Gi)):
        prod, infl = parts(G); out.append((nm, prod.item(), infl.item(), crown_width(G)))
    return out
for seed in range(3):
    print(f"seed {seed}:")
    for nm, p, i, w in run(seed): print(f"   {nm:22s} product 2*aq.ak {p:8.4f}   formula inflation {i:8.4f}   CROWN score width {w:8.4f}")
