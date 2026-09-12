# Numerical check of the derived CROWN score-box formula on a toy bilinear head: s = (Gq^T Wq x) . (G^-1 Wk x) over an l_inf box.
import sys, torch, numpy as np
import os; sys.path.insert(0, os.path.join(os.environ.get("REPO", "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"), "alpha-beta-CROWN/complete_verifier"))
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
n, dh = 16, 4
Wq = torch.randn(dh, n); Wk = torch.randn(dh, n); bq = torch.randn(dh); bk = torch.randn(dh)
class Head(torch.nn.Module):
    def __init__(s, G):
        super().__init__(); s.q = torch.nn.Linear(n, dh); s.k = torch.nn.Linear(n, dh)
        with torch.no_grad(): s.q.weight.copy_(G.T @ Wq); s.q.bias.copy_(G.T @ bq); s.k.weight.copy_(torch.linalg.inv(G) @ Wk); s.k.bias.copy_(torch.linalg.inv(G) @ bk)
    def forward(s, x):  # x (1,1,n) -> score (1,1,1): q (1,1,dh) @ k^T (1,dh,1)
        q = s.q(x); k = s.k(x); return torch.matmul(q, k.transpose(-1, -2))
def formula(G, xc, eps):
    A = G.T @ Wq; B = torch.linalg.inv(G) @ Wk                     # rows: grad of q'_c, k'_c w.r.t. x
    cq = A @ xc + G.T @ bq; ck = B @ xc + torch.linalg.inv(G) @ bk
    aq = eps * A.abs().sum(1); ak = eps * B.abs().sum(1)             # exact half-widths
    Js = ck @ A + cq @ B                                             # score Jacobian (n,)
    D = (ak[:, None] * A + aq[:, None] * B).sum(0); Dp = (ak[:, None] * A - aq[:, None] * B).sum(0)
    sc = cq @ ck
    l = sc - (aq * ak).sum() - eps * (Js - D).abs().sum(); u = sc + (aq * ak).sum() + eps * (Js + Dp).abs().sum()
    lin = 2 * eps * Js.abs().sum(); sig = torch.sign(Js)
    infl_lin = 2 * (aq * (ak - eps * (B @ sig))).sum()
    return l.item(), u.item(), (u - l).item(), lin.item(), infl_lin.item()
xc = torch.randn(1, 1, n); eps = 0.05
for trial in range(4):
    G = torch.eye(dh) if trial == 0 else torch.linalg.matrix_exp(0.5 * torch.randn(dh, dh))
    m = Head(G).eval(); lp = BoundedModule(m, torch.zeros(1, 1, n), bound_opts={"sparse_intermediate_bounds": False})
    bx = BoundedTensor(xc, PerturbationLpNorm(norm=np.inf, x_L=xc - eps, x_U=xc + eps))
    lb, ub = lp.compute_bounds(x=(bx,), method="CROWN")
    l, u, w, lin, il = formula(G, xc[0, 0], eps)
    print(f"trial {trial}: CROWN [{lb.item():+.6f}, {ub.item():+.6f}] width {ub.item()-lb.item():.6f} | formula [{l:+.6f}, {u:+.6f}] width {w:.6f} | first-order width {lin:.6f}, inflation exact {w-lin:.6f} linearised {il:.6f}")
