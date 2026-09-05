import sys, os, torch, torch.nn as nn
sys.path.insert(0, os.path.join("alpha-beta-CROWN","complete_verifier"))
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
torch.set_num_threads(2); torch.manual_seed(0)

n=h=m=6; r=1.0
W1=torch.randn(h,n); W2=torch.randn(m,h)
# residual block: out = relu( W2 relu(W1 s) + Ws s ).  Identity relu(u)=u+relu(-u) gives
#   = relu( (W2 W1 + Ws) s + W2 relu(-W1 s) ) = relu( L s + W2 relu(-W1 s) ),  L=W2W1+Ws.
# Case A: shortcut CANCELS the main linear part (Ws=-W2W1 => L=0). Case B: random Ws.
for label, Ws in [("cancel (Ws=-W2W1)", -(W2@W1)), ("random Ws", torch.randn(m,n))]:
    L = W2@W1 + Ws
    class Orig(nn.Module):
        def forward(self,s): return torch.relu(s@W2.t().mm(torch.eye(h)).t()*0 + (torch.relu(s@W1.t())@W2.t()) + s@Ws.t())
    class Rewr(nn.Module):
        def forward(self,s): return torch.relu(s@L.t() + torch.relu(s@(-W1).t())@W2.t())
    orig, rewr = Orig().eval(), Rewr().eval()
    s0=torch.zeros(1,n)
    d=(orig(torch.randn(5,n))-rewr(torch.randn(5,n))).abs()  # careful: different random; recompute properly
    xs=torch.randn(8,n); d=(orig(xs)-rewr(xs)).abs().max().item()
    def width(net):
        bm=BoundedModule(net, s0)
        bx=BoundedTensor(s0, PerturbationLpNorm(norm=float('inf'), eps=r))
        lb,ub=bm.compute_bounds(x=(bx,), method="IBP")
        return (ub-lb)[0]
    wo, wr = width(orig), width(rewr)
    print(f"[{label}]  equiv max|Δ|={d:.2e}  |L|_sum={L.abs().sum():.2f}")
    print(f"    IBP out width  orig mean={wo.mean():.3f} max={wo.max():.3f} | rewr mean={wr.mean():.3f} max={wr.max():.3f}"
          f"  -> {'REWRITE TIGHTER' if wr.mean()<wo.mean()-1e-4 else 'not tighter'}")
