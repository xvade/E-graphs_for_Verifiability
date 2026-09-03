"""PILOT: does collapsing CReLU pairs tighten a FULL CROWN bound on a REAL trained net?

Goal constraints: real (not hand-crafted) model, no min/max in the technique, improve
full CROWN-Optimized, replicate across trainings. The rewrite needs REDUNDANT ReLU
structure, which standard training destroys -- so the model must have it ARCHITECTURALLY.
CReLU (Shang et al. ICML 2016), CReLU(z)=[relu(z), relu(-z)], is the canonical published
activation whose complementary-pair redundancy is weight-INDEPENDENT (present in every
training) -- exactly what constraint 4 (replicate) wants.

Collapse (exact, pure ReLU algebra, NO min/max), per CReLU layer, using relu(-z)=relu(z)-z:
  W+ relu(z) + W- relu(-z) = (W+ + W-) relu(z) - W- z,  and z is linear in the prev layer
  -> half the unstable ReLUs + cascading (DenseNet-style) linear skips. auto_LiRPA bounds DAGs.

This pilot: 1 MLP, MNIST, few epochs; gate function-identity in float64; count BoundRelu
coords (baseline must be 2x collapsed = not shared); CROWN-Optimized margin lb on ~20 imgs
across an eps sweep. If delta ~0 -> diagnose before scaling to seeds x tasks.

Run: alpha-beta-CROWN/.venv/bin/python NNs/reassoc_results/crelu_pilot.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "alpha-beta-CROWN", "complete_verifier"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.bound_ops import BoundRelu

torch.manual_seed(0)
torch.set_num_threads(4)   # loaded login node -> cap, avoid thread thrash
DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                    "alpha-beta-CROWN", "complete_verifier", "datasets")
DIN, W1, W2, DOUT = 784, 64, 64, 10


class CReLUNet(nn.Module):
    """Baseline: the NATURAL torch CReLU net (cat([relu(z),relu(-z)]) -- already
    independently relaxed by CROWN; nothing artificially loose)."""
    def __init__(self):
        super().__init__()
        self.A1 = nn.Linear(DIN, W1)
        self.A2 = nn.Linear(2 * W1, W2)
        self.A3 = nn.Linear(2 * W2, DOUT)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z1 = self.A1(x); c1 = torch.cat([F.relu(z1), F.relu(-z1)], 1)
        z2 = self.A2(c1); c2 = torch.cat([F.relu(z2), F.relu(-z2)], 1)
        return self.A3(c2)


class CollapsedNet(nn.Module):
    """Exact equivalent with the CReLU pairs collapsed: W1+W2 relus (vs 2W1+2W2),
    plus linear skips. Weights derived from a trained CReLUNet in float64."""
    def __init__(self, base):
        super().__init__()
        d = torch.float64
        A1w = base.A1.weight.data.to(d); A1b = base.A1.bias.data.to(d)
        A2w = base.A2.weight.data.to(d); A2b = base.A2.bias.data.to(d)
        A3w = base.A3.weight.data.to(d); A3b = base.A3.bias.data.to(d)
        A2p, A2m = A2w[:, :W1], A2w[:, W1:]          # act on relu(z1), relu(-z1)
        A3p, A3m = A3w[:, :W2], A3w[:, W2:]
        M2 = A2p + A2m;  S2 = -A2m @ A1w;  b2p = A2b - A2m @ A1b     # z2 = M2 relu(z1) + S2 x + b2p
        M3 = A3p + A3m                                              # out = M3 relu(z2) - A3m z2 + A3b
        K1 = -A3m @ M2;  K0 = -A3m @ S2;  bout = A3b - A3m @ b2p    # substitute z2
        def lin(W, b=None):
            m = nn.Linear(W.shape[1], W.shape[0], bias=b is not None)
            m.weight.data = W.to(torch.float32)
            if b is not None: m.bias.data = b.to(torch.float32)
            return m
        self.z1 = lin(A1w, A1b)
        self.z2_h1 = lin(M2); self.z2_x = lin(S2, b2p)
        self.o_h2 = lin(M3); self.o_h1 = lin(K1); self.o_x = lin(K0, bout)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h1 = F.relu(self.z1(x))
        h2 = F.relu(self.z2_h1(h1) + self.z2_x(x))
        return self.o_h2(h2) + self.o_h1(h1) + self.o_x(x)


def train(net, epochs=1, n_train=12000):
    tr = datasets.MNIST(DATA, train=True, download=False, transform=transforms.ToTensor())
    # preload a subset into ONE tensor (avoids per-item DataLoader overhead on a loaded node)
    X = torch.stack([tr[i][0] for i in range(n_train)])
    Y = torch.tensor([tr[i][1] for i in range(n_train)])
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    net.train(); bs = 256
    for ep in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, bs):
            b = perm[i:i + bs]
            opt.zero_grad(); loss = F.cross_entropy(net(X[b]), Y[b]); loss.backward(); opt.step()
    net.eval()


def relu_coords(net, x0):
    bm = BoundedModule(net, x0, verbose=False)
    return sum(int(np.prod(n.output_shape[1:])) for n in bm.nodes() if isinstance(n, BoundRelu))


def margin_lb(net, x, y, eps, method):
    """Per-image lower bound on min_j (logit_y - logit_j), j!=y. >0 => verified."""
    bm = BoundedModule(net, x[:1], verbose=False)
    n = x.shape[0]
    C = -torch.eye(DOUT)[None].repeat(n, 1, 1)
    C[torch.arange(n), :, y] += 1.0                 # rows: y - j  (row y is 0)
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=(x - eps).clamp(0, 1), x_U=(x + eps).clamp(0, 1))
    lb, _ = bm.compute_bounds(x=(BoundedTensor(x, ptb),), C=C, method=method)
    lb = lb.clone(); lb[torch.arange(n), y] = float("inf")  # ignore the y-y row
    return lb.min(1).values.detach()


if __name__ == "__main__":
    print("training...", flush=True)
    base = CReLUNet(); train(base, epochs=1)
    print("trained.", flush=True)
    coll = CollapsedNet(base); coll.eval()

    # test accuracy + function-identity gate (float64 exactness)
    te = datasets.MNIST(DATA, train=False, download=False, transform=transforms.ToTensor())
    xt = torch.stack([te[i][0] for i in range(500)]); yt = torch.tensor([te[i][1] for i in range(500)])
    with torch.no_grad():
        pb = base(xt); pc = coll(xt)
        acc = (pb.argmax(1) == yt).float().mean().item()
        f32 = (pb - pc).abs().max().item()
        b64 = CReLUNet(); b64.load_state_dict(base.state_dict())
        bd = b64.double()(xt.double()); cd_diff = (bd - CollapsedNet(base).double()(xt.double())).abs().max().item()
    print(f"test acc={acc:.3f} | fn-identity max|dup-merged| f32={f32:.2e} (float64 collapse gate={cd_diff:.2e})")

    x0 = xt[:1]
    print(f"BoundRelu coords: baseline={relu_coords(base, x0)}  collapsed={relu_coords(coll, x0)} "
          f"(expect 2x: {2*(W1+W2)} vs {W1+W2})")

    correct = (pb.argmax(1) == yt)
    idx = torch.where(correct)[0][:12]
    xs, ys = xt[idx], yt[idx]
    for eps in [0.03, 0.05, 0.08]:
        for method in ["CROWN", "CROWN-Optimized"]:
            mb = margin_lb(base, xs, ys, eps, method)
            mc = margin_lb(coll, xs, ys, eps, method)
            d = (mc - mb)
            print(f"eps={eps:.2f} {method:16s} verified base={int((mb>0).sum())}/{len(idx)} "
                  f"coll={int((mc>0).sum())}/{len(idx)} | mean margin lb {mb.mean():+.4f}->{mc.mean():+.4f} "
                  f"| per-img delta mean={d.mean():+.4f} min={d.min():+.4f} (>=0 frac {(d>=-1e-4).float().mean():.2f})")
