"""REPLICATION: CReLU-pair collapse tightens full CROWN-Optimized across INDEPENDENT
trainings (seeds x tasks) -- proving the improvement is ARCHITECTURAL, not from
coincidental weight values (goal constraint 4).

Same rewrite as crelu_pilot.py (exact, no min/max: W+ relu(z)+W- relu(-z) =
(W++W-) relu(z) - W- z, halving unstable ReLUs + linear skips). Here: 3 seeds x
{MNIST, FashionMNIST}, ~100 correctly-classified test imgs each, fixed eps per task,
report per-image paired CROWN-Optimized margin-lb deltas (the clean "improves" evidence),
verified-accuracy change, and per-layer coefficient-cancellation stats (mechanism trace).

Run: alpha-beta-CROWN/.venv/bin/python NNs/reassoc_results/crelu_replicate.py
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "alpha-beta-CROWN", "complete_verifier"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.bound_ops import BoundRelu

torch.set_num_threads(4)
DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                    "alpha-beta-CROWN", "complete_verifier", "datasets")
DIN, W1, W2, DOUT = 784, 64, 64, 10


class CReLUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.A1 = nn.Linear(DIN, W1); self.A2 = nn.Linear(2 * W1, W2); self.A3 = nn.Linear(2 * W2, DOUT)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z1 = self.A1(x); c1 = torch.cat([F.relu(z1), F.relu(-z1)], 1)
        z2 = self.A2(c1); c2 = torch.cat([F.relu(z2), F.relu(-z2)], 1)
        return self.A3(c2)


class CollapsedNet(nn.Module):
    def __init__(self, base):
        super().__init__()
        d = torch.float64
        A1w, A1b = base.A1.weight.data.to(d), base.A1.bias.data.to(d)
        A2w, A2b = base.A2.weight.data.to(d), base.A2.bias.data.to(d)
        A3w, A3b = base.A3.weight.data.to(d), base.A3.bias.data.to(d)
        A2p, A2m = A2w[:, :W1], A2w[:, W1:]; A3p, A3m = A3w[:, :W2], A3w[:, W2:]
        M2 = A2p + A2m; S2 = -A2m @ A1w; b2p = A2b - A2m @ A1b
        M3 = A3p + A3m; K1 = -A3m @ M2; K0 = -A3m @ S2; bout = A3b - A3m @ b2p
        self.cancel = (  # mean coord-wise |W++W-|/(|W+|+|W-|) over the two CReLU->linear maps
            ((A2p + A2m).abs() / (A2p.abs() + A2m.abs() + 1e-12)).mean().item(),
            ((A3p + A3m).abs() / (A3p.abs() + A3m.abs() + 1e-12)).mean().item())
        def lin(W, b=None):
            m = nn.Linear(W.shape[1], W.shape[0], bias=b is not None); m.weight.data = W.to(torch.float32)
            if b is not None: m.bias.data = b.to(torch.float32)
            return m
        self.z1 = lin(A1w, A1b); self.z2_h1 = lin(M2); self.z2_x = lin(S2, b2p)
        self.o_h2 = lin(M3); self.o_h1 = lin(K1); self.o_x = lin(K0, bout)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h1 = F.relu(self.z1(x)); h2 = F.relu(self.z2_h1(h1) + self.z2_x(x))
        return self.o_h2(h2) + self.o_h1(h1) + self.o_x(x)


def load(task):
    tf = transforms.ToTensor()
    if task == "MNIST":
        tr = datasets.MNIST(DATA, True, download=False, transform=tf); te = datasets.MNIST(DATA, False, download=False, transform=tf)
    elif task == "FashionMNIST":
        tr = datasets.FashionMNIST(DATA, True, download=True, transform=tf); te = datasets.FashionMNIST(DATA, False, download=True, transform=tf)
    elif task == "KMNIST":
        tr = datasets.KMNIST(DATA, True, download=True, transform=tf); te = datasets.KMNIST(DATA, False, download=True, transform=tf)
    return tr, te


def train(net, tr, seed, epochs=3, n=30000):
    g = torch.Generator().manual_seed(seed)
    X = torch.stack([tr[i][0] for i in range(n)]); Y = torch.tensor([tr[i][1] for i in range(n)])
    opt = torch.optim.Adam(net.parameters(), 1e-3); net.train(); bs = 256
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            bi = perm[i:i + bs]
            opt.zero_grad(); F.cross_entropy(net(X[bi]), Y[bi]).backward(); opt.step()
    net.eval()


def relu_coords(net, x0):
    bm = BoundedModule(net, x0, verbose=False)
    return sum(int(np.prod(n.output_shape[1:])) for n in bm.nodes() if isinstance(n, BoundRelu))


def margin_lb(net, x, y, eps, method):
    bm = BoundedModule(net, x[:1], verbose=False); n = x.shape[0]
    C = -torch.eye(DOUT)[None].repeat(n, 1, 1); C[torch.arange(n), :, y] += 1.0
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=(x - eps).clamp(0, 1), x_U=(x + eps).clamp(0, 1))
    lb, _ = bm.compute_bounds(x=(BoundedTensor(x, ptb),), C=C, method=method)
    lb = lb.clone(); lb[torch.arange(n), y] = float("inf")
    return lb.min(1).values.detach()


TASKS = [("MNIST", 0.05), ("FashionMNIST", 0.03)]
SEEDS = [0, 1, 2]
NIMG = 100

if __name__ == "__main__":
    rows = []
    for task, eps in TASKS:
        try:
            tr, te = load(task)
        except Exception as e:
            print(f"{task}: load failed ({e}); falling back to KMNIST"); task = "KMNIST"; tr, te = load("KMNIST")
        xt = torch.stack([te[i][0] for i in range(1500)]); yt = torch.tensor([te[i][1] for i in range(1500)])
        for seed in SEEDS:
            torch.manual_seed(seed)
            base = CReLUNet(); train(base, tr, seed); coll = CollapsedNet(base); coll.eval()
            with torch.no_grad():
                pb = base(xt); acc = (pb.argmax(1) == yt).float().mean().item()
                # float64 exactness gate on COPIES (.double() mutates in place)
                gate = (copy.deepcopy(base).double()(xt.double())
                        - copy.deepcopy(coll).double()(xt.double())).abs().max().item()
            idx = torch.where(pb.argmax(1) == yt)[0][:NIMG]; xs, ys = xt[idx], yt[idx]
            nb, nc = relu_coords(base, xs[:1]), relu_coords(coll, xs[:1])
            mb = margin_lb(base, xs, ys, eps, "CROWN-Optimized")
            mc = margin_lb(coll, xs, ys, eps, "CROWN-Optimized")
            d = mc - mb
            r = dict(task=task, seed=seed, acc=acc, gate=gate, nb=nb, nc=nc, eps=eps,
                     vb=int((mb > 0).sum()), vc=int((mc > 0).sum()), n=len(idx),
                     mmb=mb.mean().item(), mmc=mc.mean().item(),
                     dmean=d.mean().item(), dmin=d.min().item(), dmax=d.max().item(),
                     frac_impr=(d > 1e-4).float().mean().item(), cancel=coll.cancel)
            rows.append(r)
            print(f"[{task} seed{seed}] acc={acc:.3f} gate={gate:.1e} relu {nb}->{nc} | "
                  f"eps={eps} verified {r['vb']}->{r['vc']}/{r['n']} | margin lb {r['mmb']:+.3f}->{r['mmc']:+.3f} | "
                  f"per-img delta mean={r['dmean']:+.4f} min={r['dmin']:+.4f} improved={r['frac_impr']:.2f} | "
                  f"cancel={r['cancel'][0]:.2f},{r['cancel'][1]:.2f}", flush=True)

    print("\n===== SUMMARY (CROWN-Optimized, per independent training) =====")
    allpos = all(r["dmin"] > 0 for r in rows); allver = all(r["vc"] >= r["vb"] for r in rows)
    print(f"trainings: {len(rows)} | EVERY image improved (dmin>0) in all: {allpos} | "
          f"verified-acc non-decreasing in all: {allver}")
    for r in rows:
        print(f"  {r['task']:13s} seed{r['seed']}: verified {r['vb']:3d}->{r['vc']:3d}/{r['n']} "
              f"(+{r['vc']-r['vb']}), mean per-img margin delta {r['dmean']:+.4f} (min {r['dmin']:+.4f})")
