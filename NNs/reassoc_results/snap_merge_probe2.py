"""STEP 1b (extended feasibility gate): do near-PROPORTIONAL row pairs occur under
STANDARD training practices that are independently known to breed redundant/co-adapted
units -- dropout, strong weight decay, longer training on a small subset?

Baseline vanilla MLP had min LS-residual ~0.66 (no snappable pairs; width didn't help).
Before falling back to a bespoke proportionality regularizer (which is planted structure
and will draw the "not a real net" objection), probe whether a REAL training variant
produces residuals small enough that a snap is cheap (<= ~0.1-0.2).

Reports, per config, the residual distribution + top pairs for both hidden layers.
Run: alpha-beta-CROWN/.venv/bin/python NNs/reassoc_results/snap_merge_probe2.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "alpha-beta-CROWN", "complete_verifier"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

torch.set_num_threads(4)
DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                    "alpha-beta-CROWN", "complete_verifier", "datasets")
DIN, DOUT = 784, 10


class MLP(nn.Module):
    def __init__(self, H, p=0.0):
        super().__init__()
        self.A1 = nn.Linear(DIN, H); self.A2 = nn.Linear(H, H); self.A3 = nn.Linear(H, DOUT)
        self.p = p
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h1 = F.dropout(F.relu(self.A1(x)), self.p, self.training)
        h2 = F.dropout(F.relu(self.A2(h1)), self.p, self.training)
        return self.A3(h2)


def train(net, seed=0, epochs=3, n=30000, wd=0.0, lr=1e-3):
    tr = datasets.MNIST(DATA, True, download=False, transform=transforms.ToTensor())
    g = torch.Generator().manual_seed(seed)
    X = torch.stack([tr[i][0] for i in range(n)]); Y = torch.tensor([tr[i][1] for i in range(n)])
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=wd); net.train(); bs = 256
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            opt.zero_grad(); F.cross_entropy(net(X[b]), Y[b]).backward(); opt.step()
    net.eval()


def pair_residuals(Wb):
    H = Wb.shape[0]
    norms = np.linalg.norm(Wb, axis=1); live = norms > 1e-3 * norms.max()
    recs = []
    for i in range(H):
        if not live[i]: continue
        a = Wb[i]; aa = a @ a
        for j in range(i+1, H):
            if not live[j]: continue
            t = Wb[j]; beta = (a @ t) / aa
            res_ls = np.linalg.norm(t - beta*a) / (np.linalg.norm(t) + 1e-12)
            recs.append((i, j, beta, res_ls))
    return recs


CONFIGS = [
    ("vanilla",       dict(H=64,  p=0.0, wd=0.0,  epochs=3)),
    ("dropout0.5",    dict(H=64,  p=0.5, wd=0.0,  epochs=8)),
    ("dropout0.5_128",dict(H=128, p=0.5, wd=0.0,  epochs=8)),
    ("wd1e-3",        dict(H=64,  p=0.0, wd=1e-3, epochs=6)),
    ("wd3e-3_drop",   dict(H=64,  p=0.3, wd=3e-3, epochs=8)),
    ("long_small",    dict(H=64,  p=0.0, wd=0.0,  epochs=25, n=4000)),
]

if __name__ == "__main__":
    te = datasets.MNIST(DATA, False, download=False, transform=transforms.ToTensor())
    xt = torch.stack([te[i][0] for i in range(2000)]); yt = torch.tensor([te[i][1] for i in range(2000)])
    for name, cfg in CONFIGS:
        H, p = cfg["H"], cfg["p"]
        torch.manual_seed(0)
        net = MLP(H, p)
        train(net, epochs=cfg["epochs"], wd=cfg["wd"], n=cfg.get("n", 30000))
        with torch.no_grad():
            acc = (net(xt).argmax(1) == yt).float().mean().item()
        print(f"\n===== {name}  (H={H} p={p} wd={cfg['wd']} ep={cfg['epochs']})  acc={acc:.3f} =====")
        for lname, lin in [("A1", net.A1), ("A2", net.A2)]:
            Wb = torch.cat([lin.weight.data, lin.bias.data[:, None]], 1).numpy()
            recs = pair_residuals(Wb)
            res = np.array([r[3] for r in recs]); order = np.argsort(res)
            qs = np.quantile(res, [0, 0.001, 0.01, 0.05])
            best = recs[order[0]]
            print(f"  {lname}: min/.1%/1%/5% residual = " + ", ".join(f"{q:.3f}" for q in qs)
                  + f"  | best pair ({best[0]},{best[1]}) beta={best[2]:+.3f}")
