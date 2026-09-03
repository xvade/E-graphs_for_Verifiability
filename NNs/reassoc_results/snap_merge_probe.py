"""STEP 1 (feasibility gate): does a standard-trained plain-ReLU MLP contain
near-PROPORTIONAL row pairs in a hidden layer?  (row_j ~ beta * row_i, incl. bias)

If yes, we can SNAP the pair to exact proportionality (small function change),
MERGE the two now-proportional ReLU neurons into one in the next layer, and (step 7)
CERTIFY via CROWN that the snap changed the output negligibly over the eps-ball --
verifying the ORIGINAL net better by routing through the merged surrogate.

This probe just measures the residual distribution across widths -- it decides
feasibility BEFORE we build the snap/merge/certify pipeline. Standard training
decorrelates features, so 64-wide may be thin; overparameterization (128/256)
breeds near-duplicates. Dead (near-zero-norm) rows are excluded (trivially
"proportional", boring).

Run: alpha-beta-CROWN/.venv/bin/python NNs/reassoc_results/snap_merge_probe.py
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
    def __init__(self, H):
        super().__init__()
        self.A1 = nn.Linear(DIN, H); self.A2 = nn.Linear(H, H); self.A3 = nn.Linear(H, DOUT)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h1 = F.relu(self.A1(x)); h2 = F.relu(self.A2(h1)); return self.A3(h2)


def train(net, seed=0, epochs=3, n=30000):
    tr = datasets.MNIST(DATA, True, download=False, transform=transforms.ToTensor())
    g = torch.Generator().manual_seed(seed)
    X = torch.stack([tr[i][0] for i in range(n)]); Y = torch.tensor([tr[i][1] for i in range(n)])
    opt = torch.optim.Adam(net.parameters(), 1e-3); net.train(); bs = 256
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            opt.zero_grad(); F.cross_entropy(net(X[b]), Y[b]).backward(); opt.step()
    net.eval()


def pair_residuals(Wb):
    """Wb: (H, d+1) augmented rows [w | b]. For each pair (i<j) fit t~beta*a by
    least squares (beta any sign) and via rank-1 SVD (symmetric). Return arrays."""
    H = Wb.shape[0]
    norms = np.linalg.norm(Wb, axis=1)
    live = norms > 1e-3 * norms.max()          # drop dead/near-zero rows
    recs = []
    for i in range(H):
        if not live[i]:
            continue
        a = Wb[i]; aa = a @ a
        for j in range(i+1, H):
            if not live[j]:
                continue
            t = Wb[j]
            beta = (a @ t) / aa                 # LS fit t ~ beta a (both signs)
            res_ls = np.linalg.norm(t - beta*a) / (np.linalg.norm(t) + 1e-12)
            M = np.stack([a, t])                # 2 x (d+1)
            s = np.linalg.svd(M, compute_uv=False)
            res_svd = s[1] / (np.sqrt(s[0]**2 + s[1]**2) + 1e-12)  # symmetric off-axis frac
            recs.append((i, j, beta, res_ls, res_svd))
    return recs


if __name__ == "__main__":
    te = datasets.MNIST(DATA, False, download=False, transform=transforms.ToTensor())
    xt = torch.stack([te[i][0] for i in range(2000)]); yt = torch.tensor([te[i][1] for i in range(2000)])
    for H in [64, 128, 256]:
        torch.manual_seed(0)
        net = MLP(H); train(net)
        with torch.no_grad():
            acc = (net(xt).argmax(1) == yt).float().mean().item()
        print(f"\n===== H={H}  test acc={acc:.3f} =====")
        for name, lin, nxt in [("A1", net.A1, net.A2), ("A2", net.A2, net.A3)]:
            Wb = torch.cat([lin.weight.data, lin.bias.data[:, None]], 1).numpy()
            recs = pair_residuals(Wb)
            res_ls = np.array([r[3] for r in recs]); res_svd = np.array([r[4] for r in recs])
            order = np.argsort(res_ls)
            qs = np.quantile(res_ls, [0, 0.001, 0.01, 0.05, 0.5])
            print(f" {name} ({lin.weight.shape[0]} rows, {len(recs)} live pairs) "
                  f"LS-residual quantiles [min,.1%,1%,5%,50%]: "
                  + ", ".join(f"{q:.3f}" for q in qs))
            print(f"   top-5 closest pairs (i,j,beta,res_ls,res_svd):")
            for k in order[:5]:
                i, j, beta, rls, rsv = recs[k]
                print(f"     ({i:3d},{j:3d}) beta={beta:+.3f} res_ls={rls:.4f} res_svd={rsv:.4f}")
