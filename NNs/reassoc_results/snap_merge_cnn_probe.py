"""Does a trained CNN have near-PROPORTIONAL structure the snap-merge rule could exploit?
Two distinct structures to measure separately:

  (A) UNROLLED rows: interpret conv as its dense im2col/Toeplitz matrix and measure pairwise
      row residuals. Prediction: still ~1.0 -- shifted kernels have disjoint supports, so
      "CNN interpreted as an MLP" does NOT create proportional rows by itself.
  (B) CHANNEL kernels: per output-channel kernel (C_in*k*k dims), pairwise residuals across
      channels. Kernels are LOW-dimensional (9, 27, ...) so near-proportional pairs are far
      likelier than 784-dim MLP rows. A proportional channel pair => proportional feature maps
      everywhere => an exact channel MERGE (collapses H*W ReLUs at once).

Compare residual floors to the dense-MLP floor (~0.4-0.66). If (B) is small, Finding 1 flips
for CNNs and the full snap-merge pipeline (at the channel level) is worth building.

Run: alpha-beta-CROWN/.venv/bin/python NNs/reassoc_results/snap_merge_cnn_probe.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "alpha-beta-CROWN", "complete_verifier"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

torch.set_num_threads(4)
DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                    "alpha-beta-CROWN", "complete_verifier", "datasets")


class CNN(nn.Module):
    def __init__(self, c1=16, c2=32, k=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, k, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, k, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(c2 * 7 * 7, 10)
    def forward(self, x):
        h = self.pool(F.relu(self.conv1(x)))          # 28->14
        h = self.pool(F.relu(self.conv2(h)))          # 14->7
        return self.fc(h.flatten(1))


def train(net, seed=0, epochs=3, n=30000, wd=0.0, p=0.0):
    tr = datasets.MNIST(DATA, True, download=False, transform=transforms.ToTensor())
    g = torch.Generator().manual_seed(seed)
    X = torch.stack([tr[i][0] for i in range(n)]); Y = torch.tensor([tr[i][1] for i in range(n)])
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=wd); net.train(); bs = 256
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            out = net(X[b])
            if p > 0: pass
            opt.zero_grad(); F.cross_entropy(out, Y[b]).backward(); opt.step()
    net.eval()


def pair_residuals(rows):
    """rows: (N, d) numpy. LS beta on each pair, relative residual; exclude near-zero rows."""
    N = rows.shape[0]; norms = np.linalg.norm(rows, axis=1)
    live = norms > 1e-3 * norms.max(); res = []
    for i in range(N):
        if not live[i]: continue
        a = rows[i]; aa = a @ a
        for j in range(i+1, N):
            if not live[j]: continue
            t = rows[j]; beta = (a @ t) / aa
            res.append(np.linalg.norm(t - beta*a) / (np.linalg.norm(t) + 1e-12))
    return np.array(res)


def im2col_rows(conv, in_hw):
    """Build the dense unrolled matrix of a conv layer (one row per output neuron =
    out_channel x spatial). Return a representative subset of rows (cap for speed)."""
    W = conv.weight.data; Cout, Cin, k, _ = W.shape; H, Wd = in_hw
    pad = conv.padding[0]; Hout, Wout = H, Wd  # padding='same' here
    rows = []
    # sample a grid of output positions to keep it cheap; full input dim = Cin*H*W
    for oc in range(min(Cout, 8)):
        for (oy, ox) in [(3, 3), (3, 10), (10, 3), (7, 7)]:
            r = np.zeros(Cin * H * Wd)
            for ic in range(Cin):
                for ky in range(k):
                    for kx in range(k):
                        iy = oy + ky - pad; ix = ox + kx - pad
                        if 0 <= iy < H and 0 <= ix < Wd:
                            r[ic * H * Wd + iy * Wd + ix] = W[oc, ic, ky, kx].item()
            rows.append(r)
    return np.stack(rows)


def report(net, tag):
    print(f"\n===== {tag} =====")
    for name, conv, hw in [("conv1", net.conv1, (28, 28)), ("conv2", net.conv2, (14, 14))]:
        W = conv.weight.data                                  # (Cout,Cin,k,k)
        kern = W.reshape(W.shape[0], -1).numpy()              # (B) per-channel kernel rows
        rk = pair_residuals(kern)
        qs = np.quantile(rk, [0, 0.01, 0.05, 0.5])
        print(f" {name} (B) channel-kernel pairs ({kern.shape[0]} ch, dim={kern.shape[1]}): "
              f"min/1%/5%/50% res = " + ", ".join(f"{q:.3f}" for q in qs))
        ru = pair_residuals(im2col_rows(conv, hw))            # (A) unrolled dense rows
        print(f" {name} (A) unrolled dense-row pairs (sampled): min res = {ru.min():.3f}  "
              f"(shifted kernels -> expect ~1.0)")


if __name__ == "__main__":
    te = datasets.MNIST(DATA, False, download=False, transform=transforms.ToTensor())
    xt = torch.stack([te[i][0] for i in range(2000)]); yt = torch.tensor([te[i][1] for i in range(2000)])
    for tag, cfg in [("vanilla c16/32", dict()),
                     ("wide c32/64", dict(c1=32, c2=64)),
                     ("wd1e-3 c16/32", dict())]:
        wd = 1e-3 if "wd" in tag else 0.0
        torch.manual_seed(0); net = CNN(**cfg); train(net, wd=wd)
        with torch.no_grad():
            acc = (net(xt).argmax(1) == yt).float().mean().item()
        report(net, f"{tag} acc={acc:.3f}")
