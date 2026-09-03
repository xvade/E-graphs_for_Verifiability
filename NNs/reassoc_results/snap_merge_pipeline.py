"""STEPS 2-7: certified neuron-merging via row-proportionality snapping.

Procedure (the user's 7 steps):
  1 take an MLP  2 verify with CROWN  3 find near-proportional row pairs (row_j ~ beta*row_i)
  4 SNAP the pair to exact proportionality  5 MERGE the two now-proportional ReLU neurons in
  the next layer  6 reverify with CROWN  7 CERTIFY the snap changed the output negligibly.

Three nets, all in exact algebra:
  orig  --(lossy snap: A1[j]:=beta*A1[i], b1[j]:=beta*b1[i])-->  snapped
  snapped --(EXACT merge: drop neuron j, A2[:,i]+=beta*A2[:,j])-->  merged   (relu(beta z)=beta relu(z), beta>0)
float64 gate: snapped == merged (exact).

Step-7 certificate (analytic, SOUND, cheap). The snap changes only pre-activation z_j.
  d_j(x) = max over eps-box of |Delta z_j| = |r.c + rb| + sum_k rho_k|r_k|,  r=A1_orig[j]-beta*A1[i]
where c,rho are box center/radius (post [0,1] clamp). relu is 1-Lipschitz, so the induced
output-margin change is bounded by propagating d_j through |A2[:,j]|, relu(1-Lip), |A3|, |C|:
  delta_m(x) <= d_j * (|C| |A3| |A2[:,j]|)_m .
=> a SOUND certificate for the ORIGINAL net:  margin_orig >= margin_lb(merged) - delta.
Headline: verified-count of  (lb_merged - delta)  vs direct CROWN-Opt on orig, same images.

Analytic gain screen (to rank pairs; CROWN is ground truth): merging removes neuron j's
INDEPENDENT relaxation. Per next-layer coord k the pair's slack shrinks by
  (|A2[k,i]| + |beta||A2[k,j]| - |A2[k,i]+beta A2[k,j]|) * chordgap_i   (>=0, strict iff signs cancel)
propagated downstream by |C||A3|. Pairs pay iff gain > delta.

Run: alpha-beta-CROWN/.venv/bin/python NNs/reassoc_results/snap_merge_pipeline.py [config]
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "alpha-beta-CROWN", "complete_verifier"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

torch.set_num_threads(4)
DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                    "alpha-beta-CROWN", "complete_verifier", "datasets")
DIN, DOUT = 784, 10


class MLP(nn.Module):
    def __init__(self, H, p=0.0):
        super().__init__()
        self.A1 = nn.Linear(DIN, H); self.A2 = nn.Linear(H, H); self.A3 = nn.Linear(H, DOUT)
        self.H = H; self.p = p
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h1 = F.dropout(F.relu(self.A1(x)), self.p, self.training)
        h2 = F.dropout(F.relu(self.A2(h1)), self.p, self.training)
        return self.A3(h2)


class MergedNet(nn.Module):
    """orig with a set of layer-1 neuron pairs (i<-j, factor beta) snapped+merged out.
    Given the snap A1[j]=beta A1[i], neuron j == beta*neuron i, so drop j and fold
    beta*A2[:,j] into A2[:,i]. Exact w.r.t. the SNAPPED net."""
    def __init__(self, base, pairs):
        super().__init__()
        d = torch.float64
        A1w = base.A1.weight.data.to(d).clone(); A1b = base.A1.bias.data.to(d).clone()
        A2w = base.A2.weight.data.to(d).clone(); A2b = base.A2.bias.data.to(d).clone()
        A3w = base.A3.weight.data.to(d).clone(); A3b = base.A3.bias.data.to(d).clone()
        drop = []
        for (i, j, beta) in pairs:
            A2w[:, i] = A2w[:, i] + beta * A2w[:, j]   # fold merged neuron
            drop.append(j)
        keep = [k for k in range(base.H) if k not in drop]
        A1w, A1b = A1w[keep], A1b[keep]; A2w = A2w[:, keep]
        def lin(W, b):
            m = nn.Linear(W.shape[1], W.shape[0]); m.weight.data = W.float(); m.bias.data = b.float(); return m
        self.A1 = lin(A1w, A1b); self.A2 = lin(A2w, A2b); self.A3 = lin(A3w, A3b)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.A3(F.relu(self.A2(F.relu(self.A1(x)))))


def snapped_net(base, pairs):
    """orig with A1[j]:=beta A1[i], b1[j]:=beta b1[i] (lossy). Same shape as orig."""
    net = copy.deepcopy(base); net.p = 0.0
    with torch.no_grad():
        for (i, j, beta) in pairs:
            net.A1.weight.data[j] = beta * net.A1.weight.data[i]
            net.A1.bias.data[j] = beta * net.A1.bias.data[i]
    net.eval(); return net


def train(net, seed=0, epochs=3, n=30000, wd=0.0, prox=0.0, prox_pairs=None):
    """prox>0 adds a soft penalty pulling listed row-pairs toward proportionality (only
    used for the pipeline-existence net; clearly labeled). prox_pairs: list of (i,j)."""
    tr = datasets.MNIST(DATA, True, download=False, transform=transforms.ToTensor())
    g = torch.Generator().manual_seed(seed)
    X = torch.stack([tr[i][0] for i in range(n)]); Y = torch.tensor([tr[i][1] for i in range(n)])
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=wd); net.train(); bs = 256
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i0 in range(0, n, bs):
            b = perm[i0:i0+bs]; opt.zero_grad()
            loss = F.cross_entropy(net(X[b]), Y[b])
            if prox > 0 and prox_pairs:
                W = net.A1.weight; Bb = net.A1.bias
                for (i, j) in prox_pairs:
                    a = torch.cat([W[i], Bb[i:i+1]]); t = torch.cat([W[j], Bb[j:j+1]])
                    beta = (a @ t) / (a @ a + 1e-9)
                    loss = loss + prox * ((t - beta * a) ** 2).sum() / (t @ t + 1e-9)
            loss.backward(); opt.step()
    net.eval()


def ibp_layer1_gap(net, x, eps):
    """chord slack gap per layer-1 neuron per image: for unstable (l<0<u) neuron the max
    CROWN/relu relaxation slack ~ -l*u/(u-l); stable -> 0. Returns (n,H)."""
    W = net.A1.weight.data; b = net.A1.bias.data
    c = ((x.clamp(0, 1)).view(x.shape[0], -1)); xl = (c - eps).clamp(0, 1); xu = (c + eps).clamp(0, 1)
    cen = (xl + xu) / 2; rad = (xu - xl) / 2
    zc = cen @ W.t() + b; zr = rad @ W.abs().t()
    l = zc - zr; u = zc + zr
    unstable = (l < 0) & (u > 0)
    gap = torch.where(unstable, (-l * u) / (u - l + 1e-12), torch.zeros_like(l))
    return gap, l, u


def analytic_pair_screen(net, x, y, eps, C):
    """For every layer-1 pair (i<j): LS beta on [w|b], residual, per-image mean delta and
    mean gain (binding-margin coord). Returns list of dicts sorted by (gain-delta)."""
    W1 = net.A1.weight.data; b1 = net.A1.bias.data
    A2 = net.A2.weight.data; A3 = net.A3.weight.data
    H = net.H
    Wb = torch.cat([W1, b1[:, None]], 1)
    norms = Wb.norm(dim=1); live = norms > 1e-3 * norms.max()
    gap, l, u = ibp_layer1_gap(net, x, eps)          # (n,H)
    c = x.clamp(0, 1).view(x.shape[0], -1); xl = (c - eps).clamp(0, 1); xu = (c + eps).clamp(0, 1)
    cen = (xl + xu) / 2; rad = (xu - xl) / 2
    # downstream magnitude to each margin coord m through relu2(1-Lip): |C||A3| : (n,DOUT,H2)
    CA3 = C.abs() @ A3.abs()                          # (n,DOUT,H2) but C is (n,DOUT,DOUT)... compute below
    recs = []
    absA2 = A2.abs()
    for i in range(H):
        if not live[i]: continue
        ai = Wb[i]
        for j in range(i+1, H):
            if not live[j]: continue
            tj = Wb[j]; beta = float((ai @ tj) / (ai @ ai))
            if beta <= 0: continue                   # need beta>0 for relu(beta z)=beta relu(z)
            r = W1[j] - beta * W1[i]; rb = float(b1[j] - beta * b1[i])
            res = float((tj - beta * ai).norm() / (tj.norm() + 1e-12))
            # d_j per image (box max of |Delta z_j|), Delta z_j = -(r.x + rb)
            dj = (cen @ r + rb).abs() + rad @ r.abs()          # (n,)
            # cancellation vector over next-layer coords k: (|A2ki|+beta|A2kj|-|A2ki+beta A2kj|)
            canc = absA2[:, i] + beta * absA2[:, j] - (A2[:, i] + beta * A2[:, j]).abs()   # (H2,)
            # downstream |C||A3| to each margin coord, times chordgap_i (per image)
            # delta_m = dj * (|C||A3| |A2[:,j]|)_m ; gain_m = chordgap_i * (|C||A3| canc)_m
            CA3 = (C.abs() @ A3.abs())                  # (n,DOUT,H2)
            dvec = (CA3 @ absA2[:, j])                  # (n,DOUT)
            gvec = (CA3 @ canc)                         # (n,DOUT)
            gapi = gap[:, i]                            # (n,)
            delta_m = dj[:, None] * dvec                # (n,DOUT)
            gain_m = gapi[:, None] * gvec               # (n,DOUT)
            # binding margin coord = smallest (per image) of the true margins is unknown here;
            # use the worst-case delta (max over m) and matched gain (max canc coord) as summary
            recs.append(dict(i=i, j=j, beta=beta, res=res,
                             dmean=float(delta_m.max(1).values.mean()),
                             gmean=float(gain_m.max(1).values.mean()),
                             canc_ratio=float((A2[:, i] + beta * A2[:, j]).abs().sum() /
                                              (absA2[:, i].sum() + beta * absA2[:, j].sum() + 1e-12))))
    recs.sort(key=lambda d: (d["gmean"] - d["dmean"]), reverse=True)
    return recs


def margin_lb(net, x, y, eps, method):
    bm = BoundedModule(net, x[:1], verbose=False); n = x.shape[0]
    C = -torch.eye(DOUT)[None].repeat(n, 1, 1); C[torch.arange(n), :, y] += 1.0
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=(x - eps).clamp(0, 1), x_U=(x + eps).clamp(0, 1))
    lb, _ = bm.compute_bounds(x=(BoundedTensor(x, ptb),), C=C, method=method)
    lb = lb.clone(); lb[torch.arange(n), y] = float("inf")
    return lb.min(1).values.detach()


def analytic_delta(base, pairs, x, y, eps, per_coord=False):
    """SOUND per-image, per-margin-coord bound on |margin_orig - margin_snap|. Returns the
    binding value delta (n,) = max_m of the coordinate bound (so lb_merged - delta is sound),
    or the full (n,DOUT) per-coord bound if per_coord=True."""
    W1 = base.A1.weight.data; b1 = base.A1.bias.data; A2 = base.A2.weight.data; A3 = base.A3.weight.data
    n = x.shape[0]
    C = -torch.eye(DOUT)[None].repeat(n, 1, 1); C[torch.arange(n), :, y] += 1.0  # (n,DOUT,DOUT)
    c = x.clamp(0, 1).view(n, -1); xl = (c - eps).clamp(0, 1); xu = (c + eps).clamp(0, 1)
    cen = (xl + xu) / 2; rad = (xu - xl) / 2
    total = torch.zeros(n, DOUT)
    for (i, j, beta) in pairs:
        r = W1[j] - beta * W1[i]; rb = float(b1[j] - beta * b1[i])
        dj = (cen @ r + rb).abs() + rad @ r.abs()            # (n,)
        prop = (C.abs() @ A3.abs()) @ A2[:, j].abs()          # (n,DOUT)
        total = total + dj[:, None] * prop
    return total if per_coord else total.max(1).values        # (n,DOUT) or (n,)


def delta_soundness(base, pairs, x, y, eps, nsamp=1000):
    """EMPIRICAL check that the analytic per-coord delta upper-bounds the true margin change:
    sample random points in each eps-box, forward orig vs snapped, take max|Delta margin_m|,
    assert <= delta_m. Returns worst (emp - delta) over all images/coords (must be <= 0)."""
    snp = snapped_net(base, pairs); n = x.shape[0]
    dc = analytic_delta(base, pairs, x, y, eps, per_coord=True)          # (n,DOUT)
    c = x.clamp(0, 1).view(n, -1)
    worst = -1e9
    with torch.no_grad():
        for k in range(n):
            lo = (c[k] - eps).clamp(0, 1); hi = (c[k] + eps).clamp(0, 1)
            pts = lo + (hi - lo) * torch.rand(nsamp, c.shape[1])
            g = base(pts) - snp(pts)                                     # (nsamp,DOUT)
            dm = (g[:, y[k]][:, None] - g).abs().max(0).values           # (DOUT,) true max |Delta margin_m|
            worst = max(worst, float((dm - dc[k]).max()))
    return worst


def run(base, x, y, eps, pairs, tag):
    snp = snapped_net(base, pairs); mrg = MergedNet(base, pairs)
    with torch.no_grad():
        gate = (copy.deepcopy(snp).double()(x.double()) - copy.deepcopy(mrg).double()(x.double())).abs().max().item()
        emp = (base(x) - snp(x)).abs().max().item()           # empirical fn change from snap
    lb_o = margin_lb(base, x, y, eps, "CROWN-Optimized")
    lb_s = margin_lb(snp,  x, y, eps, "CROWN-Optimized")
    lb_m = margin_lb(mrg,  x, y, eps, "CROWN-Optimized")
    delta = analytic_delta(base, pairs, x, y, eps)
    snd = delta_soundness(base, pairs, x, y, eps)             # must be <= 0 (delta dominates true change)
    cert = lb_m - delta                                       # SOUND certificate for ORIG
    v = lambda t: int((t > 0).sum())
    cert_gain = cert - lb_o                                   # certificate improvement over direct CROWN
    print(f"\n--- {tag}: {len(pairs)} pair(s) snapped ---")
    print(f"  gate(snap==merge)={gate:.1e}  empirical max|f_snap-f_orig|={emp:.3f}  "
          f"analytic delta mean={delta.mean():.3f} max={delta.max():.3f}  "
          f"delta-soundness worst(emp-delta)={snd:+.3f} ({'OK<=0' if snd <= 0 else 'VIOLATED'})")
    print(f"  verified/{len(x)}:  orig={v(lb_o)}  snapped={v(lb_s)}  merged={v(lb_m)}  "
          f"CERT(lb_merged-delta)={v(cert)}")
    print(f"  mean margin lb: orig={lb_o.mean():+.3f} merged={lb_m.mean():+.3f} cert={cert.mean():+.3f}")
    print(f"  step5 isolation (merge tightens CROWN): merged-vs-snapped mean = {(lb_m-lb_s).mean():+.4f}, "
          f"min = {(lb_m-lb_s).min():+.4f}")
    print(f"  HEADLINE certificate-for-ORIG (cert - lb_orig): mean={cert_gain.mean():+.4f} "
          f"min={cert_gain.min():+.4f} | images where cert>lb_orig: "
          f"{int((cert_gain>1e-6).sum())}/{len(x)}, where strictly worse: {int((cert_gain<-1e-6).sum())}")
    return dict(tag=tag, v_orig=v(lb_o), v_merged=v(lb_m), v_cert=v(cert),
                delta_mean=float(delta.mean()), gate=gate,
                cert_gain_mean=float(cert_gain.mean()), n_better=int((cert_gain>1e-6).sum()))


# disjoint proportional pairs for the existence net (each mergeable independently)
PROX_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
RES_MAX = 0.20   # only genuinely-proportional rows are snap candidates
CONFIGS = {
    "vanilla":  dict(H=64, p=0.0, wd=0.0, epochs=3, prox=0.0, eps=[0.02, 0.03, 0.05]),
    "existence":dict(H=64, p=0.0, wd=0.0, epochs=4, prox=5.0, eps=[0.03, 0.05, 0.08]),
}

if __name__ == "__main__":
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "vanilla"
    cfg = CONFIGS[cfg_name]; NIMG = 60
    te = datasets.MNIST(DATA, False, download=False, transform=transforms.ToTensor())
    xt = torch.stack([te[i][0] for i in range(1500)]); yt = torch.tensor([te[i][1] for i in range(1500)])
    torch.manual_seed(0); net = MLP(cfg["H"], cfg["p"])
    prox_pairs = PROX_PAIRS if cfg["prox"] > 0 else None
    train(net, epochs=cfg["epochs"], wd=cfg["wd"], prox=cfg["prox"], prox_pairs=prox_pairs)
    with torch.no_grad():
        pb = net(xt); acc = (pb.argmax(1) == yt).float().mean().item()
    idx = torch.where(pb.argmax(1) == yt)[0][:NIMG]; xs, ys = xt[idx], yt[idx]
    print(f"===== config={cfg_name} H={cfg['H']} acc={acc:.3f} n={len(xs)} =====")

    for eps in cfg["eps"]:
        C = -torch.eye(DOUT)[None].repeat(len(xs), 1, 1); C[torch.arange(len(xs)), :, ys] += 1.0
        recs = analytic_pair_screen(net, xs, ys, eps, C)
        cands = [r for r in recs if r["res"] < RES_MAX]     # snap candidates: proportional only
        print(f"\n##### eps={eps} #####")
        print(f"beta>0 live pairs={len(recs)}; snap-candidates (res<{RES_MAX})={len(cands)}; "
              f"closest-pair residual overall={min(r['res'] for r in recs):.3f}")
        if not cands:
            best = recs[0]
            print(f"  NO SNAPPABLE PAIRS: technique N/A on this net. Best-by-net pair only prunes "
                  f"(beta={best['beta']:+.3f}, res={best['res']:.3f}). Crossover: a pair must reach "
                  f"res well below {RES_MAX} to have gain>delta; real training floors ~0.4-0.66.")
            continue
        print("  top snap-candidates (gain-delta):")
        for r in cands[:6]:
            print(f"    ({r['i']:3d},{r['j']:3d}) beta={r['beta']:+.3f} res={r['res']:.3f} "
                  f"gain~{r['gmean']:.3f} delta~{r['dmean']:.3f} net~{r['gmean']-r['dmean']:+.3f} "
                  f"canc={r['canc_ratio']:.2f}")
        # snap ALL disjoint candidates (accumulate gain; deltas add linearly)
        seen = set(); dis = []
        for r in cands:
            if r["i"] in seen or r["j"] in seen: continue
            seen.update([r["i"], r["j"]]); dis.append((r["i"], r["j"], r["beta"]))
        run(net, xs, ys, eps, [dis[0]], "best-single")
        run(net, xs, ys, eps, dis, f"all-{len(dis)}-disjoint")
