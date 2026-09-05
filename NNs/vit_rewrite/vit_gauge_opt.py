#!/usr/bin/env python
"""
LEARN the attention gauge (2026-09-04). Heavy: compute node.

The gauge rewrite  (X Wq_h)(X Wk_h)^T = (X Wq_h G)(X Wk_h G^-T)^T ,  A (X Wv_h) Wo_h = A (X Wv_h G)(G^-1 Wo_h)
is EXACT for every invertible G, but the McCormick relaxation CROWN uses for the two bilinear products depends on
the per-coordinate widths of the operands, which G changes. Instead of the closed-form SVD balance (R4/R5), pick
G by gradient ascent on the CROWN lower bound itself: the bound is differentiable w.r.t. the effective weights
(auto_LiRPA), and the effective weights are differentiable in G (exact gauge algebra), so we chain-rule
  dLB/dG = sum_i dLB/dW_i' * dW_i'/dG .
Tuning boxes are eps=1/255 boxes around CIFAR-10 TRAIN images (disjoint from the benchmark's test instances),
so the learned G is a fixed, input-independent rewrite evaluated out-of-sample on the 100 benchmark instances.
A tiny conditioning penalty keeps G well-conditioned (fp32 realization of the rewritten weights stays exact).

  python vit_gauge_opt.py --model pgd_2_3_16 --steps 300 --batch 32 --init svd --obj mix --out gauges/pgd_mix.pt
"""
import sys, os, pickle, time, argparse, numpy as np, torch, torch.nn as nn
REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
sys.path.insert(0, os.path.join(REPO, "alpha-beta-CROWN/complete_verifier")); sys.path.insert(0, os.path.join(REPO, "NNs/vit_rewrite"))
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from vit_model import ViT, stock_attn_weights, svd_gauges, gate_variants
from vit_bounds import specs_C, BENCH
CIF = os.path.join(REPO, "alpha-beta-CROWN/complete_verifier/datasets/cifar-10-batches-py")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1); STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

def cifar_train_boxes(n, net, dev, seed=0, eps_pix=1 / 255, batch_file="data_batch_1"):
    """eps-boxes around correctly-classified CIFAR TRAIN images, built exactly like the benchmark vnnlibs."""
    d = pickle.load(open(os.path.join(CIF, batch_file), "rb"), encoding="bytes")
    X = torch.tensor(d[b"data"]).reshape(-1, 3, 32, 32).float() / 255; Y = torch.tensor(d[b"labels"])
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed)); X, Y = X[perm], Y[perm]
    x0 = (X - MEAN) / STD; xl = ((X - eps_pix).clamp(0, 1) - MEAN) / STD; xu = ((X + eps_pix).clamp(0, 1) - MEAN) / STD
    with torch.no_grad(): pred = torch.cat([net(x0[i:i + 500].to(dev)).argmax(1).cpu() for i in range(0, len(x0), 500)])
    keep = (pred == Y).nonzero().ravel()[:n]
    return x0[keep], xl[keep], xu[keep], Y[keep]

def effective_weights(stock, Gq, Ga, H, dh):
    """Differentiable exact gauge algebra (float32). Returns per layer [Wq,Wk,Wv,Wo,bq,bk,bv]."""
    out = []
    for l, (Wq, Wk, Wv, Wo, bq, bk, bv, bo) in enumerate(stock):
        q, k, v, o, bq2, bk2, bv2 = [], [], [], [], [], [], []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh); G = Gq[l, h]; Gi = torch.linalg.inv(G); Ga_ = Ga[l, h]; Gai = torch.linalg.inv(Ga_)
            q.append(Wq[:, sl] @ G); bq2.append(bq[sl] @ G); k.append(Wk[:, sl] @ Gi.T); bk2.append(bk[sl] @ Gi.T)
            v.append(Wv[:, sl] @ Ga_); bv2.append(bv[sl] @ Ga_); o.append(Gai @ Wo[sl, :])
        out.append([torch.cat(q, 1), torch.cat(k, 1), torch.cat(v, 1), torch.cat(o, 0), torch.cat(bq2), torch.cat(bk2), torch.cat(bv2)])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pgd_2_3_16"); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_train", type=int, default=512); ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--init", default="svd", help="svd|id")
    ap.add_argument("--obj", default="mix", help="mean|min|mix : which lb statistic to maximize"); ap.add_argument("--cond_pen", type=float, default=1e-4)
    ap.add_argument("--softmax", default="lse"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="both", help="both|qk|av : which gauges to learn")
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip on G"); ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--debug", type=int, default=0, help="anomaly detection + per-weight grad NaN report")
    ap.add_argument("--hard", type=int, default=0, help="1: pick the n_train tuning boxes with the smallest |stock CROWN min-lb| out of a pool of --pool boxes (benchmark-like, near the decision boundary)")
    ap.add_argument("--pool", type=int, default=1000)
    a = ap.parse_args(); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; onnx_path = os.path.join(BENCH, "onnx", a.model + ".onnx")
    stock64, H, dh, L = stock_attn_weights(onnx_path); stock = [tuple(t.float().to(dev) for t in ws) for ws in stock64]
    Gq0, Ga0 = svd_gauges(onnx_path) if a.init == "svd" else (torch.eye(dh, dtype=torch.float64).expand(L, H, dh, dh).clone(),) * 2
    Gq = nn.Parameter(Gq0.float().clone().to(dev), requires_grad=a.which in ("both", "qk")); Ga = nn.Parameter(Ga0.float().clone().to(dev), requires_grad=a.which in ("both", "av"))
    work = ViT(onnx_path).to(dev)
    names = ["Wq", "Wk", "Wv", "Wo", "bq", "bk", "bv"]
    for blk in work.blocks:
        for nm in names: getattr(blk.attn, nm).requires_grad_(True)
    x0, xl, xu, Y = cifar_train_boxes(a.pool if a.hard else a.n_train, work, dev, seed=a.seed)
    bound_opts = {"softmax": a.softmax, "conv_mode": "matrix"}
    lirpa = BoundedModule(work, torch.empty(1, 3, 32, 32, device=dev), bound_opts=bound_opts, device=dev)
    if a.hard:   # stock (identity-gauge) CROWN min-lb over the pool; keep the boxes nearest the decision boundary
        mins = []
        with torch.no_grad():
            for i in range(0, len(x0), a.batch):
                idx = slice(i, min(len(x0), i + a.batch)); C = torch.stack([specs_C(int(y))[0] for y in Y[idx]]).to(dev)
                bx = BoundedTensor(x0[idx].to(dev), PerturbationLpNorm(norm=np.inf, x_L=xl[idx].to(dev), x_U=xu[idx].to(dev)))
                lb, _ = lirpa.compute_bounds(x=(bx,), method="CROWN", C=C, bound_lower=True, bound_upper=False); mins.append(lb.min(1).values.cpu())
        m = torch.cat(mins); order = m.abs().argsort()[:a.n_train]
        print(f"# pool of {len(m)} boxes: stock min-lb mean {m.mean():+.4f}, frac_ver {(m > 0).float().mean():.3f}; HARD subset of {len(order)}: min-lb mean {m[order].mean():+.4f}, "
              f"range [{m[order].min():+.4f}, {m[order].max():+.4f}], frac_ver {(m[order] > 0).float().mean():.3f}", flush=True)
        x0, xl, xu, Y = x0[order], xl[order], xu[order], Y[order]
    print(f"# tuning boxes: {len(x0)} CIFAR-train images (correctly classified{', hard subset' if a.hard else ''}), eps=1/255", flush=True)
    params = [p for p in (Gq, Ga) if p.requires_grad]; opt = torch.optim.Adam(params, lr=a.lr)
    def load_eff(effs):
        with torch.no_grad():
            for l, blk in enumerate(work.blocks):
                for nm, t in zip(names, effs[l]): getattr(blk.attn, nm).copy_(t)
    def objective(lb):
        if a.obj == "mean": return lb.mean()
        if a.obj == "min": return lb.min(1).values.mean()
        return 0.5 * lb.mean() + 0.5 * lb.min(1).values.mean()
    def evaluate(n=128):
        n = min(n, len(x0))
        effs = effective_weights(stock, Gq.detach(), Ga.detach(), H, dh); load_eff(effs); tot = []; mins = []
        with torch.no_grad():
            for i in range(0, n, a.batch):
                idx = slice(i, min(n, i + a.batch)); C = torch.stack([specs_C(int(y))[0] for y in Y[idx]]).to(dev)
                bx = BoundedTensor(x0[idx].to(dev), PerturbationLpNorm(norm=np.inf, x_L=xl[idx].to(dev), x_U=xu[idx].to(dev)))
                lb, _ = lirpa.compute_bounds(x=(bx,), method="CROWN", C=C, bound_lower=True, bound_upper=False)
                tot.append(lb.cpu()); mins.append(lb.min(1).values.cpu())
        lb = torch.cat(tot); m = torch.cat(mins); return lb.mean().item(), m.mean().item(), (m > 0).float().mean().item()
    ev = evaluate(); print(f"# init={a.init} obj={a.obj} which={a.which}: held-in eval (first 128 boxes) mean_lb={ev[0]:+.4f} mean_min_lb={ev[1]:+.4f} frac_ver={ev[2]:.3f}", flush=True)
    t0 = time.time(); best = (-1e9, None)
    for step in range(a.steps):
        effs = effective_weights(stock, Gq, Ga, H, dh); load_eff(effs)
        for blk in work.blocks:
            for nm in names: getattr(blk.attn, nm).grad = None
        idx = torch.randint(0, len(x0), (a.batch,)); C = torch.stack([specs_C(int(y))[0] for y in Y[idx]]).to(dev)
        bx = BoundedTensor(x0[idx].to(dev), PerturbationLpNorm(norm=np.inf, x_L=xl[idx].to(dev), x_U=xu[idx].to(dev)))
        if a.debug:
            with torch.autograd.detect_anomaly():
                lb, _ = lirpa.compute_bounds(x=(bx,), method="CROWN", C=C, bound_lower=True, bound_upper=False)
                obj = objective(lb); (-obj).backward()
        else:
            lb, _ = lirpa.compute_bounds(x=(bx,), method="CROWN", C=C, bound_lower=True, bound_upper=False)
            obj = objective(lb); (-obj).backward()
        flat = [t for l in range(L) for t in effs[l]]; grads = [getattr(work.blocks[l].attn, nm).grad for l in range(L) for nm in names]
        assert all(g is not None for g in grads), "no gradient reached the effective weights"
        if a.debug:
            for l in range(L):
                for nm in names:
                    g = getattr(work.blocks[l].attn, nm).grad; print(f"    L{l} {nm}: grad finite={bool(torch.isfinite(g).all())} |g|={g.norm().item():.3e} nan_frac={(~torch.isfinite(g)).float().mean().item():.3f}")
        torch.autograd.backward(flat, grads)
        if a.cond_pen > 0:   # keep G well conditioned
            pen = a.cond_pen * sum((p ** 2).sum() + (torch.linalg.inv(p) ** 2).sum() for p in params); pen.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
        if not torch.isfinite(gn) or not torch.isfinite(obj):
            print(f"  step {step:4d}: NON-FINITE (obj={obj.item()}, grad_norm={gn.item()}) -> skipped", flush=True); opt.zero_grad(); continue
        opt.step(); opt.zero_grad()
        if step % a.log_every == 0 or step == a.steps - 1:
            with torch.no_grad(): cond = max(torch.linalg.cond(p.reshape(-1, dh, dh)).max().item() for p in params)
            ev = evaluate(); print(f"  step {step:4d} batch_obj={obj.item():+.4f} grad_norm={gn.item():.3e} | eval mean_lb={ev[0]:+.4f} mean_min_lb={ev[1]:+.4f} frac_ver={ev[2]:.3f} | max cond(G)={cond:.2f} | {time.time()-t0:.0f}s", flush=True)
            if ev[0] + ev[1] > best[0]: best = (ev[0] + ev[1], (Gq.detach().double().cpu().clone(), Ga.detach().double().cpu().clone()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if best[1] is None: print("# no finite step -> nothing saved"); return
    torch.save({"qk": best[1][0], "av": best[1][1], "args": vars(a)}, a.out); print(f"# saved best gauges -> {a.out}")
    # exactness gate of the learned gauge (float64 vs stock)
    res = gate_variants(onnx_path, verbose=False, variants={"learned": dict(qk_gauge=best[1][0], av_gauge=best[1][1])})
    print(f"# GATE learned gauge: fp64 max|diff| vs stock = {res['learned']:.3e}  (faithful fp32 vs ort = {res['faithful_fp32_vs_ort']:.3e})")

if __name__ == "__main__":
    main()
