#!/usr/bin/env python
"""Attention-gauge rewrite for the GenBaB (Shi et al., TACAS 2025) CIFAR-10 ViTs (genbab_benchmarks/cifar/vit_*).

Same exact rewrite family as vit_model.py, expressed on the PyTorch nn.Linear weights (W is out x in, head h = rows sl):
    Q_h G      : W_q[sl,:] <- G^T W_q[sl,:],   b_q[sl] <- G^T b_q[sl]
    K_h G^{-T} : W_k[sl,:] <- G^-1 W_k[sl,:],  b_k[sl] <- G^-1 b_k[sl]
    V_h Ga     : W_v[sl,:] <- Ga^T W_v[sl,:],  b_v[sl] <- Ga^T b_v[sl]
    W_o        : W_o[:,sl] <- W_o[:,sl] Ga^{-T}
so QK^T and A V W_o are unchanged for any invertible per-head G, Ga (fp64 gate on random points in the boxes).

Subcommands:
  learn  --name vit_2_3 --out gauges/genbab_vit_2_3.pt [--steps 200 --batch 32 --n_train 512 --lr 0.01 --init id]
         gradient ascent on the vanilla CROWN (lse) lower bound over eps-boxes around correctly classified CIFAR-TRAIN
         images (same normalization as the specs; specs are test-set images -> disjoint), objective 'mix' as before.
  eval   --name vit_2_3 [--gauge gauges/genbab_vit_2_3.pt]  vanilla CROWN (lse) on all benchmark instances, stock vs gauged.
  export --name vit_2_3 --gauge ... --tag learnedG   writes genbab_benchmarks/cifar/<name>_<tag>/{model.pth,config.yaml (byte copy),
         instances.csv (copy), specs -> symlink} for the UNMODIFIED official abcrown pipeline; fp64 gate + fp32 storage check.
"""
import sys, os, re, copy, shutil, pickle, time, argparse, importlib.util, numpy as np, torch, torch.nn as nn
REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; GB = os.path.join(REPO, "genbab_benchmarks/cifar")
sys.path.insert(0, os.path.join(REPO, "alpha-beta-CROWN/complete_verifier"))
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
CIF = os.path.join(REPO, "alpha-beta-CROWN/complete_verifier/datasets/cifar-10-batches-py")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1); STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
NAMES = ["query.weight", "query.bias", "key.weight", "key.bias", "value.weight", "value.bias", "out.weight"]

def load_model(name, sd_path=None):
    cls = "ViT_" + name.split("_", 1)[1]
    spec = importlib.util.spec_from_file_location("gbvit", os.path.join(GB, "models/vit.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    net = getattr(mod, cls)(); ck = torch.load(sd_path or os.path.join(GB, name, "model.pth"), map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck; net.load_state_dict(sd); net.eval()   # GenBaB checkpoints wrap state_dict with optimizer/epoch/best/model_cert
    def strip(m):   # eval-mode Dropout is the identity; auto_LiRPA refuses Dropout modules
        for k, c in m.named_children():
            if isinstance(c, nn.Dropout): setattr(m, k, nn.Identity())
            else: strip(c)
    strip(net)
    attn = [blk[0].fn[1] for blk in net[1]]   # MultiHeadAttention per layer
    return net, sd, attn, attn[0].num_heads, attn[0].head_dim, len(attn)

def stock_tensors(attn):
    return [[getattr(getattr(a, n.split(".")[0]), n.split(".")[1]).detach().clone() for n in NAMES] for a in attn]

def effective(stock, Gq, Ga, H, dh):
    """Differentiable exact gauge algebra on nn.Linear weights. stock[l] = [Wq,bq,Wk,bk,Wv,bv,Wo]."""
    out = []
    for l, (Wq, bq, Wk, bk, Wv, bv, Wo) in enumerate(stock):
        q, bq2, k, bk2, v, bv2, o = [], [], [], [], [], [], []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh); G = Gq[l, h]; Gi = torch.linalg.inv(G); A = Ga[l, h]; Ai = torch.linalg.inv(A)
            q.append(G.T @ Wq[sl]); bq2.append(G.T @ bq[sl]); k.append(Gi @ Wk[sl]); bk2.append(Gi @ bk[sl])
            v.append(A.T @ Wv[sl]); bv2.append(A.T @ bv[sl]); o.append(Wo[:, sl] @ Ai.T)
        out.append([torch.cat(q, 0), torch.cat(bq2), torch.cat(k, 0), torch.cat(bk2), torch.cat(v, 0), torch.cat(bv2), torch.cat(o, 1)])
    return out

def load_eff(attn, effs):
    with torch.no_grad():
        for a, e in zip(attn, effs):
            for n, t in zip(NAMES, e): getattr(getattr(a, n.split(".")[0]), n.split(".")[1]).copy_(t)

def gauged_state_dict(name, Gq, Ga):
    net, sd, attn, H, dh, L = load_model(name); st = stock_tensors(attn); load_eff(attn, effective(st, Gq.float(), Ga.float(), H, dh))
    out = copy.deepcopy(sd)
    for l, a in enumerate(attn):
        for n in NAMES: out[f"1.{l}.0.fn.1.{n}"] = getattr(getattr(a, n.split(".")[0]), n.split(".")[1]).detach().clone()
    assert set(out) == set(sd); return out

def fp64_gate(name, Gq, Ga, boxes, n_pts=64, seed=0):
    """max |stock - gauged| in float64 over random points inside the benchmark boxes (the rewrite is exact up to rounding)."""
    net, _, attn, H, dh, L = load_model(name); net = net.double(); g = net.__class__(); g.load_state_dict(net.state_dict()); g = g.double().eval()
    _, _, gattn, _, _, _ = load_model(name); gattn = [blk[0].fn[1] for blk in g[1]]
    load_eff(gattn, effective([[t.double() for t in ws] for ws in stock_tensors(attn)], Gq.double(), Ga.double(), H, dh))
    for m in g.modules():
        if isinstance(m, nn.Dropout): pass
    for k, c in list(g.named_modules()):
        for kk, cc in c.named_children():
            if isinstance(cc, nn.Dropout): setattr(c, kk, nn.Identity())
    gen = torch.Generator().manual_seed(seed); worst = 0.0
    with torch.no_grad():
        for xl, xu, y in boxes:
            u = torch.rand((n_pts,) + xl.shape[1:], generator=gen, dtype=torch.float64); x = xl.double() + u * (xu - xl).double()
            worst = max(worst, (net(x) - g(x)).abs().max().item())
    return worst

def parse_spec(path):
    t = open(path).read(); lab = int(re.search(r"label: (\d+)", t).group(1))
    lb = [float(v) for v in re.findall(r"\(assert \(>= X_\d+ ([-\d.eE]+)\)\)", t)]; ub = [float(v) for v in re.findall(r"\(assert \(<= X_\d+ ([-\d.eE]+)\)\)", t)]
    return torch.tensor(lb).reshape(1, 3, 32, 32), torch.tensor(ub).reshape(1, 3, 32, 32), lab

def bench_boxes(name):
    files = [l.strip() for l in open(os.path.join(GB, name, "instances.csv")) if l.strip()]
    ids = [int(re.search(r"(\d+)\.vnnlib", f).group(1)) for f in files]
    return ids, [parse_spec(os.path.join(GB, name, f)) for f in files]

def specs_C(y):
    C = torch.zeros(9, 10); r = 0
    for i in range(10):
        if i != y: C[r, y] = 1; C[r, i] = -1; r += 1
    return C

def cifar_train_boxes(n, net, dev, seed=0, eps_pix=1 / 255, batch_file="data_batch_1"):
    d = pickle.load(open(os.path.join(CIF, batch_file), "rb"), encoding="bytes")
    X = torch.tensor(d[b"data"]).reshape(-1, 3, 32, 32).float() / 255; Y = torch.tensor(d[b"labels"])
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed)); X, Y = X[perm], Y[perm]
    x0 = (X - MEAN) / STD; xl = ((X - eps_pix).clamp(0, 1) - MEAN) / STD; xu = ((X + eps_pix).clamp(0, 1) - MEAN) / STD
    with torch.no_grad(): pred = torch.cat([net(x0[i:i + 500].to(dev)).argmax(1).cpu() for i in range(0, len(x0), 500)])
    keep = (pred == Y).nonzero().ravel()[:n]
    return x0[keep], xl[keep], xu[keep], Y[keep]

def crown_min_lbs(lirpa, boxes_x0, xl, xu, Y, dev, batch, method="CROWN"):
    mins, widths = [], []
    with torch.no_grad():
        for i in range(0, len(Y), batch):
            idx = slice(i, min(len(Y), i + batch)); C = torch.stack([specs_C(int(y)) for y in Y[idx]]).to(dev)
            bx = BoundedTensor(boxes_x0[idx].to(dev), PerturbationLpNorm(norm=np.inf, x_L=xl[idx].to(dev), x_U=xu[idx].to(dev)))
            lb, ub = lirpa.compute_bounds(x=(bx,), method=method, C=C); mins.append(lb.min(1).values.cpu()); widths.append((ub - lb).mean(1).cpu())
    return torch.cat(mins), torch.cat(widths)

def load_gauge(path, L, H, dh):
    if path is None: I = torch.eye(dh, dtype=torch.float64).expand(L, H, dh, dh).clone(); return I, I.clone()
    g = torch.load(path); return g["qk"], g["av"]

def cmd_eval(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, sd, attn, H, dh, L = load_model(a.name); ids, boxes = bench_boxes(a.name)
    x0 = torch.cat([(l + u) / 2 for l, u, _ in boxes]); xl = torch.cat([l for l, _, _ in boxes]); xu = torch.cat([u for _, u, _ in boxes]); Y = torch.tensor([y for _, _, y in boxes])
    st = stock_tensors(attn); net = net.to(dev); lirpa = BoundedModule(net, torch.empty(1, 3, 32, 32, device=dev), bound_opts={"softmax": a.softmax, "conv_mode": "matrix"}, device=dev)
    res = {}
    for tag, gp in [("stock", None), ("gauged", a.gauge)] if a.gauge else [("stock", None)]:
        Gq, Ga = load_gauge(gp, L, H, dh); load_eff(attn, effective([[t.to(dev) for t in ws] for ws in st], Gq.float().to(dev), Ga.float().to(dev), H, dh))
        m, w = crown_min_lbs(lirpa, x0, xl, xu, Y, dev, a.batch); res[tag] = (m, w)
        print(f"# {a.name} {tag}: vanilla CROWN({a.softmax}) on {len(ids)} benchmark instances: verified {(m > 0).sum().item()}/{len(ids)}, mean min-lb {m.mean():+.4f}, median {m.median():+.4f}, mean width {w.mean():.4f}", flush=True)
    if a.gauge:
        d = res["gauged"][0] - res["stock"][0]; dw = res["gauged"][1] - res["stock"][1]
        print(f"# PAIRED gauged-stock: min-lb tighter on {(d > 0).sum().item()}/{len(d)}, looser {(d < 0).sum().item()}, mean {d.mean():+.4f} worst {d.min():+.4f}; width mean rel {(dw.sum() / res['stock'][1].sum()).item():+.3f}")
        print(f"# fp64 GATE (random points in benchmark boxes): {fp64_gate(a.name, Gq, Ga, boxes):.2e}")
        if a.save_json:
            import json; json.dump({"ids": ids, "stock_min": res["stock"][0].tolist(), "gauged_min": res["gauged"][0].tolist(), "stock_w": res["stock"][1].tolist(), "gauged_w": res["gauged"][1].tolist()}, open(a.save_json, "w"))

def cmd_learn(a):
    torch.manual_seed(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, sd, attn, H, dh, L = load_model(a.name); net = net.to(dev); st = [[t.to(dev) for t in ws] for ws in stock_tensors(attn)]
    for p in net.parameters(): p.requires_grad_(False)
    for at in attn:
        for n in NAMES: getattr(getattr(at, n.split(".")[0]), n.split(".")[1]).requires_grad_(True)
    I = torch.eye(dh).expand(L, H, dh, dh).clone().to(dev)
    Gq = nn.Parameter(I.clone(), requires_grad=a.which in ("both", "qk")); Ga = nn.Parameter(I.clone(), requires_grad=a.which in ("both", "av"))
    x0, xl, xu, Y = cifar_train_boxes(a.n_train, net, dev, seed=a.seed)
    print(f"# {a.name}: H={H} dh={dh} L={L}; tuning boxes: {len(x0)} CIFAR-TRAIN images (correctly classified), eps=1/255; device {dev}", flush=True)
    lirpa = BoundedModule(net, torch.empty(1, 3, 32, 32, device=dev), bound_opts={"softmax": a.softmax, "conv_mode": "matrix"}, device=dev)
    params = [p for p in (Gq, Ga) if p.requires_grad]; opt = torch.optim.Adam(params, lr=a.lr)
    def objective(lb):
        if a.obj == "mean": return lb.mean()
        if a.obj == "min": return lb.min(1).values.mean()
        return 0.5 * lb.mean() + 0.5 * lb.min(1).values.mean()
    def evaluate(n=128):
        n = min(n, len(x0)); load_eff(attn, effective(st, Gq.detach(), Ga.detach(), H, dh))
        m, w = crown_min_lbs(lirpa, x0[:n], xl[:n], xu[:n], Y[:n], dev, a.batch); return m.mean().item(), (m > 0).float().mean().item(), w.mean().item()
    ev = evaluate(); print(f"# init=id obj={a.obj} which={a.which}: held-in eval (first 128 boxes) mean_min_lb={ev[0]:+.4f} frac_ver={ev[1]:.3f} width={ev[2]:.4f}", flush=True)
    t0 = time.time(); best = (ev[0], (Gq.detach().double().cpu().clone(), Ga.detach().double().cpu().clone()), -1)
    for step in range(a.steps):
        effs = effective(st, Gq, Ga, H, dh); load_eff(attn, effs)
        leaves = [getattr(getattr(at, n.split(".")[0]), n.split(".")[1]) for at in attn for n in NAMES]
        for p in leaves: p.grad = None
        idx = torch.randint(0, len(x0), (a.batch,)); C = torch.stack([specs_C(int(y)) for y in Y[idx]]).to(dev)
        bx = BoundedTensor(x0[idx].to(dev), PerturbationLpNorm(norm=np.inf, x_L=xl[idx].to(dev), x_U=xu[idx].to(dev)))
        with torch.autograd.set_detect_anomaly(bool(a.debug)):
            lb, _ = lirpa.compute_bounds(x=(bx,), method="CROWN", C=C, bound_lower=True, bound_upper=False); obj = objective(lb); (-obj).backward()
        grads = [p.grad for p in leaves]; assert all(g is not None for g in grads), "no gradient reached the effective weights"
        if a.debug:
            for p, g, nm in zip(leaves, grads, [f"L{l} {n}" for l in range(L) for n in NAMES]): print(f"    {nm}: finite={bool(torch.isfinite(g).all())} |g|={g.norm().item():.3e}")
        torch.autograd.backward([t for e in effs for t in e], grads)
        if a.cond_pen > 0: (a.cond_pen * sum((p ** 2).sum() + (torch.linalg.inv(p) ** 2).sum() for p in params)).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
        if not torch.isfinite(gn) or not torch.isfinite(obj): print(f"  step {step:4d}: NON-FINITE -> skipped", flush=True); opt.zero_grad(); continue
        opt.step(); opt.zero_grad()
        if step % a.log_every == 0 or step == a.steps - 1:
            with torch.no_grad(): cond = max(torch.linalg.cond(p.reshape(-1, dh, dh)).max().item() for p in params)
            ev = evaluate(); print(f"  step {step:4d} batch_obj={obj.item():+.4f} grad_norm={gn.item():.3e} | eval mean_min_lb={ev[0]:+.4f} frac_ver={ev[1]:.3f} width={ev[2]:.4f} | max cond(G)={cond:.2f} | {time.time()-t0:.0f}s", flush=True)
            if ev[0] > best[0]: best = (ev[0], (Gq.detach().double().cpu().clone(), Ga.detach().double().cpu().clone()), step)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"qk": best[1][0], "av": best[1][1], "args": vars(a), "best_step": best[2]}, a.out); print(f"# saved best gauges (step {best[2]}, held-in mean_min_lb {best[0]:+.4f}) -> {a.out}; wall {time.time()-t0:.0f}s")
    ids, boxes = bench_boxes(a.name); print(f"# fp64 GATE on benchmark boxes: {fp64_gate(a.name, best[1][0], best[1][1], boxes[:16]):.2e}")

def cmd_export(a):
    net, sd, attn, H, dh, L = load_model(a.name); Gq, Ga = load_gauge(a.gauge, L, H, dh); ids, boxes = bench_boxes(a.name)
    out = gauged_state_dict(a.name, Gq, Ga); dst = os.path.join(GB, f"{a.name}_{a.tag}"); os.makedirs(dst, exist_ok=True)
    ck = torch.load(os.path.join(GB, a.name, "model.pth"), map_location="cpu"); ck = {k: (out if k == "state_dict" else v) for k, v in ck.items()} if "state_dict" in ck else out
    torch.save(ck, os.path.join(dst, "model.pth"))   # same checkpoint layout as stock, only the state_dict changed; shutil.copyfile(os.path.join(GB, a.name, "config.yaml"), os.path.join(dst, "config.yaml")); shutil.copyfile(os.path.join(GB, a.name, "instances.csv"), os.path.join(dst, "instances.csv"))
    if not os.path.exists(os.path.join(dst, "specs")): os.symlink(os.path.join("..", a.name, "specs"), os.path.join(dst, "specs"))
    changed = [k for k in out if not torch.equal(out[k], sd[k])]; print(f"# wrote {dst}: {len(changed)}/{len(out)} tensors changed: {changed}")
    print(f"# fp64 GATE (random points in benchmark boxes): {fp64_gate(a.name, Gq, Ga, boxes):.2e}")
    # fp32 storage check: reload both from disk, compare at random points in the boxes (float32 inference, as the pipeline does)
    g, _, _, _, _, _ = load_model(a.name, os.path.join(dst, "model.pth")); gen = torch.Generator().manual_seed(1); worst = 0.0; agree = 0
    with torch.no_grad():
        for xl, xu, y in boxes:
            x = xl + torch.rand((64,) + xl.shape[1:], generator=gen) * (xu - xl); worst = max(worst, (net(x) - g(x)).abs().max().item()); agree += int(net((xl + xu) / 2).argmax() == g((xl + xu) / 2).argmax())
    print(f"# fp32 stored-model discrepancy sup over box samples: {worst:.2e}; center predictions agree {agree}/{len(boxes)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["learn", "eval", "export"]); ap.add_argument("--name", default="vit_2_3")
    ap.add_argument("--gauge", default=None); ap.add_argument("--out", default=None); ap.add_argument("--tag", default="learnedG"); ap.add_argument("--save_json", default=None)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--n_train", type=int, default=512); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--obj", default="mix"); ap.add_argument("--cond_pen", type=float, default=1e-4); ap.add_argument("--softmax", default="lse"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--which", default="both"); ap.add_argument("--clip", type=float, default=1.0); ap.add_argument("--log_every", type=int, default=10); ap.add_argument("--debug", type=int, default=0)
    a = ap.parse_args(); {"learn": cmd_learn, "eval": cmd_eval, "export": cmd_export}[a.cmd](a)
