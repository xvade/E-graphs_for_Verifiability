"""Is a SIGN-FLIP diagonal gauge CROWN-neutral?  Plain CROWN relaxes x.y with planes through the corner x = x_l (lower: (x_l, y_l); upper:
(x_l, y_u)); negating a coordinate (G = diag(+-1), an exact rewrite) moves that shared corner to x_u, so the plain-CROWN bound should MOVE,
while CROWN-Optimized (alpha interpolates the two corners) should not.  Also: do the per-class learned gauges keep their advantage over the
single learned gauge under CROWN-Optimized?  Runs plain CROWN and CROWN-Optimized on a few test boxes at fixed eps for: identity, the
single / label-0 / label-1 learned gauges, and 3 random diag(+-1) gauges (both sides).  Run from NNs/transformer_rewrite on a GPU.
  python diagnostics/_sign_flip_test.py <name> <max_len> <n_sent> <eps>"""
import sys, time, gc, torch, numpy as np; args = sys.argv[1:]; sys.argv = ["x"]; sys.path.insert(0, "."); import deept_gauge as g
name, max_len, n_sent, eps = args[0], int(args[1]), int(args[2]), float(args[3]); dev = "cuda"; m, tok, net = g.build(name, dev); L, H, dh = len(net.layers), net.H, net.dh
st = [[t.to(dev) for t in w] for w in g.stock_tensors(net)]; data = g.load_sst("test"); S = g.short_instances(net, m, tok, data, max_len, n_sent, seed=0)
inst = [(j, i, e, ex["label"]) for j, ex, e, toks in S for i in g.positions(toks)]; print(f"# {name}: {len(inst)} instances <= {max_len} tokens, eps {eps}", flush=True)
short = {"small6": "small6", "small_6": "small6", "big_3": "big3"}; key = "small6" if "small_6" in name else "big3"
zoo = {"identity": g.eye_gauge(L, H, dh, torch.float64)}
for t in ("", "lab0", "lab1"): zoo["learned" + (("_" + t) if t else "")] = g.load_gauge(f"gauges/deept_{key}_{t + '_' if t else ''}seed0.pt", L, H, dh)
gen = torch.Generator().manual_seed(0)
for r in range(3):
    sq = torch.randint(0, 2, (L, H, dh), generator=gen).double() * 2 - 1; sa = torch.randint(0, 2, (L, H, dh), generator=gen).double() * 2 - 1
    zoo[f"sign{r}"] = (torch.diag_embed(sq), torch.diag_embed(sa))
zoo["scale"] = (torch.diag_embed(torch.rand(L, H, dh, generator=gen).double() * 1.5 + 0.5), torch.diag_embed(torch.rand(L, H, dh, generator=gen).double() * 1.5 + 0.5))   # positive diagonal: must be neutral in both tiers
res = {}
for nm, (gq, ga) in zoo.items():
    g.load_eff(net, g.fold64(st, gq, ga, H, dh)); t0 = time.time(); rows = []
    for (j, i, e, y) in inst:
        lp = g.make_lirpas(net, [e.shape[1]], dev, "lse", alpha=True)[e.shape[1]]
        c = g.crown_lb(lp, e, i, eps, y, dev); v = g.crown_lb(lp, e, i, eps, y, dev, method="CROWN-Optimized", grad=True); rows.append((c, v)); del lp; gc.collect(); torch.cuda.empty_cache()
    res[nm] = np.array(rows); c, v = res[nm][:, 0], res[nm][:, 1]
    print(f"# {nm:14s}: plain CROWN mean lb {np.nanmean(c):+.4f} verified {(c > 0).sum()} | CROWN-Optimized mean lb {np.nanmean(v):+.4f} verified {(v > 0).sum()}  [{time.time()-t0:.0f}s]", flush=True)
b = res["identity"]
for nm in zoo:
    if nm == "identity": continue
    d = res[nm] - b
    print(f"# {nm:14s} vs identity: plain delta mean {np.nanmean(d[:, 0]):+.4f} max|.| {np.nanmax(np.abs(d[:, 0])):.2e} (tighter {(d[:, 0] > 1e-6).sum()} looser {(d[:, 0] < -1e-6).sum()}) | alpha delta mean {np.nanmean(d[:, 1]):+.4f} max|.| {np.nanmax(np.abs(d[:, 1])):.2e} (tighter {(d[:, 1] > 1e-4).sum()} looser {(d[:, 1] < -1e-4).sum()})", flush=True)
for y in (0, 1):
    mk = np.array([x[3] == y for x in inst])
    if mk.sum() == 0: continue
    print(f"# label {y} ({mk.sum()} inst): " + " | ".join(f"{nm} plain {np.nanmean(res[nm][mk, 0]):+.3f} alpha {np.nanmean(res[nm][mk, 1]):+.3f}" for nm in ("identity", "learned", "learned_lab0", "learned_lab1")), flush=True)
import json; json.dump({"inst": [(j, i, e.shape[1], y) for j, i, e, y in inst], "res": {k: v.tolist() for k, v in res.items()}}, open(f"results/sign_flip_{key}.json", "w")); print("SIGN_DONE")
