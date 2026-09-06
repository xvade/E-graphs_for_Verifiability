"""Is the bisection 'certified radius' limited by the CROWN lb crossing zero, or by the lse-softmax NaN cliff?
For a sample of the paired-eval instances: stock and gauged lb at the found radius r (grid lo) and at the next grid point
(r + step): 'zero' if lb(r+step) is finite <= 0, 'nan' if it is NaN.  CPU, run from NNs/transformer_rewrite."""
import sys, json, random, numpy as np, torch; sys.argv = [sys.argv[0]] + sys.argv[1:]; sys.path.insert(0, "."); import deept_gauge as dg
name, res_path, gauge_path, n_sample = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); hi, iters = 0.1, 10; step = hi / 2 ** iters
dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = dg.build(name, dev); L, H, dh = len(net.layers), net.H, net.dh; st = [[t.to(dev) for t in w] for w in dg.stock_tensors(net)]
data = dg.load_sst("test"); S = dg.short_instances(net, m, tok, data, 12, 40, seed=0); emb = {j: (e, ex["label"]) for j, ex, e, toks in S}
d = json.load(open(res_path)); inst = d["inst"]; idx = list(range(len(inst))); random.Random(0).shuffle(idx); idx = idx[:n_sample]
lirpas = dg.make_lirpas(net, [inst[k][2] for k in idx], dev, "lse"); Gq, Ga = dg.load_gauge(gauge_path, L, H, dh)
out = {}
for tag, (gq, ga), rad in [("stock", dg.eye_gauge(L, H, dh, torch.float64), d["stock_rad"]), ("gauged", (Gq, Ga), d["gauged_rad"])]:
    dg.load_eff(net, dg.effective(st, gq.float().to(dev), ga.float().to(dev), H, dh)); rows = []
    for k in idx:
        j, i, n, y = inst[k]; e, lab = emb[j]; r = rad[k]
        lb_r = dg.crown_lb(lirpas[n], e, i, r, lab, dev) if r > 0 else float("nan"); lb_n = dg.crown_lb(lirpas[n], e, i, r + step, lab, dev) if r < hi else float("nan")
        kind = "cap" if r >= hi else ("zero" if np.isfinite(lb_n) and lb_n <= 0 else ("nan" if np.isnan(lb_n) else "?"))
        rows.append((k, r, lb_r, lb_n, kind)); print(f"  {tag} inst {k} len {n} r {r:.4f}: lb(r) {lb_r:+.3f} lb(r+step) {lb_n:+.3f} -> {kind}", flush=True)
    kinds = [x[4] for x in rows]; out[tag] = rows
    print(f"# {name} {tag}: of {len(rows)} sampled instances the radius is limited by lb crossing zero on {kinds.count('zero')}, by the lse NaN cliff on {kinds.count('nan')}, capped on {kinds.count('cap')}, other {kinds.count('?')}; mean lb at radius (finite) {np.nanmean([x[2] for x in rows]):+.3f}", flush=True)
json.dump(out, open(res_path.replace(".json", "_nancliff.json"), "w"))
