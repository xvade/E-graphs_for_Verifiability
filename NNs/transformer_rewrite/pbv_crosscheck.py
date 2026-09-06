"""Cross-check: take the exact (sentence, position) instances of a PBVerification run (their res.json) and compute the
auto_LiRPA CROWN certified radius for the same instances, stock vs gauged, so that the two verifiers can be compared on
identical instances.

    python pbv_crosscheck.py --name sst_bert_small_3 --gauge gauges/deept_small3_seed0.pt \
        --pbv_stock results/pbv_s3_origin_stock.json --pbv_gauged results/pbv_s3_origin_gauged.json --save_json results/pbv_s3_crosscheck.json
"""
import argparse, json, os, sys, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deept_gauge import build, load_sst, embed, stock_tensors, effective, load_eff, eye_gauge, load_gauge, make_lirpas, certified_radius

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--gauge", required=True); ap.add_argument("--pbv_stock", required=True)
ap.add_argument("--pbv_gauged", required=True); ap.add_argument("--hi", type=float, default=0.1); ap.add_argument("--iters", type=int, default=10); ap.add_argument("--save_json", default=None)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); L, H, dh = len(net.layers), net.H, net.dh
st = [[t.to(dev) for t in w] for w in stock_tensors(net)]
PS, PG = json.load(open(a.pbv_stock)), json.load(open(a.pbv_gauged))
data = load_sst("test"); by_tokens = {}
for j, ex in enumerate(data):
    e, toks = embed(m, tok, ex); by_tokens.setdefault(" ".join(toks), (j, ex, e, toks))
inst = []   # (example idx in their file, position, e, label, their stock radius, their gauged radius)
for si, (exs, exg) in enumerate(zip(PS["examples"], PG["examples"])):
    key = " ".join(exs["tokens"]); assert key == " ".join(exg["tokens"])
    if key not in by_tokens: print(f"# WARNING: sentence {si} not found in test set: {key[:60]}"); continue
    j, ex, e, toks = by_tokens[key]; assert ex["label"] == int(exs["label"])
    for bs, bg in zip(exs["bounds"], exg["bounds"]):
        assert bs["position"] == bg["position"]; inst.append((si, bs["position"], e, ex["label"], bs["eps"], bg["eps"]))
print(f"# {a.name}: {len(inst)} instances matched from {a.pbv_stock}", flush=True)
lirpas = make_lirpas(net, sorted({e.shape[1] for _, _, e, _, _, _ in inst}), dev)
Gq, Ga = load_gauge(a.gauge, L, H, dh); res = {}
for tag, (gq, ga) in [("stock", eye_gauge(L, H, dh, torch.float64)), ("gauged", (Gq, Ga))]:
    load_eff(net, effective(st, gq.float().to(dev), ga.float().to(dev), H, dh)); t0 = time.time()
    res[tag] = np.array([certified_radius(lirpas[e.shape[1]], e, i, y, dev, hi=a.hi, iters=a.iters) for _, i, e, y, _, _ in inst])
    print(f"# auto_LiRPA CROWN {tag}: mean radius {res[tag].mean():.5f}  [{time.time()-t0:.0f}s]", flush=True)
s, g = res["stock"], res["gauged"]; ts = np.array([x[4] for x in inst]); tg = np.array([x[5] for x in inst])
def line(name, s, g):
    d = g - s; print(f"# {name}: n={len(s)} mean {s.mean():.5f} -> {g.mean():.5f} ({100*(g.mean()/s.mean()-1):+.1f}% ratio of means); larger {(d>0).sum()}, smaller {(d<0).sum()}, equal {(d==0).sum()}; median rel {100*np.median(g/np.maximum(s,1e-12)-1):+.1f}%")
line("auto_LiRPA CROWN (same instances)", s, g); line("PBVerification (their file)", ts, tg)
print(f"# stock radii, theirs vs auto_LiRPA: mean {ts.mean():.5f} vs {s.mean():.5f}; per-instance ratio theirs/ours median {np.median(ts/np.maximum(s,1e-12)):.3f}")
for (si, i, e, y, a_s, a_g), r_s, r_g in zip(inst, s, g): print(f"  ex {si:2d} pos {i} len {e.shape[1]:2d}: theirs {a_s:.5f}->{a_g:.5f} ({100*(a_g/a_s-1):+.1f}%) | auto_LiRPA {r_s:.5f}->{r_g:.5f} ({100*(r_g/max(r_s,1e-12)-1):+.1f}%)")
if a.save_json: json.dump({"inst": [(si, i, e.shape[1], y) for si, i, e, y, _, _ in inst], "lirpa_stock": s.tolist(), "lirpa_gauged": g.tolist(), "pbv_stock": ts.tolist(), "pbv_gauged": tg.tolist()}, open(a.save_json, "w"))
