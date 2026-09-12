"""Rigorous certificate transfer with the gauge UNFOLDED: G^{-1} and A^{-1} as interval parameters.

Folding the gauge into the attention weights (deept_gauge.fold64) stores W_q' = fl32(G^T W_q) etc.  Once the query/value sides are
rounded, no interval on the key/out-projection sides can make a member of the certified family equal the original network (W_q'
has left the row space of W_q), so intervals on the folded G^{-1} sides alone are NOT a rigorous transfer.  The unfolded rewrite is:

    q' = G^T q            (stock query projection, then an exact constant map: the gauge G is DEFINED as its fp32 matrix)
    k' = G^{-1} k         (stock key projection, then G^{-1} as an INTERVAL parameter containing the true inverse of the fp32 G)
    v' = A^T v            (stock value projection, then the exact constant A)
    out = W_o A^{-T} c'   (A^{-1} as an INTERVAL parameter, then the stock out-projection)

For every G^{-1} in its interval that is the exact inverse, q'.k' = q^T G G^{-1} k = q.k and A^{-T} A^T c = c in real arithmetic, so the
family bounded by CROWN contains an exact rewrite of the original network and the certificate transfers.  The intervals are the two
fp32 neighbours of the fp64 inverse (half-width one ulp of the entry, ~6e-8 relative; the fp64 inverse error kappa*u64 ~ 1e-15 is
seven orders below), or the smallest fp32-representable envelope of +-delta if --delta is given.  auto_LiRPA treats the two
interval matmuls per layer as bilinear nodes (McCormick), which is where the bound loss (if any) comes from.

    deept_unfolded.py --name sst_bert_small_6 --gauge gauges/deept_small6_seed0.pt --max_len 6 --n_sent 2 --iters 4 \
        --eps_list 0.01,0.02,0.03 --ref_json results/deept_small6_wint_smoke.json --save_json results/deept_small6_unf_smoke.json

Sets: stock / gauged (plain fp32 networks; taken from --ref_json when its instances match, else computed), gauged_unf (the unfolded
interval network), stock_unf (identity gauge, G^{-1} = I as an interval: isolates the interval-node overhead).  Per-instance resume
from --save_json (ckpt pre-emption safe).
"""
import sys, os, json, time, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deept_gauge import (run_meta, build, load_data, short_instances, positions, stock_tensors, fold64, load_eff, eye_gauge, make_lirpas, certified_radius,
                         crown_lb, load_gauge, IntervalLinear, box, DeepTNet, load_deept)
from auto_LiRPA import BoundedParameter, PerturbationLpNorm

def blockdiag(m):   # (H, dh, dh) -> (H*dh, H*dh)
    return torch.block_diag(*[m[h] for h in range(m.shape[0])])

def fp32_envelope(t64, delta, inblock, beta):
    """smallest fp32 interval [lo, hi] containing [t64 - delta, t64 + delta] (delta = None -> the two fp32 neighbours of fl32(t64)) on the
    in-block entries; structural off-block zeros get the exact interval [0, 0].  `beta` (per entry, fp64) bounds |true inverse - t64|
    (see inverse_error_bound); the interval is REQUIRED to contain t64 +- beta, so it contains the exact inverse of the fp32 gauge."""
    w = t64.float(); ninf = torch.full_like(w, -float("inf")); pinf = torch.full_like(w, float("inf"))
    if delta is None: lo = torch.nextafter(w, ninf); hi = torch.nextafter(w, pinf)
    else:
        lo = (t64 - delta).float(); hi = (t64 + delta).float()   # round-to-nearest, then push outward where the cast rounded inward
        lo = torch.where(lo.double() > t64 - delta, torch.nextafter(lo, ninf), lo); hi = torch.where(hi.double() < t64 + delta, torch.nextafter(hi, pinf), hi)
    # widen to cover t64 +- beta (the verified inverse error): matters only for entries so small that their fp32 ulp is below beta
    lo2 = (t64 - beta).float(); hi2 = (t64 + beta).float()
    lo2 = torch.where(lo2.double() > t64 - beta, torch.nextafter(lo2, ninf), lo2); hi2 = torch.where(hi2.double() < t64 + beta, torch.nextafter(hi2, pinf), hi2)
    lo = torch.minimum(lo, lo2); hi = torch.maximum(hi, hi2)
    z = ~inblock; lo = torch.where(z, torch.zeros_like(lo), lo); hi = torch.where(z, torch.zeros_like(hi), hi)
    assert bool((t64[z] == 0).all()), "off-block entries must be structural zeros"
    ok = (lo.double() <= t64 - beta) & (t64 + beta <= hi.double()); assert bool(ok[inblock].all()), "interval does not contain the true inverse (t64 +- beta)"
    nz = inblock & (t64 != 0); margin = (torch.minimum(t64 - lo.double(), hi.double() - t64) - beta)[inblock]   # absolute slack beyond the verified error bound
    stats = {"n_interval": int(inblock.sum()), "halfwidth_abs_max": float(((hi - lo).double() / 2)[inblock].max()), "halfwidth_rel_max": float((((hi - lo).double() / 2)[nz] / t64[nz].abs()).max()),
             "margin_abs_min": float(margin.min()), "beta_max": float(beta[inblock].max()), "entry_abs_max": float(t64.abs().max())}
    return w, lo, hi, stats

def inverse_error_bound(G32, Y):
    """verified entrywise bound on |inv(G) - Y| for the fp32 matrix G (as fp64) and its fp64 approximate inverse Y (per head, (H, d, d)):
    with R = I - G Y and rho = ||R||_inf < 1, inv(G) = Y (I - R)^{-1} so |inv(G) - Y| <= ||Y||_inf * rho / (1 - rho) entrywise.  R is
    computed in fp64; its own rounding error is <= d * u64 * ||G||_inf ||Y||_inf per entry, added before taking the norm."""
    G = G32.double(); d = G.shape[-1]; R = torch.eye(d, dtype=torch.float64) - G @ Y; u64 = 2.0 ** -53
    rho = R.abs().sum(-1).max(-1).values + d * u64 * G.abs().sum(-1).max(-1).values * Y.abs().sum(-1).max(-1).values * d   # (H,)
    assert bool((rho < 1e-6).all()), f"inverse residual too large: {rho}"
    beta = Y.abs().sum(-1).max(-1).values * rho / (1 - rho)   # (H,)
    return rho, beta

def unfold_maps(Gq, Ga):
    """per layer (Wt_q, Wt_kinv64, Wt_v, Wt_ainv64, beta_k, beta_a, rho) in the x @ Wt convention: q'_row = q_row @ G (= (G^T q)^T),
    k'_row = k_row @ G^{-T}, v'_row = v_row @ A, c''_row = c'_row @ A^{-1} (= (A^{-T} c')^T).  Gq, Ga: (L, H, dh, dh) fp32 (the gauge as
    defined); inverses in fp64 with a verified entrywise error bound (beta, block-diagonal like the map; off-block = 0)."""
    out = []
    for l in range(Gq.shape[0]):
        G = Gq[l]; A = Ga[l]; Gi = torch.linalg.inv(G.double()); Ai = torch.linalg.inv(A.double())
        rg, bg = inverse_error_bound(G, Gi); ra, ba = inverse_error_bound(A, Ai); H, d = G.shape[0], G.shape[-1]
        bk = blockdiag(bg.view(H, 1, 1).expand(H, d, d)); bav = blockdiag(ba.view(H, 1, 1).expand(H, d, d))
        out.append((blockdiag(G), blockdiag(Gi.transpose(-1, -2)), blockdiag(A), blockdiag(Ai), bk, bav, max(rg.max().item(), ra.max().item())))
    return out

def install_unfolded(net, Gq, Ga, interval=True, delta=None, dtype=torch.float32):
    """wrap the stock attention projections with the unfolded gauge maps.  interval=True: G^{-1}, A^{-1} as BoundedParameters (fp32
    envelopes verified to contain the exact inverses, see fp32_envelope / inverse_error_bound); interval=False: plain parameters in
    `dtype` (fp64 exactness check).  Returns interval stats."""
    dev = next(net.parameters()).device; net._orig_attn = []; stats = []; H, d = Gq.shape[1], Gq.shape[-1]
    inblock = blockdiag(torch.ones(H, d, d, dtype=torch.bool))
    def const(Wt): return nn.Parameter(Wt.to(dtype).unsqueeze(0).to(dev), requires_grad=False)
    def ival(Wt64, beta):
        if not interval: return const(Wt64)
        w, lo, hi, st = fp32_envelope(Wt64, delta, inblock, beta); stats.append(st)
        return BoundedParameter(w.unsqueeze(0).to(dev), PerturbationLpNorm(norm=np.inf, x_L=lo.unsqueeze(0).to(dev), x_U=hi.unsqueeze(0).to(dev)), requires_grad=False)
    rho_max = 0.0
    for layer, (Wq, Ki, Wv, Ai, bk, bav, rho) in zip(net.layers, unfold_maps(Gq, Ga)):
        at = layer.attention.self; od = layer.attention.output; net._orig_attn.append((at.query, at.key, at.value, od.dense)); rho_max = max(rho_max, rho)
        at.query = nn.Sequential(at.query, IntervalLinear(const(Wq), None)); at.key = nn.Sequential(at.key, IntervalLinear(ival(Ki, bk), None))
        at.value = nn.Sequential(at.value, IntervalLinear(const(Wv), None)); od.dense = nn.Sequential(IntervalLinear(ival(Ai, bav), None), od.dense)
    if not stats: return None
    return {"n_interval": sum(s["n_interval"] for s in stats), "halfwidth_abs_max": max(s["halfwidth_abs_max"] for s in stats), "halfwidth_rel_max": max(s["halfwidth_rel_max"] for s in stats),
            "margin_abs_min": min(s["margin_abs_min"] for s in stats), "beta_max": max(s["beta_max"] for s in stats), "rho_max": rho_max, "entry_abs_max": max(s["entry_abs_max"] for s in stats)}

def remove_unfolded(net):
    for l, (q, k, v, o) in zip(net.layers, net._orig_attn): l.attention.self.query, l.attention.self.key, l.attention.self.value, l.attention.output.dense = q, k, v, o
    net._orig_attn = []

def exactness_check(name, Gq32, Ga32, boxes, n_pts=32, seed=0):
    """fp64 forward of the unfolded network (exact fp64 inverses, fp32-defined gauge) vs the original, and vs the fp32-folded network
    (fold64 rounding), over random points in the boxes.  The first must be at fp64 round-off; the second at fp32 round-off (~1e-9)."""
    m0, _ = load_deept(name); n0 = DeepTNet(m0).double().eval(); m1, _ = load_deept(name); n1 = DeepTNet(m1).double().eval(); m2, _ = load_deept(name); n2 = DeepTNet(m2).double().eval()
    H, dh = n0.H, n0.dh; install_unfolded(n1, Gq32, Ga32, interval=False, dtype=torch.float64)
    load_eff(n2, fold64([[t.double() for t in w] for w in stock_tensors(n2)], Gq32.double(), Ga32.double(), H, dh))   # fp32-folded weights, run in fp64
    gen = torch.Generator().manual_seed(seed); w_unf = w_fold = 0.0
    with torch.no_grad():
        for e, i, eps in boxes:
            xl, xu = box(e.double(), i, eps); u = torch.rand((n_pts,) + xl.shape[1:], generator=gen, dtype=torch.float64); x = xl + u * (xu - xl)
            y0 = n0(x); w_unf = max(w_unf, (y0 - n1(x)).abs().max().item()); w_fold = max(w_fold, (y0 - n2(x)).abs().max().item())
    return w_unf, w_fold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sst_bert_small_6"); ap.add_argument("--gauge", required=True); ap.add_argument("--data", default="sst"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=6); ap.add_argument("--n_sent", type=int, default=2); ap.add_argument("--k_words", type=int, default=1)
    ap.add_argument("--eps_list", default="0.01,0.02,0.03"); ap.add_argument("--hi", type=float, default=0.1); ap.add_argument("--iters", type=int, default=4); ap.add_argument("--softmax", default="lse")
    ap.add_argument("--delta", default="ulp", help="'ulp' = two fp32 neighbours of the fp64 inverse; a float = fp32 envelope of +-delta (absolute)")
    ap.add_argument("--sets", default="gauged_unf,stock_unf", help="interval sets to run (comma list of gauged_unf, stock_unf)")
    ap.add_argument("--rad_mode", default="bisect", help="bisect: full bisection for the interval sets; certify: test the plain set's radius, step down the bisection grid on failure (lower bound)")
    ap.add_argument("--ref_json", default=None, help="JSON of an earlier deept_gauge.py eval / this script: plain stock/gauged (and any other complete set) taken from it when the instances match")
    ap.add_argument("--save_json", default=None); ap.add_argument("--subset_len", type=int, default=0, help="keep only instances with <= this many tokens (0 = all)")
    ap.add_argument("--n_check", type=int, default=24, help="instances for the fp64 exactness check"); ap.add_argument("--confirm", type=int, default=0, help="1: only recompute the interval-net lb at each recorded radius of the finished --sets in --save_json")
    a = ap.parse_args(); assert a.k_words == 1, "one-word spec only"
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); L, H, dh = len(net.layers), net.H, net.dh; st = [[t.to(dev) for t in w] for w in stock_tensors(net)]
    data = load_data(a, "test"); S = short_instances(net, m, tok, data, a.max_len, a.n_sent, seed=a.seed)
    inst = [(j, i, e, ex["label"], toks) for j, ex, e, toks in S for i in positions(toks)]
    if a.subset_len: inst = [x for x in inst if x[2].shape[1] <= a.subset_len]
    keys = [(j, i, e.shape[1], y) for j, i, e, y, _ in inst]; eps_list = [float(x) for x in a.eps_list.split(",")]
    print(f"# {a.name}: {len(S)} test sentences <= {a.max_len} tokens, {len(inst)} (sentence, position) instances" + (f" (<= {a.subset_len} tokens)" if a.subset_len else "") + f"; eps {eps_list}; bisection hi {a.hi} iters {a.iters}; device {dev}", flush=True)
    Gq64, Ga64 = load_gauge(a.gauge, L, H, dh); Gq32, Ga32 = Gq64.float(), Ga64.float()   # the gauge IS its fp32 matrix
    delta = None if a.delta == "ulp" else float(a.delta)
    # exactness in fp64: unfolded (exact inverses) vs original must be at round-off; the fp32-folded network shows what the plain gauged set actually bounds
    w_unf, w_fold = exactness_check(a.name, Gq32, Ga32, [(e, i, eps_list[0]) for j, i, e, y, _ in inst[:a.n_check]])
    print(f"# fp64 exactness over {min(a.n_check, len(inst))} boxes x 32 points at eps {eps_list[0]}: |original - unfolded| = {w_unf:.2e} (must be round-off); |original - fp32-folded| = {w_fold:.2e}", flush=True)
    assert w_unf < 1e-12, "unfolded network is not an exact rewrite (block-diagonal orientation?)"
    # sets: {tag: {"rad": [...], "fixed": {eps_str: [...]}, "done": bool, "calls": [...]}}
    def load_sets(path):
        out = {}
        if not path or not os.path.exists(path): return out
        d = json.load(open(path)); ref_keys = [tuple(k) for k in d["inst"]]; pos = {k: n for n, k in enumerate(ref_keys)}
        if not all(k in pos for k in keys): print(f"# {path}: instances differ, ignored", flush=True); return out
        idx = [pos[k] for k in keys]
        for t in [k[:-4] for k in d if k.endswith("_rad")]:
            if all(t in d["fixed"].get(str(eps), {}) for eps in eps_list):
                out[t] = {"rad": [d[f"{t}_rad"][n] for n in idx], "fixed": {str(eps): [d["fixed"][str(eps)][t][n] for n in idx] for eps in eps_list}, "done": True, "calls": []}
                if f"{t}_rad_lb" in d and idx == list(range(len(keys))): out[t]["rad_lb"] = d[f"{t}_rad_lb"]   # (possibly partial) confirmation pass of THIS instance list
        for t, s in d.get("partial", {}).items():
            if idx == list(range(len(keys))) and s.get("inst_n") == len(keys): out[t] = s   # in-progress set of THIS instance list
        return out
    res = {}
    for path in (a.ref_json, a.save_json):
        for t, s in load_sets(path).items():
            if t not in res or (not res[t]["done"] and s["done"]): res[t] = s; state = "complete" if s["done"] else "partial (%d instances)" % len(s["rad"]); print(f"# set '{t}' {state} from {path}", flush=True)
    def save():
        if not a.save_json: return
        d = {"inst": keys, "gauge": a.gauge, "delta": a.delta, "rad_mode": a.rad_mode, "fixed": {str(eps): {} for eps in eps_list}, "partial": {}, "meta": res.get("_meta", {})}
        for t, s in res.items():
            if t == "_meta": continue
            if s["done"]:
                d[f"{t}_rad"] = s["rad"]; [d["fixed"][str(eps)].__setitem__(t, s["fixed"][str(eps)]) for eps in eps_list]
                if "rad_lb" in s: d[f"{t}_rad_lb"] = s["rad_lb"]
            else: d["partial"][t] = {**s, "inst_n": len(keys)}
        d["run"] = run_meta(); json.dump(d, open(a.save_json + ".tmp", "w")); os.replace(a.save_json + ".tmp", a.save_json)
    def run_set(tag, lirpas, ref_tag=None):
        s = res.setdefault(tag, {"rad": [], "fixed": {str(eps): [] for eps in eps_list}, "done": False, "calls": []}); t0 = time.time(); n0 = len(s["rad"])
        for n in range(n0, len(inst)):
            j, i, e, y, _ = inst[n]; lp = lirpas[e.shape[1]]; t1 = time.time(); calls = 0
            if a.rad_mode == "certify" and ref_tag in res and res[ref_tag]["done"]:
                r = res[ref_tag]["rad"][n]; g = a.hi / 2 ** a.iters; step = g; lb_r = crown_lb(lp, e, i, r, y, dev); calls += 1
                while r > 0 and not (lb_r > 0): r = max(0.0, r - step); step *= 2; lb_r = crown_lb(lp, e, i, r, y, dev); calls += 1   # NaN counts as a failure
                rad = r
            else: rad = certified_radius(lp, e, i, y, dev, hi=a.hi, iters=a.iters); calls = a.iters + 1; lb_r = crown_lb(lp, e, i, rad, y, dev) if rad > 0 else float("nan"); calls += 1
            s["rad"].append(rad); s.setdefault("rad_lb", []).append(lb_r)
            for eps in eps_list: s["fixed"][str(eps)].append(crown_lb(lp, e, i, eps, y, dev)); calls += 1
            s["calls"].append(calls); save()
            lbs = "".join(", lb@%g %+.4f" % (eps, s["fixed"][str(eps)][-1]) for eps in eps_list)
            print(f"  [{tag}] inst {n + 1}/{len(inst)} (sent {j} pos {i} len {e.shape[1]}): radius {rad:.5f}{lbs}  {calls} calls {time.time() - t1:.0f}s", flush=True)
        s["done"] = True; save(); rad = np.array(s["rad"]); fx = {eps: np.array(s["fixed"][str(eps)]) for eps in eps_list}; cpi = np.mean(s["calls"][n0:]) if len(inst) > n0 else 0.0
        print(f"# {tag}: certified radius mean {rad.mean():.4f} median {np.median(rad):.4f} | " + "; ".join(f"eps {eps}: verified {(v > 0).sum()}/{len(inst)} (nan {np.isnan(v).sum()}) mean lb {np.nanmean(v):+.4f}" for eps, v in fx.items()) + f"  [{time.time() - t0:.0f}s for {len(inst) - n0} instances, {cpi:.1f} calls/inst]", flush=True)
    lengths = [e.shape[1] for _, _, e, _, _ in inst]
    if a.confirm:   # recompute the interval network's lb at every recorded radius of the finished interval sets (NaN-safe evidence, one call per instance)
        load_eff(net, fold64(st, *eye_gauge(L, H, dh, torch.float64), H, dh))
        for tag in a.sets.split(","):
            s = res[tag]; assert s["done"], tag; gq, ga = (Gq32, Ga32) if tag == "gauged_unf" else eye_gauge(L, H, dh, torch.float32)
            stats = install_unfolded(net, gq, ga, interval=True, delta=delta); lirpas = make_lirpas(net, lengths, dev, a.softmax); lbs = s.setdefault("rad_lb", []); t0 = time.time()
            for n in range(len(lbs), len(inst)):
                j, i, e, y, _ = inst[n]; lbs.append(crown_lb(lirpas[e.shape[1]], e, i, s["rad"][n], y, dev) if s["rad"][n] > 0 else float("nan")); save()
                if (n + 1) % 25 == 0: print(f"  [{tag} confirm] {n + 1}/{len(inst)}  {time.time() - t0:.0f}s", flush=True)
            v = np.array(lbs); r = np.array(s["rad"]); print(f"# {tag} CONFIRM: lb at the recorded radius finite and > 0 on {int(((v > 0) & np.isfinite(v)).sum())}/{int((r > 0).sum())} instances with radius > 0 (NaN {int(np.isnan(v[r > 0]).sum())}, <= 0 {int((v[r > 0] <= 0).sum())}); min lb {np.nanmin(v[r > 0]) if (r > 0).any() else float('nan'):+.3e}  [{time.time() - t0:.0f}s]", flush=True)
            del lirpas; remove_unfolded(net); torch.cuda.empty_cache()
        return
    # plain sets (fp32-folded weights, as in deept_gauge.py eval)
    for tag, (gq, ga) in (("stock", eye_gauge(L, H, dh, torch.float64)), ("gauged", (Gq64, Ga64))):
        if res.get(tag, {}).get("done"): continue
        load_eff(net, fold64(st, gq, ga, H, dh)); lirpas = make_lirpas(net, lengths, dev, a.softmax); run_set(tag, lirpas); del lirpas; torch.cuda.empty_cache()
    load_eff(net, fold64(st, *eye_gauge(L, H, dh, torch.float64), H, dh))   # back to the stock weights: the unfolded sets wrap the STOCK projections
    with torch.no_grad(): stock_logits = [net(e.to(dev)) for _, _, e, _, _ in inst[:a.n_check]]
    for tag in a.sets.split(","):
        if res.get(tag, {}).get("done"): continue
        gq, ga = (Gq32, Ga32) if tag == "gauged_unf" else eye_gauge(L, H, dh, torch.float32)
        stats = install_unfolded(net, gq, ga, interval=True, delta=delta); res.setdefault("_meta", {})[tag] = stats
        print(f"# {tag}: {stats['n_interval']} interval entries (G^-1, A^-1 blocks; |entry| <= {stats['entry_abs_max']:.3f}); half-width max {stats['halfwidth_abs_max']:.2e} absolute, {stats['halfwidth_rel_max']:.2e} relative; VERIFIED inclusion of the exact inverse: residual rho <= {stats['rho_max']:.2e}, entrywise error beta <= {stats['beta_max']:.2e}, slack beyond beta >= {stats['margin_abs_min']:.2e}", flush=True)
        with torch.no_grad():   # fp32 forward sanity: unfolded fp32 network vs the stock fp32 network on the box centres (fp32 arithmetic noise expected)
            worst = max((net(e.to(dev)) - stock_logits[n]).abs().max().item() for n, (_, _, e, _, _) in enumerate(inst[:a.n_check]))
        print(f"# {tag}: fp32 forward |unfolded - stock| on {min(a.n_check, len(inst))} box centres = {worst:.2e}", flush=True)
        lirpas = make_lirpas(net, lengths, dev, a.softmax); run_set(tag, lirpas, ref_tag=tag[:-4]); del lirpas; remove_unfolded(net); torch.cuda.empty_cache()
    # paired summaries
    pairs = [(t, b) for t, b in (("gauged", "stock"), ("gauged_unf", "gauged"), ("stock_unf", "stock"), ("gauged_unf", "stock"), ("gauged_wint", "gauged"), ("gauged_unf", "gauged_wint"), ("stock_wint", "stock")) if res.get(t, {}).get("done") and res.get(b, {}).get("done")]
    for t, b in pairs:
        rt, rb = np.array(res[t]["rad"]), np.array(res[b]["rad"]); dr = rt - rb
        print(f"# PAIRED radius {t}-{b} over {len(dr)} instances: larger on {(dr > 0).sum()}, smaller on {(dr < 0).sum()}, equal {(dr == 0).sum()}; mean rel change {np.mean(dr / np.maximum(rb, 1e-9)):+.4f}; mean radius {rb.mean():.4f} -> {rt.mean():.4f}; max |delta| {np.abs(dr).max():.2e}")
        for eps in eps_list:
            s_, g_ = np.array(res[b]["fixed"][str(eps)]), np.array(res[t]["fixed"][str(eps)]); d = g_ - s_
            print(f"# PAIRED {t}-{b} eps {eps}: lb tighter on {(d > 0).sum()}/{len(d)}, looser {(d < 0).sum()}, mean delta {np.nanmean(d):+.2e}, max |delta| {np.nanmax(np.abs(d)) if np.isfinite(d).any() else float('nan'):.2e}; verified {(s_ > 0).sum()} -> {(g_ > 0).sum()}; flips unverified->verified {((s_ <= 0) & (g_ > 0)).sum()}, verified->unverified {((s_ > 0) & (g_ <= 0)).sum()}")
    save()

if __name__ == "__main__": main()
