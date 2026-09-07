"""Towards a FORMULA for the attention gauge (goal 2026-09-07): can the learned per-head gauges be derived from the weights?

Model of what CROWN pays for.  For a scalar product x*y with operand intervals of widths w(x), w(y), the McCormick planes
auto_LiRPA uses have a maximal gap proportional to w(x)*w(y).  In head (l,h) the scores are s = sum_c q'_c k'_c with
q' = G^T q, k' = G^-1 k, and the widths of q'_c, k'_c over the box are widths of LINEAR functionals of the layer input x_l.  If
the layer input varies over the box as x_l = centre + M_l z with z in the unit l_inf ball (M_l = the "box shape"), the width of
a^T x_l is 2||a^T M_l||_1.  So the relaxation slack of head (l,h) is modelled by the SURROGATE
      S_qk(G) = sum_c ||(G^T Wq_h)_c M_l||_p * ||(G^-1 Wk_h)_c M_l||_p,        p = 1 (exact width model) or 2 (heuristic)
and for the value/output side (context c' = P v', v' = Ga^T v, out = c' (Wo Ga^-T)^T)
      S_av(Ga) = sum_c ||(Ga^T Wv_h)_c M_l||_p * ||(Wo_h Ga^-T)_{:,c}||_p .
Both are invariant under diagonal G (row c scales by d_c, the partner by 1/d_c) -- exactly the neutrality CROWN shows -- and change
under mixing.  For p = 2 the minimiser is CLOSED FORM: with A = Wq_h M_l, B = Wk_h M_l the minimum of sum_c ||a_c|| ||b_c|| over
factorisations A^T B = sum_c a_c b_c^T is the nuclear norm, attained by the SVD balancing A' = sqrt(S) U^T, B' = sqrt(S) V^T
(= the ViT R45 gauge when M = I).  For p = 1 it is a 32x32 weight-only optimisation per head.

Box shapes M_l tested: "iso" (M = I, purely the weights) and "jac" (M_l = [2 eps J_l^(t)] over tokens t and boxes: the Jacobian
of the layer input w.r.t. the perturbed embedding row at the box centre -- exact at layer 1 for this model, since its no_var
LayerNorm is linear; first-order at deeper layers; computed on RANDOM-token boxes, so no dataset is involved).

Mode `validate` (the gate): score gauges of KNOWN CROWN quality (identity; learned on SST / Yelp / random tokens; verifier-trained;
the overfit one) under the four surrogates, next to the held-in CROWN metric (mean margin lb at each box's stock radius) on
random-token boxes and on the learner's own 48 SST-dev boxes; ablate the SST gauge (QK-only, AV-only, one layer at a time); and
build + score the four candidate gauges (svd_iso, svd_jac, l1_iso, l1_jac).  A surrogate is credible only if it orders the known
gauges like CROWN does (the overfit gauge must come out worst).

    python gauge_formula.py validate --name sst_bert_small_6 --gauges gauges/deept_small6_seed0.pt,... --out results/formula_small6.json
"""
import argparse, json, os, random, sys, time, math, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deept_gauge import (build, load_data, short_instances, pos_sets, positions, stock_tensors, effective, load_eff, eye_gauge, make_lirpas,
                         certified_radius, crown_lb, load_gauge, load_sst)

def layer_inputs(net, e, i, delta):
    """residual stream entering each layer (L, T, hid) for the embedding e with row i shifted by delta (differentiable in delta)"""
    onehot = torch.zeros_like(e); onehot[0, i] = 1.0
    x = net.ln0(e + onehot * delta); B, n, _ = x.shape; xs = []
    for l in net.layers:
        xs.append(x[0]); a = l.attention.self
        q = a.query(x).view(B, n, net.H, net.dh).transpose(1, 2); k = a.key(x).view(B, n, net.H, net.dh).transpose(1, 2); v = a.value(x).view(B, n, net.H, net.dh).transpose(1, 2)
        p = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(net.dh), dim=-1); c = torch.matmul(p, v).transpose(1, 2).reshape(B, n, net.H * net.dh)
        h = l.attention.output.LayerNorm(l.attention.output.dense(c) + x); x = l.output.LayerNorm(l.output.dense(torch.relu(l.intermediate.dense(h))) + h)
    return torch.stack(xs)

def box_shapes(net, boxes, dev):
    """M_l (hid x N) = [2 eps J_l^(t)] concatenated over boxes and tokens; J = d x_l[t] / d e[i] at the centre (stock network)"""
    hid = net.hid; L = len(net.layers); cols = [[] for _ in range(L)]
    for e, i, y, n, eps in boxes:
        e = e.to(dev); J = torch.autograd.functional.jacobian(lambda d: layer_inputs(net, e, i, d), torch.zeros(hid, device=dev), vectorize=True)   # (L, T, hid, hid_in)
        for l in range(L): cols[l].append((2.0 * eps * J[l]).permute(1, 0, 2).reshape(hid, -1))   # (hid, T*hid_in)
    return [torch.cat(c, 1).double() for c in cols]

def surrogate(Wq, Wk, Wv, Wo, Gq, Ga, M, p, H, dh):
    """per-layer (S_qk, S_av) for one layer's stock (Wq, Wk, Wv, Wo) (nn.Linear convention: Wq (H*dh x hid), Wo (hid x H*dh)), gauges Gq/Ga (H, dh, dh), box shape M (hid x N) or None (= I)"""
    Sqk = torch.zeros((), dtype=torch.float64, device=Wq.device); Sav = torch.zeros((), dtype=torch.float64, device=Wq.device)
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh); G = Gq[h]; A = Ga[h]
        qa = G.T @ Wq[sl]; kb = torch.linalg.inv(G) @ Wk[sl]; va = A.T @ Wv[sl]; ob = Wo[:, sl] @ torch.linalg.inv(A).T          # rows of qa/kb/va (dh x hid); ob (hid_out x dh)
        if M is not None: qa, kb, va = qa @ M, kb @ M, va @ M
        Sqk = Sqk + (qa.norm(p=p, dim=1) * kb.norm(p=p, dim=1)).sum(); Sav = Sav + (va.norm(p=p, dim=1) * ob.norm(p=p, dim=0)).sum()
    return Sqk, Sav

def attn_bound_nodes(lirpa):
    """per layer l: {'q','k','v'} -> the nodes whose intermediate bounds CROWN computes for the head-split projections (the transposes
    feeding QK^T and PV: q, v (1,H,n,dh); k (1,H,dh,n)), found by walking downstream from each projection weight parameter"""
    import re; byname = {n.name: n for n in lirpa.nodes()}; out = {}
    for n in lirpa.nodes():
        mm = re.fullmatch(r"layers\.(\d+)\.attention\.self\.(query|key|value)\.weight", getattr(n, "ori_name", None) or "")
        if not mm: continue
        cur = n
        for _ in range(8):
            if not cur.output_name: cur = None; break
            cur = byname[cur.output_name[0]]
            if isinstance(getattr(cur, "lower", None), torch.Tensor) and cur.lower.dim() == 4: break
        else: cur = None
        if cur is not None: out.setdefault(int(mm.group(1)), {})[mm.group(2)[0]] = cur
    return out

def crown_width_surrogate(net, nodes, H, dh):
    """right after a compute_bounds call: per layer (S_qk, S_av) with the widths CROWN actually derived for q', k', v'
    (summed over tokens) and the live folded W_o columns -- the width-product cost model with NO box-shape approximation"""
    res = []
    for l in range(len(net.layers)):
        d = nodes.get(l, {})
        if not all(k in d for k in "qkv"): res.append((float("nan"), float("nan"))); continue
        wq = (d["q"].upper - d["q"].lower)[0].sum(1); wk = (d["k"].upper - d["k"].lower)[0].sum(2); wv = (d["v"].upper - d["v"].lower)[0].sum(1)   # (H, dh)
        Wo = net.attn_modules()[l][3].weight.detach()                                                                                     # (hid, H*dh)
        res.append(((wq * wk).sum().item(), (wv * Wo.abs().sum(0).view(H, dh)).sum().item()))
    return res

def svd_balance(A, B):
    """closed-form p=2 minimiser: G with G^T A = sqrt(S) U^T for A^T B = U S V^T (A, B: dh x N, full row rank); via the dh x dh problem"""
    Qa, Ra = torch.linalg.qr(A.T); Qb, Rb = torch.linalg.qr(B.T)       # A^T = Qa Ra, B^T = Qb Rb
    U, S, Vh = torch.linalg.svd(Ra @ Rb.T)                                 # A^T B = Qa (Ra Rb^T) Qb^T
    return torch.linalg.solve(Ra, U) @ torch.diag(S.sqrt())                # G = Ra^-1 U sqrt(S)

def candidate_svd(st64, Ms, H, dh):
    """per head: Gq from (Wq_h M, Wk_h M), Ga from (Wv_h M, Wo_h^T)"""
    Gq, Ga = [], []
    for l, (Wq, bq, Wk, bk, Wv, bv, Wo) in enumerate(st64):
        M = Ms[l] if Ms is not None else None; gq, ga = [], []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh); A = Wq[sl] if M is None else Wq[sl] @ M; B = Wk[sl] if M is None else Wk[sl] @ M
            gq.append(svd_balance(A, B)); Av = Wv[sl] if M is None else Wv[sl] @ M; ga.append(svd_balance(Av, Wo[:, sl].T))
        Gq.append(torch.stack(gq)); Ga.append(torch.stack(ga))
    return torch.stack(Gq), torch.stack(Ga)

def candidate_l1(st64, Ms, H, dh, init, steps=400, lr=0.02, cond_pen=1e-4, log=None):
    """p = 1 surrogate minimised by Adam on all heads at once, from `init` (Gq, Ga)"""
    Gq = nn.Parameter(init[0].clone()); Ga = nn.Parameter(init[1].clone()); opt = torch.optim.Adam([Gq, Ga], lr=lr); best = (float("inf"), None)
    for step in range(steps + 1):
        tot = 0.0
        for l, (Wq, bq, Wk, bk, Wv, bv, Wo) in enumerate(st64):
            s1, s2 = surrogate(Wq, Wk, Wv, Wo, Gq[l], Ga[l], Ms[l] if Ms is not None else None, 1, H, dh); tot = tot + s1 + s2
        val = tot.item()
        if val < best[0]: best = (val, (Gq.detach().clone(), Ga.detach().clone()))
        if step == steps: break
        loss = tot + cond_pen * sum((p ** 2).sum() + (torch.linalg.inv(p) ** 2).sum() for p in (Gq, Ga))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([Gq, Ga], 10.0); opt.step()
        if log and step % 100 == 0: print(f"    l1 step {step}: surrogate {val:.4g}", flush=True)
    return best[1], best[0]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["validate"]); ap.add_argument("--name", default="sst_bert_small_6"); ap.add_argument("--gauges", default="")
    ap.add_argument("--n_sent", type=int, default=12); ap.add_argument("--pos", type=int, default=2); ap.add_argument("--max_len", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_sst", type=int, default=48); ap.add_argument("--l1_steps", type=int, default=400); ap.add_argument("--hi", type=float, default=0.1); ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="random"); ap.add_argument("--split", default="dev"); ap.add_argument("--softmax", default="lse")
    a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); L, H, dh, hid = len(net.layers), net.H, net.dh, net.hid
    st = [[t.to(dev) for t in w] for w in stock_tensors(net)]; st64 = [[t.double() for t in w] for w in st]; I64 = eye_gauge(L, H, dh, torch.float64)
    def set_gauge(gq, ga): load_eff(net, effective(st, gq.float().to(dev), ga.float().to(dev), H, dh))
    # ---- boxes: random-token (data-free) and the learner's own SST-dev tuning boxes
    a.data = "random"; R = load_data(a, "dev"); Sr = short_instances(net, m, tok, R, a.max_len, a.n_sent, seed=a.seed); rng = random.Random(a.seed)
    boxes_r = [[e, i, ex["label"], e.shape[1], None] for j, ex, e, toks in Sr for i in pos_sets(toks, 1, rng, a.pos)]
    # the learner's own tuning boxes, rebuilt from the args stored in the first gauge file (cmd_learn's sampling, same seed); [:n_sst] = its eval subset
    gpaths = [p for p in a.gauges.split(",") if p]; ga = torch.load(gpaths[0]).get("args", {}) if gpaths else {}
    la = argparse.Namespace(name=a.name, data=ga.get("data", "auto"), seed=ga.get("seed", 0)); D = load_data(la, ga.get("split", "dev"))
    Ss = short_instances(net, m, tok, D, ga.get("max_len", 8), ga.get("n_sent", 60), seed=la.seed); rng2 = random.Random(la.seed)
    boxes_s = [[e, i, ex["label"], e.shape[1], None] for j, ex, e, toks in Ss for i in pos_sets(toks, ga.get("k_words", 1), rng2, ga.get("pos_per_sent", 3))][:a.n_sst]
    print(f"# learner boxes rebuilt from {gpaths[0] if gpaths else '-'}: data {la.data} split {ga.get('split', 'dev')} max_len {ga.get('max_len', 8)} n_sent {ga.get('n_sent', 60)} pos {ga.get('pos_per_sent', 3)} seed {la.seed}", flush=True)
    lirpas = make_lirpas(net, [b[3] for b in boxes_r + boxes_s], dev, a.softmax); set_gauge(*I64); t0 = time.time()
    for b in boxes_r + boxes_s: b[4] = certified_radius(lirpas[b[3]], b[0], b[1], b[2], dev, hi=a.hi, iters=8)
    print(f"# {a.name}: {len(boxes_r)} random-token boxes (eps = stock radius, mean {np.mean([b[4] for b in boxes_r]):.4f}) + {len(boxes_s)} SST-dev boxes (mean {np.mean([b[4] for b in boxes_s]):.4f})  [{time.time()-t0:.0f}s]", flush=True)
    # ---- box shapes from the random boxes
    t0 = time.time(); Ms = box_shapes(net, [(b[0], b[1], b[2], b[3], b[4]) for b in boxes_r], dev); print(f"# box shapes M_l: {[tuple(M.shape) for M in Ms][0]} per layer  [{time.time()-t0:.0f}s]", flush=True)
    # verifier-free variants: eps := 1 on every box (pure stacked Jacobians; a common eps scale cancels in the SVD, only the per-box
    # RATIOS -- which came from CROWN radii -- are dropped), on the same random centres (Mu) and on a second seed's centres (Mu2)
    Mu = box_shapes(net, [(b[0], b[1], b[2], b[3], 1.0) for b in boxes_r], dev)
    a2 = argparse.Namespace(name=a.name, data="random", seed=a.seed + 1); Sr2 = short_instances(net, m, tok, load_data(a2, "dev"), a.max_len, a.n_sent, seed=a.seed + 1); rng3 = random.Random(a.seed + 1)
    Mu2 = box_shapes(net, [(e, i, ex["label"], e.shape[1], 1.0) for j, ex, e, toks in Sr2 for i in pos_sets(toks, 1, rng3, a.pos)], dev)
    print(f"# uniform-eps box shapes: seed {a.seed} {tuple(Mu[0].shape)}, seed {a.seed + 1} {tuple(Mu2[0].shape)}  [{time.time()-t0:.0f}s]", flush=True)
    # ---- gauge zoo
    zoo = {"identity": I64}
    for pth in gpaths: zoo[os.path.basename(pth).replace("_seed0.pt", "").replace("deept_", "").replace("pbvtrained_", "pbv_")] = load_gauge(pth, L, H, dh)
    first = [k for k in zoo if k != "identity"]
    if first:
        Gq0, Ga0 = zoo[first[0]]; zoo[f"{first[0]}:qk_only"] = (Gq0, I64[1]); zoo[f"{first[0]}:av_only"] = (I64[0], Ga0)
        for l in range(L):
            gq = I64[0].clone(); ga = I64[1].clone(); gq[l] = Gq0[l]; ga[l] = Ga0[l]; zoo[f"{first[0]}:layer{l}"] = (gq, ga)
    # ---- candidates
    t0 = time.time(); zoo["cand:svd_iso"] = candidate_svd(st64, None, H, dh); zoo["cand:svd_jac"] = candidate_svd(st64, Ms, H, dh)
    zoo["cand:svd_jac_u"] = candidate_svd(st64, Mu, H, dh); zoo["cand:svd_jac_u2"] = candidate_svd(st64, Mu2, H, dh)
    print(f"# svd candidates built [{time.time()-t0:.0f}s]; l1 candidates:", flush=True)
    (gq, ga), v = candidate_l1(st64, None, H, dh, zoo["cand:svd_iso"], steps=a.l1_steps, log=True); zoo["cand:l1_iso"] = (gq, ga)
    (gq, ga), v = candidate_l1(st64, Ms, H, dh, zoo["cand:svd_jac"], steps=a.l1_steps, log=True); zoo["cand:l1_jac"] = (gq, ga)
    # ---- score everything
    try: bnodes = {n: attn_bound_nodes(lp) for n, lp in lirpas.items()}
    except Exception as ex: bnodes = {n: {} for n in lirpas}; print(f"# CROWN-width surrogate disabled: node identification failed ({ex})", flush=True)
    print(f"# CROWN-width surrogate: q/k/v bound nodes found for {[len(v) for v in bnodes.values()]} layers per sentence length", flush=True)
    rows = {}; print(f"\n# {'gauge':28s} | surrogate / identity:  l1_iso   l1_jac   l2_iso   l2_jac  (QK+AV; QK, AV for l1_jac) | CROWN-width surrogate / identity (random boxes; sst boxes) | held-in CROWN: random mean lb, frac ver | SST-dev mean lb, frac ver | max cond", flush=True)
    base = {}
    for name, (Gq, Ga) in zoo.items():
        Gq, Ga = Gq.double().to(dev), Ga.double().to(dev); S = {}
        for key, (M, p) in {"l1_iso": (None, 1), "l1_jac": (Ms, 1), "l2_iso": (None, 2), "l2_jac": (Ms, 2)}.items():
            sq = sa = 0.0; per_layer = []
            for l, (Wq, bq, Wk, bk, Wv, bv, Wo) in enumerate(st64):
                s1, s2 = surrogate(Wq, Wk, Wv, Wo, Gq[l], Ga[l], M[l] if M is not None else None, p, H, dh); sq += s1.item(); sa += s2.item(); per_layer.append((s1.item(), s2.item()))
            S[key] = (sq + sa, sq, sa, per_layer)
        if name == "identity": base = {k: v[0] for k, v in S.items()}; base_layers = {k: v[3] for k, v in S.items()}
        set_gauge(Gq, Ga); t0 = time.time(); cw = {"rand": np.zeros((L, 2)), "sst": np.zeros((L, 2))}
        def held_in(bs, key):
            vals = []
            for b in bs:
                vals.append(crown_lb(lirpas[b[3]], b[0], b[1], b[4], b[2], dev))
                try: cw[key] += np.array(crown_width_surrogate(net, bnodes[b[3]], H, dh))
                except Exception as ex: cw[key] += np.nan; print(f"    (crown-width surrogate failed: {ex})", flush=True)
            return np.array(vals)
        vr = held_in(boxes_r, "rand"); vs = held_in(boxes_s, "sst")
        cond = max(torch.linalg.cond(Gq.reshape(-1, dh, dh)).max().item(), torch.linalg.cond(Ga.reshape(-1, dh, dh)).max().item())
        if name == "identity": base_cw = {k: v.sum() for k, v in cw.items()}; base_cw_layers = {k: v.copy() for k, v in cw.items()}
        cwr = {k: cw[k].sum() / base_cw[k] for k in cw}; cwq = {k: cw[k][:, 0].sum() / base_cw[k] for k in cw}; cwa = {k: cw[k][:, 1].sum() / base_cw[k] for k in cw}
        rows[name] = {"surr": {k: v[:3] for k, v in S.items()}, "per_layer_l1_jac": S["l1_jac"][3], "crown_width": {k: v.tolist() for k, v in cw.items()}, "random_lb": float(np.nanmean(vr)), "random_ver": float(np.mean(vr > 0)), "sst_lb": float(np.nanmean(vs)), "sst_ver": float(np.mean(vs > 0)), "cond": cond}
        print(f"  {name:28s} | {S['l1_iso'][0]/base['l1_iso']:7.3f}  {S['l1_jac'][0]/base['l1_jac']:7.3f}  {S['l2_iso'][0]/base['l2_iso']:7.3f}  {S['l2_jac'][0]/base['l2_jac']:7.3f}  (QK {S['l1_jac'][1]/base['l1_jac']:.3f}, AV {S['l1_jac'][2]/base['l1_jac']:.3f}) | CROWN-width rand {cwr['rand']:.3f} (QK {cwq['rand']:.3f}, AV {cwa['rand']:.3f}) sst {cwr['sst']:.3f} (QK {cwq['sst']:.3f}, AV {cwa['sst']:.3f}) | {np.nanmean(vr):+.4f} {np.mean(vr > 0):.2f} | {np.nanmean(vs):+.4f} {np.mean(vs > 0):.2f} | {cond:5.1f}  [{time.time()-t0:.0f}s]", flush=True)
    # per-layer view of the l1_jac surrogate for identity vs the first learned gauge (layer 1 = exact box shape)
    if first:
        print(f"\n# per-layer l1_jac surrogate (QK, AV), identity -> {first[0]}:")
        for l in range(L):
            i0 = base_layers["l1_jac"][l]; g0 = rows[first[0]]["per_layer_l1_jac"][l]
            c0 = base_cw_layers["sst"][l]; c1 = np.array(rows[first[0]]["crown_width"]["sst"][l])
            print(f"  layer {l}: l1_jac QK {i0[0]:.4g} -> {g0[0]:.4g} ({g0[0]/i0[0]:.3f}), AV {i0[1]:.4g} -> {g0[1]:.4g} ({g0[1]/i0[1]:.3f}) | CROWN-width (sst boxes) QK {c0[0]:.4g} -> {c1[0]:.4g} ({c1[0]/c0[0]:.3f}), AV {c0[1]:.4g} -> {c1[1]:.4g} ({c1[1]/c0[1]:.3f})")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump({"rows": rows, "boxes_random": len(boxes_r), "boxes_sst": len(boxes_s)}, open(a.out, "w"), indent=1)
        gd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gauges")
        for k in ("cand:svd_iso", "cand:svd_jac", "cand:svd_jac_u", "cand:svd_jac_u2", "cand:l1_iso", "cand:l1_jac"):
            torch.save({"qk": zoo[k][0].double().cpu(), "av": zoo[k][1].double().cpu(), "formula": k}, os.path.join(gd, f"formula_{a.name}_{k.split(':')[1]}.pt"))
        print(f"# saved candidates to gauges/formula_{a.name}_*.pt and {a.out}")

if __name__ == "__main__": main()
