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

def out_shapes(net, boxes, dev):
    """N_l^out (hid x T*boxes): gradient of the centre margin w.r.t. the attention sub-layer's dense output u_l = W_o c' (per token), stacked
    over boxes -- the functional through which the FINAL bound reads layer l's attention output (first order, at the box centre)"""
    L = len(net.layers); cols = [[] for _ in range(L)]
    for e, i, y, n, eps in boxes:
        with torch.enable_grad():
            x = net.ln0(e.to(dev).clone().requires_grad_(True)); B, nT, _ = x.shape; us = []
            for l in net.layers:
                a = l.attention.self
                q = a.query(x).view(B, nT, net.H, net.dh).transpose(1, 2); k = a.key(x).view(B, nT, net.H, net.dh).transpose(1, 2); v = a.value(x).view(B, nT, net.H, net.dh).transpose(1, 2)
                pr = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(net.dh), dim=-1); c = torch.matmul(pr, v).transpose(1, 2).reshape(B, nT, net.H * net.dh)
                u = l.attention.output.dense(c); u.retain_grad(); us.append(u)
                h = l.attention.output.LayerNorm(u + x); x = l.output.LayerNorm(l.output.dense(torch.relu(l.intermediate.dense(h))) + h)
            logits = net.classifier(torch.tanh(net.pooler.dense(x[:, 0]))); (logits[0, y] - logits[0, 1 - y]).backward()
        for l in range(L): cols[l].append(us[l].grad[0].T.detach())   # (hid, T)
    return [torch.cat(c, 1).double() for c in cols]

def weight_shapes(net, st64, dev):
    """weight-only downstream functionals on u_l (the no_var LayerNorm is linear: LN(u) = Γ P u + β, P = I - 11ᵀ/hid):
    N_ffn = (W_1 Γ_1 P)ᵀ  -- the same layer's FFN pre-activations (ReLU inputs CROWN must bound);
    N_next = ([W_q; W_k; W_v]_{l+1} Γ_2 P Γ_1 P)ᵀ -- the next layer's projections through the residual path (FFN branch skipped); last layer: the pooler"""
    hid = net.hid; P = torch.eye(hid, dtype=torch.float64, device=dev) - torch.ones(hid, hid, dtype=torch.float64, device=dev) / hid; Nf, Nn = [], []
    for l, layer in enumerate(net.layers):
        G1 = torch.diag(layer.attention.output.LayerNorm.weight.detach().double()); G2 = torch.diag(layer.output.LayerNorm.weight.detach().double())
        Nf.append((layer.intermediate.dense.weight.detach().double() @ G1 @ P).T)
        Wn = torch.cat([st64[l + 1][0], st64[l + 1][2], st64[l + 1][4]], 0) if l + 1 < len(net.layers) else net.pooler.dense.weight.detach().double()
        Nn.append((Wn @ G2 @ P @ G1 @ P).T)
    return Nf, Nn

def combine_N(blocks):
    """concatenate functional blocks with equal Frobenius weight (their natural scales are unrelated)"""
    return torch.cat([b / b.norm() for b in blocks], 1)

def surrogate(Wq, Wk, Wv, Wo, Gq, Ga, M, p, H, dh, N=None):
    """per-layer (S_qk, S_av) for one layer's stock (Wq, Wk, Wv, Wo) (nn.Linear convention: Wq (H*dh x hid), Wo (hid x H*dh)), gauges Gq/Ga (H, dh, dh),
    box shape M (hid x N) or None (= I); N (hid x F) = downstream functionals reading the attention output (None = every hidden coordinate, i.e. the l1 column norm)"""
    Sqk = torch.zeros((), dtype=torch.float64, device=Wq.device); Sav = torch.zeros((), dtype=torch.float64, device=Wq.device)
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh); G = Gq[h]; A = Ga[h]
        qa = G.T @ Wq[sl]; kb = torch.linalg.inv(G) @ Wk[sl]; va = A.T @ Wv[sl]; ob = Wo[:, sl] @ torch.linalg.inv(A).T          # rows of qa/kb/va (dh x hid); ob (hid_out x dh)
        if M is not None: qa, kb, va = qa @ M, kb @ M, va @ M
        if N is not None: ob = N.T @ ob                                                                                                  # (F x dh): how each downstream functional reads coordinate c
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

def crown_width_surrogate(net, nodes, H, dh, Ns=None):
    """right after a compute_bounds call: per layer (S_qk, S_av) with the widths CROWN actually derived for q', k', v'
    (summed over tokens) and the live folded W_o columns -- the width-product cost model with NO box-shape approximation"""
    res = []
    for l in range(len(net.layers)):
        d = nodes.get(l, {})
        if not all(k in d for k in "qkv"): res.append((float("nan"), float("nan"), float("nan"))); continue
        wq = (d["q"].upper - d["q"].lower)[0].sum(1); wk = (d["k"].upper - d["k"].lower)[0].sum(2); wv = (d["v"].upper - d["v"].lower)[0].sum(1)   # (H, dh)
        Wo = net.attn_modules()[l][3].weight.detach()                                                                                     # (hid, H*dh)
        avN = (wv * (Ns[l].T.float() @ Wo).abs().sum(0).view(H, dh)).sum().item() if Ns is not None else float("nan")
        res.append(((wq * wk).sum().item(), (wv * Wo.abs().sum(0).view(H, dh)).sum().item(), avN))
    return res

def svd_balance(A, B):
    """closed-form p=2 minimiser: G with G^T A = sqrt(S) U^T for A^T B = U S V^T (A, B: dh x N, full row rank); via the dh x dh problem"""
    d = A.shape[0]   # fewer probe columns than d_h (tiny smoke runs only): append a tiny ridge so the operands have full row rank
    if A.shape[1] < d: A = torch.cat([A, 1e-6 * A.norm() * torch.eye(d, dtype=A.dtype, device=A.device)], 1)
    if B.shape[1] < d: B = torch.cat([B, 1e-6 * B.norm() * torch.eye(d, dtype=B.dtype, device=B.device)], 1)
    Qa, Ra = torch.linalg.qr(A.T); Qb, Rb = torch.linalg.qr(B.T)       # A^T = Qa Ra, B^T = Qb Rb
    U, S, Vh = torch.linalg.svd(Ra @ Rb.T)                                 # A^T B = Qa (Ra Rb^T) Qb^T
    # G = Ra^-1 U Λ: for ANY diagonal Λ the row-norm products are s_c (the diagonal freedom is CROWN-neutral); Λ = sqrt(S) balances the
    # two factors, but a near-zero s_c would make G singular in fp32, so the balancing scale is floored at sqrt(1e-3 · s_max)
    lam = S.clamp_min(1e-3 * S.max()).sqrt()
    return torch.linalg.solve(Ra, U) @ torch.diag(lam)

def candidate_svd(st64, Ms, H, dh, Ns=None):
    """per head: Gq from (Wq_h M, Wk_h M), Ga from (Wv_h M, Wo_h^T N) (N = I: plain columns of W_o)"""
    Gq, Ga = [], []
    for l, (Wq, bq, Wk, bk, Wv, bv, Wo) in enumerate(st64):
        M = Ms[l] if Ms is not None else None; gq, ga = [], []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh); A = Wq[sl] if M is None else Wq[sl] @ M; B = Wk[sl] if M is None else Wk[sl] @ M
            gq.append(svd_balance(A, B)); Av = Wv[sl] if M is None else Wv[sl] @ M; ga.append(svd_balance(Av, Wo[:, sl].T if Ns is None else Wo[:, sl].T @ Ns[l]))
        Gq.append(torch.stack(gq)); Ga.append(torch.stack(ga))
    return torch.stack(Gq), torch.stack(Ga)

def sens_shapes(net, boxes, dev):
    """per layer, per probe box: token weights for the column blocks of M_l (all (H, T)) -- alpha / beta = rank-1 factors of
    |d margin / d score_ij| at the centre (query side / key side: the bound reads score (i, j) this much), gamma_j = sum_i of the
    first-order width of p_ij (the softmax uncertainty that multiplies value token j).  Boxes in the order of box_shapes."""
    L = len(net.layers); hid = net.hid; out = [[] for _ in range(L)]
    def fwd(e_, delta=None, keep=None):
        x = net.ln0(e_ if delta is None else e_ + torch.zeros_like(e_).index_fill_(1, torch.tensor([i], device=dev), 1.0) * delta); B, nT, _ = x.shape; ps = []
        for l in net.layers:
            a = l.attention.self
            q = a.query(x).view(B, nT, net.H, net.dh).transpose(1, 2); k = a.key(x).view(B, nT, net.H, net.dh).transpose(1, 2); v = a.value(x).view(B, nT, net.H, net.dh).transpose(1, 2)
            sc = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(net.dh)
            if keep is not None: sc.retain_grad(); keep.append(sc)
            pr = torch.softmax(sc, dim=-1); ps.append(pr[0]); c = torch.matmul(pr, v).transpose(1, 2).reshape(B, nT, net.H * net.dh)
            h = l.attention.output.LayerNorm(l.attention.output.dense(c) + x); x = l.output.LayerNorm(l.output.dense(torch.relu(l.intermediate.dense(h))) + h)
        return torch.stack(ps), net.classifier(torch.tanh(net.pooler.dense(x[:, 0])))
    for e, i, y, n, eps in boxes:
        e = e.to(dev); scs = []
        with torch.enable_grad():
            _, logits = fwd(e.clone().requires_grad_(True), keep=scs); (logits[0, y] - logits[0, 1 - y]).backward()
        Jp = torch.autograd.functional.jacobian(lambda d: fwd(e, d)[0], torch.zeros(hid, device=dev), vectorize=True)   # (L, H, T, T, hid)
        wp = 2.0 * eps * Jp.abs().sum(-1)                                                                                  # first-order width of p_ij
        for l in range(L):
            g = scs[l].grad[0].abs().double(); U, S, Vh = torch.linalg.svd(g)                                             # (H, T, T), batched
            out[l].append((S[:, :1].sqrt() * U[:, :, 0].abs(), S[:, :1].sqrt() * Vh[:, 0, :].abs(), wp[l].sum(1).double()))   # alpha, beta (H, T); gamma (H, T)
    return out

def col_scales(sens_l, hid_in, power=1.0):
    """(H, cols) column scalings of M_l from per-box per-token weights (each token block of M_l has hid_in columns)"""
    f = lambda k: torch.cat([w[k].repeat_interleave(hid_in, 1) for w in sens_l], 1) ** power
    return f(0), f(1), f(2)

def candidate_svd_w(st64, Ms, H, dh, Ns, sens, hid_in, use=("qk", "av"), power=1.0, rho=None):
    """svd_jacN with weighted columns: QK pair (Wq_h M diag(alpha), Wk_h M diag(beta)), AV pair (Wv_h M diag(gamma), Wo_h^T N);
    rho (per layer, (cols,)) = extra column scaling of M_l for every side (CROWN-measured width inflation, see inflate_M)"""
    Gq, Ga = [], []
    for l, (Wq, bq, Wk, bk, Wv, bv, Wo) in enumerate(st64):
        M = Ms[l] if rho is None else Ms[l] * rho[l]; al, be, ga_ = col_scales(sens[l], hid_in, power) if sens is not None else (None, None, None); gq, ga = [], []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh); wq = "qk" in use and sens is not None; wv = "av" in use and sens is not None
            gq.append(svd_balance(Wq[sl] @ (M * al[h] if wq else M), Wk[sl] @ (M * be[h] if wq else M)))
            ga.append(svd_balance(Wv[sl] @ (M * ga_[h] if wv else M), Wo[:, sl].T @ Ns[l]))
        Gq.append(torch.stack(gq)); Ga.append(torch.stack(ga))
    return torch.stack(Gq), torch.stack(Ga)

def inflate_M(net, lirpas, bnodes, boxes, Ms, Gq, Ga, H, dh, hid_in, dev):
    """per layer: column scaling rho_l (cols,) of M_l = CROWN's measured width of the head-split q'/k' at token t (summed over heads and
    coordinates, under the CURRENT gauge, one plain CROWN pass per probe box at its eps) divided by the first-order width
    |(G^T W_q)_c M_{b,t}|_1 -- how much the relaxation slack of the earlier layers has inflated this token's box beyond the Jacobian shape.
    rho = 1 where the first-order width is zero (unperturbed tokens at layer 0).  Boxes in the order of box_shapes."""
    L = len(net.layers); mods = net.attn_modules(); rho = [[] for _ in range(L)]; col = [0] * L
    for e, i, y, n, eps in boxes:
        crown_lb(lirpas[n], e, i, eps, y, dev); d = bnodes[n]
        for l in range(L):
            T = e.shape[1]; Mb = Ms[l][:, col[l]:col[l] + T * hid_in].reshape(-1, T, hid_in); col[l] += T * hid_in
            Wq = mods[l][0].weight.detach().double(); Wk = mods[l][1].weight.detach().double()                                              # folded (gauged) weights, (H*dh, hid)
            pred = (torch.einsum("rh,htk->rtk", Wq, Mb).abs().sum(-1) + torch.einsum("rh,htk->rtk", Wk, Mb).abs().sum(-1)).sum(0)          # (T,)
            if all(k in d for k in "qk"):
                meas = ((d["q"].upper - d["q"].lower)[0].sum((0, 2)) + (d["k"].upper - d["k"].lower)[0].sum((0, 1))).double()               # (T,)
                r = torch.where(pred > 1e-9 * pred.max().clamp_min(1e-30), meas / pred.clamp_min(1e-30), torch.ones_like(pred))
            else: r = torch.ones(T, dtype=torch.float64, device=dev)
            rho[l].append(r.repeat_interleave(hid_in))
    return [torch.cat(r_) for r_ in rho]

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
    ap.add_argument("--hybrids", type=int, default=0, help="add learned/candidate side and layer swaps (where does the candidate fall short?)"); ap.add_argument("--cross", type=int, default=0, help="add probe-seed swaps (seed-0 candidate with layers/sides from the seed-1 candidate)")
    ap.add_argument("--radius_names", default="", help="held-out certified-radius screen on dev sentences the learner never saw: comma list of zoo names, or auto"); ap.add_argument("--n_dev", type=int, default=24); ap.add_argument("--dev_pos", type=int, default=2)
    ap.add_argument("--dev_probes", type=int, default=0, help="also build M / N_out from this many unlabeled dev sentences (disjoint from the learner's and the screen's) -> cand:svd_jacN_all_dev"); ap.add_argument("--tag", default="", help="suffix for the saved candidate gauge files")
    ap.add_argument("--sens", type=int, default=0, help="sensitivity-weighted token blocks (cand:svd_sens*)"); ap.add_argument("--infl", type=int, default=0, help="rounds of CROWN-inflation rescaling of M (cand:svd_infl<k>)")
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
    used = {j for j, ex, e, toks in Ss}; rngd = random.Random(a.seed + 7); rest = [s_ for s_ in short_instances(net, m, tok, D, ga.get("max_len", 8), None) if s_[0] not in used]; rngd.shuffle(rest)
    Sd = rest[:a.n_dev] if a.radius_names else []; Sp = rest[a.n_dev:a.n_dev + a.dev_probes] if a.dev_probes else []
    boxes_d = [(e, i, ex["label"], e.shape[1]) for j, ex, e, toks in Sd for i in pos_sets(toks, 1, rngd, a.dev_pos)]
    boxes_p = [(e, i, ex["label"], e.shape[1], 1.0) for j, ex, e, toks in Sp for i in pos_sets(toks, 1, rngd, a.pos)]
    print(f"# held-out {la.data} {ga.get('split', 'dev')} sentences (not the learner's): {len(rest)} available; radius screen {len(Sd)} sentences -> {len(boxes_d)} boxes; text probes {len(Sp)} sentences -> {len(boxes_p)} boxes", flush=True)
    lirpas = make_lirpas(net, [b[3] for b in boxes_r + boxes_s] + [b[3] for b in boxes_d], dev, a.softmax); set_gauge(*I64); t0 = time.time()
    for b in boxes_r + boxes_s: b[4] = certified_radius(lirpas[b[3]], b[0], b[1], b[2], dev, hi=a.hi, iters=8)
    print(f"# {a.name}: {len(boxes_r)} random-token boxes (eps = stock radius, mean {np.mean([b[4] for b in boxes_r]):.4f}) + {len(boxes_s)} SST-dev boxes (mean {np.mean([b[4] for b in boxes_s]):.4f})  [{time.time()-t0:.0f}s]", flush=True)
    # ---- box shapes from the random boxes
    t0 = time.time(); Ms = box_shapes(net, [(b[0], b[1], b[2], b[3], b[4]) for b in boxes_r], dev); print(f"# box shapes M_l: {[tuple(M.shape) for M in Ms][0]} per layer  [{time.time()-t0:.0f}s]", flush=True)
    # verifier-free variants: eps := 1 on every box (pure stacked Jacobians; a common eps scale cancels in the SVD, only the per-box
    # RATIOS -- which came from CROWN radii -- are dropped), on the same random centres (Mu) and on a second seed's centres (Mu2)
    Mu = box_shapes(net, [(b[0], b[1], b[2], b[3], 1.0) for b in boxes_r], dev)
    a2 = argparse.Namespace(name=a.name, data="random", seed=a.seed + 1); Sr2 = short_instances(net, m, tok, load_data(a2, "dev"), a.max_len, a.n_sent, seed=a.seed + 1); rng3 = random.Random(a.seed + 1)
    boxes_r2 = [(e, i, ex["label"], e.shape[1], 1.0) for j, ex, e, toks in Sr2 for i in pos_sets(toks, 1, rng3, a.pos)]; Mu2 = box_shapes(net, boxes_r2, dev)
    print(f"# uniform-eps box shapes: seed {a.seed} {tuple(Mu[0].shape)}, seed {a.seed + 1} {tuple(Mu2[0].shape)}  [{time.time()-t0:.0f}s]", flush=True)
    # output-side functionals N_l (how downstream reads the attention output): margin gradients at the random centres, the same
    # layer's FFN rows, the next layer's projections through the residual (weight-only), and all three combined
    t0 = time.time(); No = out_shapes(net, [(b[0], b[1], b[2], b[3], b[4]) for b in boxes_r], dev); Nf, Nn = weight_shapes(net, st64, dev); Na = [combine_N([Nf[l], Nn[l], No[l]]) for l in range(L)]
    No2 = out_shapes(net, boxes_r2, dev); Na2 = [combine_N([Nf[l], Nn[l], No2[l]]) for l in range(L)]   # second probe seed, verifier-free
    print(f"# output-side functionals: N_out {tuple(No[0].shape)}, N_ffn {tuple(Nf[0].shape)}, N_next {tuple(Nn[0].shape)}, N_all {tuple(Na[0].shape)}; seed {a.seed + 1}: N_all {tuple(Na2[0].shape)}  [{time.time()-t0:.0f}s]", flush=True)
    try: bnodes = {n: attn_bound_nodes(lp) for n, lp in lirpas.items()}
    except Exception as ex: bnodes = {n: {} for n in lirpas}; print(f"# CROWN-width surrogate disabled: node identification failed ({ex})", flush=True)
    print(f"# CROWN-width surrogate: q/k/v bound nodes found for {[len(v) for v in bnodes.values()]} layers per sentence length", flush=True)
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
    for nm, NN in (("out", No), ("ffn", Nf), ("next", Nn), ("all", Na)): zoo[f"cand:svd_jacN_{nm}"] = candidate_svd(st64, Ms, H, dh, NN)
    zoo["cand:svd_jacN_all_u"] = candidate_svd(st64, Mu, H, dh, Na); zoo["cand:svd_jacN_all_u2"] = candidate_svd(st64, Mu2, H, dh, Na2)   # no CROWN radii anywhere; _u2 = other probe seed for M and N_out
    if boxes_p:   # text probes: M and N_out from unlabeled held-out dev sentences (uniform eps), no verifier
        Mp = box_shapes(net, boxes_p, dev); Np = out_shapes(net, boxes_p, dev); Nap = [combine_N([Nf[l], Nn[l], Np[l]]) for l in range(L)]
        zoo["cand:svd_jac_dev"] = candidate_svd(st64, Mp, H, dh); zoo["cand:svd_jacN_all_dev"] = candidate_svd(st64, Mp, H, dh, Nap)
        Mpu = [torch.cat([Mu[l], Mp[l]], 1) for l in range(L)]; zoo["cand:svd_jacN_all_u+dev"] = candidate_svd(st64, Mpu, H, dh, [combine_N([Nf[l], Nn[l], No[l], Np[l]]) for l in range(L)])
    if a.sens:   # round 2a: token blocks of M weighted by how much the bound reads each score / each attended value (probe centres, no verifier)
        t1 = time.time(); sens = sens_shapes(net, [(b[0], b[1], b[2], b[3], b[4]) for b in boxes_r], dev)
        zoo["cand:svd_sens"] = candidate_svd_w(st64, Ms, H, dh, Na, sens, hid); zoo["cand:svd_sens_qk"] = candidate_svd_w(st64, Ms, H, dh, Na, sens, hid, use=("qk",))
        zoo["cand:svd_sens_av"] = candidate_svd_w(st64, Ms, H, dh, Na, sens, hid, use=("av",)); zoo["cand:svd_sens_sqrt"] = candidate_svd_w(st64, Ms, H, dh, Na, sens, hid, power=0.5)
        print(f"# sensitivity weights built [{time.time()-t1:.0f}s]", flush=True)
    if a.infl:   # round 2b: rescale each token block of M_l by CROWN's measured / first-order width under the current candidate, rebuild, repeat
        t1 = time.time(); cur = zoo["cand:svd_jacN_all"]; rho_log = []
        for it in range(1, a.infl + 1):
            set_gauge(*cur); rho = inflate_M(net, lirpas, bnodes, [(b[0], b[1], b[2], b[3], b[4]) for b in boxes_r], Ms, cur[0], cur[1], H, dh, hid, dev)
            rho_log.append([f"{r_[r_ != 1].mean().item():.2f}" if (r_ != 1).any() else "1" for r_ in rho]); cur = candidate_svd_w(st64, Ms, H, dh, Na, None, hid, rho=rho); zoo[f"cand:svd_infl{it}"] = cur
        if a.sens: zoo[f"cand:svd_sens_infl{a.infl}"] = candidate_svd_w(st64, Ms, H, dh, Na, sens, hid, rho=rho)
        set_gauge(*I64); print(f"# CROWN-inflation rescaling: mean rho per layer per round {rho_log} [{time.time()-t1:.0f}s]", flush=True)
    print(f"# svd candidates built [{time.time()-t0:.0f}s]; l1 candidates: {'skipped' if a.l1_steps <= 0 else ''}", flush=True)
    if a.l1_steps > 0:
        (gq, ga), v = candidate_l1(st64, None, H, dh, zoo["cand:svd_iso"], steps=a.l1_steps, log=True); zoo["cand:l1_iso"] = (gq, ga)
        (gq, ga), v = candidate_l1(st64, Ms, H, dh, zoo["cand:svd_jac"], steps=a.l1_steps, log=True); zoo["cand:l1_jac"] = (gq, ga)
    # ---- localisation: which side / layer of the candidate falls short of the learned gauge? (swap one side or one layer at a time)
    dd = lambda g: g.double().to(dev)
    if first and a.hybrids:
        (Lq, La), (Cq, Ca) = map(dd, zoo[first[0]]), map(dd, zoo["cand:svd_jacN_all"])
        zoo["hyb:lrnQK+candAV"] = (Lq, Ca); zoo["hyb:candQK+lrnAV"] = (Cq, La)
        for l in range(L):
            gq, ga = Cq.clone(), Ca.clone(); gq[l], ga[l] = Lq[l], La[l]; zoo[f"hyb:cand-lrn@L{l}"] = (gq, ga)
            gq, ga = Lq.clone(), La.clone(); gq[l], ga[l] = Cq[l], Ca[l]; zoo[f"hyb:lrn-cand@L{l}"] = (gq, ga)
    if a.cross:   # probe-seed dependence: the seed-0 candidate with one side / one layer taken from the seed-1 candidate
        (Xq, Xa), (Yq, Ya) = map(dd, zoo["cand:svd_jacN_all_u"]), map(dd, zoo["cand:svd_jacN_all_u2"])
        print("# probe-seed dependence of svd_jacN_all (relative Frobenius difference seed 0 vs 1, per layer): QK", [f"{((Xq[l] - Yq[l]).norm() / Xq[l].norm()).item():.2f}" for l in range(L)], "AV", [f"{((Xa[l] - Ya[l]).norm() / Xa[l].norm()).item():.2f}" for l in range(L)], flush=True)
        zoo["cross:uQK+u2AV"] = (Xq, Ya); zoo["cross:u2QK+uAV"] = (Yq, Xa)
        for l in range(L):
            gq, ga = Xq.clone(), Xa.clone(); gq[l], ga[l] = Yq[l], Ya[l]; zoo[f"cross:u-u2@L{l}"] = (gq, ga)
    # ---- score everything
    rows = {}; print(f"\n# {'gauge':28s} | surrogate / identity:  l1_iso   l1_jac   l2_iso   l2_jac  (QK+AV; QK, AV for l1_jac)  l1_jacN = with downstream functionals N | CROWN-width surrogate / identity (random boxes; learner boxes) | held-in CROWN: random mean lb, frac ver | SST-dev mean lb, frac ver | max cond", flush=True)
    base = {}
    for name, (Gq, Ga) in zoo.items():
        Gq, Ga = Gq.double().to(dev), Ga.double().to(dev); S = {}
        for key, (M, p, NN) in {"l1_iso": (None, 1, None), "l1_jac": (Ms, 1, None), "l2_iso": (None, 2, None), "l2_jac": (Ms, 2, None), "l1_jacN": (Ms, 1, Na)}.items():
            sq = sa = 0.0; per_layer = []
            for l, (Wq, bq, Wk, bk, Wv, bv, Wo) in enumerate(st64):
                s1, s2 = surrogate(Wq, Wk, Wv, Wo, Gq[l], Ga[l], M[l] if M is not None else None, p, H, dh, NN[l] if NN is not None else None); sq += s1.item(); sa += s2.item(); per_layer.append((s1.item(), s2.item()))
            S[key] = (sq + sa, sq, sa, per_layer)
        if name == "identity": base = {k: v[0] for k, v in S.items()}; base_layers = {k: v[3] for k, v in S.items()}
        set_gauge(Gq, Ga); t0 = time.time(); cw = {"rand": np.zeros((L, 3)), "sst": np.zeros((L, 3))}
        def held_in(bs, key):
            vals = []
            for b in bs:
                vals.append(crown_lb(lirpas[b[3]], b[0], b[1], b[4], b[2], dev))
                try: cw[key] += np.array(crown_width_surrogate(net, bnodes[b[3]], H, dh, Na))
                except Exception as ex: cw[key] += np.nan; print(f"    (crown-width surrogate failed: {ex})", flush=True)
            return np.array(vals)
        vr = held_in(boxes_r, "rand"); vs = held_in(boxes_s, "sst")
        cond = max(torch.linalg.cond(Gq.reshape(-1, dh, dh)).max().item(), torch.linalg.cond(Ga.reshape(-1, dh, dh)).max().item())
        if name == "identity": base_cw = {k: v[:, :2].sum() for k, v in cw.items()}; base_cwN = {k: v[:, 0].sum() + v[:, 2].sum() for k, v in cw.items()}; base_cw_layers = {k: v.copy() for k, v in cw.items()}
        cwr = {k: cw[k][:, :2].sum() / base_cw[k] for k in cw}; cwq = {k: cw[k][:, 0].sum() / base_cw[k] for k in cw}; cwa = {k: cw[k][:, 1].sum() / base_cw[k] for k in cw}; cwN = {k: (cw[k][:, 0].sum() + cw[k][:, 2].sum()) / base_cwN[k] for k in cw}
        rows[name] = {"surr": {k: v[:3] for k, v in S.items()}, "per_layer_l1_jac": S["l1_jac"][3], "crown_width": {k: v.tolist() for k, v in cw.items()}, "random_lb": float(np.nanmean(vr)), "random_ver": float(np.mean(vr > 0)), "sst_lb": float(np.nanmean(vs)), "sst_ver": float(np.mean(vs > 0)), "cond": cond}
        print(f"  {name:28s} | {S['l1_iso'][0]/base['l1_iso']:7.3f}  {S['l1_jac'][0]/base['l1_jac']:7.3f}  {S['l2_iso'][0]/base['l2_iso']:7.3f}  {S['l2_jac'][0]/base['l2_jac']:7.3f}  (QK {S['l1_jac'][1]/base['l1_jac']:.3f}, AV {S['l1_jac'][2]/base['l1_jac']:.3f}) l1_jacN {S['l1_jacN'][0]/base['l1_jacN']:.3f} (AV·N {S['l1_jacN'][2]/base['l1_jacN']:.3f}) | CROWN-width rand {cwr['rand']:.3f} (QK {cwq['rand']:.3f}, AV {cwa['rand']:.3f}; with N {cwN['rand']:.3f}) sst {cwr['sst']:.3f} (QK {cwq['sst']:.3f}, AV {cwa['sst']:.3f}; with N {cwN['sst']:.3f}) | {np.nanmean(vr):+.4f} {np.mean(vr > 0):.2f} | {np.nanmean(vs):+.4f} {np.mean(vs > 0):.2f} | {cond:5.1f}  [{time.time()-t0:.0f}s]", flush=True)
    # ---- held-out certified-radius screen (the paired protocol's metric) on dev sentences the learner never saw
    if boxes_d:
        names = [k for k in zoo if k in ("identity", first[0] if first else "", "cand:svd_jac", "cand:svd_jacN_all", "cand:svd_jacN_all_u2", "cand:svd_jacN_all_dev", "cand:svd_jacN_all_u+dev") or k.startswith(("hyb:", "cross:", "cand:svd_sens", "cand:svd_infl"))] if a.radius_names == "auto" else a.radius_names.split(",")
        print(f"\n# held-out radius screen: {len(Sd)} sentences -> {len(boxes_d)} boxes, certified radius by bisection (hi {a.hi}, 8 iters); gain = ratio of means vs stock, share = gain / learned gain", flush=True); rad = {}
        for name in names:
            set_gauge(*zoo[name]); t0 = time.time(); rad[name] = np.array([certified_radius(lirpas[b[3]], b[0], b[1], b[2], dev, hi=a.hi, iters=8) for b in boxes_d]); rows[name]["dev_radius"] = rad[name].tolist()
            r0 = rad["identity"]; r1 = rad.get(first[0]) if first else None; gain = rad[name].mean() / r0.mean() - 1
            share = gain / (r1.mean() / r0.mean() - 1) if r1 is not None and name != "identity" else float("nan")
            print(f"  {name:28s} mean radius {rad[name].mean():.5f}  vs stock {gain:+6.1%} (larger {int((rad[name] > r0 + 1e-9).sum()):3d} / smaller {int((rad[name] < r0 - 1e-9).sum()):3d})  share {share:5.2f}" + (f"  vs learned: larger {int((rad[name] > r1 + 1e-9).sum()):3d} / smaller {int((rad[name] < r1 - 1e-9).sum()):3d}" if r1 is not None and name != first[0] else "") + f"  [{time.time()-t0:.0f}s]", flush=True)
    # per-layer view of the l1_jac surrogate for identity vs the first learned gauge (layer 1 = exact box shape)
    if first:
        print(f"\n# per-layer l1_jac surrogate (QK, AV), identity -> {first[0]}:")
        for l in range(L):
            i0 = base_layers["l1_jac"][l]; g0 = rows[first[0]]["per_layer_l1_jac"][l]
            c0 = base_cw_layers["sst"][l]; c1 = np.array(rows[first[0]]["crown_width"]["sst"][l])
            print(f"  layer {l}: l1_jac QK {i0[0]:.4g} -> {g0[0]:.4g} ({g0[0]/i0[0]:.3f}), AV {i0[1]:.4g} -> {g0[1]:.4g} ({g0[1]/i0[1]:.3f}) | CROWN-width (learner boxes) QK {c0[0]:.4g} -> {c1[0]:.4g} ({c1[0]/c0[0]:.3f}), AV {c0[1]:.4g} -> {c1[1]:.4g} ({c1[1]/c0[1]:.3f}), AV·N {c0[2]:.4g} -> {c1[2]:.4g} ({c1[2]/c0[2]:.3f})")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump({"rows": rows, "boxes_random": len(boxes_r), "boxes_sst": len(boxes_s), "boxes_dev": len(boxes_d), "boxes_probe": len(boxes_p), "args": vars(a)}, open(a.out, "w"), indent=1)
        gd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gauges")
        for k in [k for k in zoo if k.startswith(("cand:", "hyb:", "cross:"))]:
            torch.save({"qk": zoo[k][0].double().cpu(), "av": zoo[k][1].double().cpu(), "formula": k, "args": vars(a)}, os.path.join(gd, f"formula_{a.name}_{k.split(':', 1)[1].replace('+', '-').replace('@', '_')}{a.tag}.pt"))
        print(f"# saved candidates to gauges/formula_{a.name}_*.pt and {a.out}")

if __name__ == "__main__": main()
