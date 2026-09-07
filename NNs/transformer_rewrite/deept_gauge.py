#!/usr/bin/env python
"""Attention-gauge rewrite for the DeepT (Bonaert et al., PLDI 2021) pretrained SST sentiment transformers
(deept_benchmarks/DeepT/Robustness-Verification-for-Transformers/sst_bert_*; BERT-style, hidden 128, 4 heads, 3/6/12 layers,
'no_var' LayerNorm = mean-subtraction (linear), ReLU MLP, tanh pooler).

Verification spec (DeepT / Shi et al. 2020): the input is the token embedding sequence (word+position+type, BEFORE the
embedding LayerNorm) of a test sentence [CLS] w_1..w_k [SEP]; ONE word position i is perturbed in an l_inf ball of radius eps
(CLS/SEP and word-piece '#' tokens are never perturbed); the property is that the true-label logit stays larger.  DeepT reports
the max certified eps per (sentence, position) via binary search on 10 test sentences sampled with seed 0 (117 positions for
the 3-layer model); we replicate that sampling and evaluate both certified radii and fixed-eps verified counts.

Rewrite: per-head G (query/key) and Ga (value/out.dense) exactly as in genbab_gauge.py (nn.Linear convention).
"""
import sys, os, re, math, json, copy, time, random, argparse, numpy as np, torch, torch.nn as nn
REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; DT = os.path.join(REPO, "deept_benchmarks/DeepT/Robustness-Verification-for-Transformers")
sys.path.insert(0, os.path.join(REPO, "alpha-beta-CROWN/complete_verifier")); sys.path.insert(0, DT)
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
NAMES = ["query.weight", "query.bias", "key.weight", "key.bias", "value.weight", "value.bias", "out.weight"]

def load_sst(split):
    """DeepT's load_data_sst (test/dev): PTB trees -> tokens, binary label (neutral dropped)."""
    data = []
    for line in open(os.path.join(DT, "..", "data", "sst", f"{split}.txt")):
        segs = line[:-1].split(" "); label = int(segs[0][1])
        if label == 2: continue
        label = 0 if label < 2 else 1; tokens = []
        for i in range(len(segs) - 1):
            if segs[i][0] == "(" and segs[i][1] in "01234" and segs[i + 1][0] != "(": tokens.append(segs[i + 1][:segs[i + 1].find(")")])
        tokens = ["(" if t == "-LRB-" else ")" if t == "-RRB-" else t for t in tokens]
        data.append({"label": label, "sent_a": tokens})
    return data

def load_deept(name):
    from Models.modeling import BertForSequenceClassification
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    d = os.path.join(DT, name); ck = os.path.join(d, "ckpt-%d" % int(open(os.path.join(d, "checkpoint")).readline()))
    m = BertForSequenceClassification.from_pretrained(ck, cache_dir=os.path.join(REPO, "deept_benchmarks/cache"), num_labels=2).eval()
    tok = BertTokenizer.from_pretrained(ck, do_lower_case=True)
    return m, tok

class DeepTNet(nn.Module):
    """Clean single-input forward (embeddings -> logits) over the DeepT model's modules; no dropout, no padding mask."""
    def __init__(self, m):
        super().__init__(); self.ln0 = m.bert.embeddings.LayerNorm; self.layers = m.bert.encoder.layer; self.pooler = m.bert.pooler; self.classifier = m.classifier
        a = self.layers[0].attention.self; self.H = a.num_attention_heads; self.dh = a.attention_head_size; self.hid = a.query.in_features; self.frozen_probs = None
    def attn_modules(self):   # per layer: (query, key, value, out_dense)
        return [(l.attention.self.query, l.attention.self.key, l.attention.self.value, l.attention.output.dense) for l in self.layers]
    # diagnostic: self.frozen_probs (nn.ParameterList per layer, or None) -> attention becomes a fixed linear map.  Must NOT be a class
    # attribute: nn.Module attribute lookup would then return the class None instead of the registered ParameterList.
    def forward(self, e):
        x = self.ln0(e); B, n, _ = x.shape
        for li, l in enumerate(self.layers):
            a = l.attention.self
            v = a.value(x).view(B, n, self.H, self.dh).transpose(1, 2)
            if self.frozen_probs is not None: p = self.frozen_probs[li]   # constant attention (ParameterList so BoundedModule traces it on the right device)
            else:
                q = a.query(x).view(B, n, self.H, self.dh).transpose(1, 2); k = a.key(x).view(B, n, self.H, self.dh).transpose(1, 2)
                p = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh), dim=-1)
            c = torch.matmul(p, v).transpose(1, 2).reshape(B, n, self.H * self.dh)
            h = l.attention.output.LayerNorm(l.attention.output.dense(c) + x)
            x = l.output.LayerNorm(l.output.dense(torch.relu(l.intermediate.dense(h))) + h)
        return self.classifier(torch.tanh(self.pooler.dense(x[:, 0])))

def embed(m, tok, example):
    """DeepT get_embeddings: [CLS] tokens [SEP] (<=32), word+position+type embeddings (pre-LayerNorm). Returns (1,n,H), tokens."""
    toks = tok.tokenize(" ".join(example["sent_a"]))[:30]; toks = ["[CLS]"] + toks + ["[SEP]"]
    ids = torch.tensor([tok.convert_tokens_to_ids(toks)]); pos = torch.arange(len(toks)).unsqueeze(0); tt = torch.zeros_like(ids)
    emb = m.bert.embeddings
    with torch.no_grad(): e = emb.word_embeddings(ids) + emb.position_embeddings(pos) + emb.token_type_embeddings(tt)
    return e, toks

def sample_sentences(net, m, tok, data, n=10, seed=0, max_len=32):
    """DeepT data_utils.sample: seed 0, random.randint over the test set, skip too long (>32 tokens incl. CLS/SEP) or misclassified."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); out = []
    for _ in range(n):
        while True:
            j = random.randint(0, len(data) - 1); ex = data[j]; e, toks = embed(m, tok, ex)
            if e.shape[1] > max_len: continue
            with torch.no_grad(): pred = net(e).argmax(1).item()
            if pred != ex["label"]: continue
            out.append((j, ex, e, toks)); break
    return out

def positions(toks):   # DeepT: i in [1, length-2], skip word-piece continuations at i or i+1
    return [i for i in range(1, len(toks) - 1) if not (toks[i][0] == "#" or toks[i + 1][0] == "#")]

def box(e, i, eps):
    xl = e.clone(); xu = e.clone(); xl[0, i] -= eps; xu[0, i] += eps; return xl, xu

def spec_C(label):
    C = torch.zeros(1, 1, 2); C[0, 0, label] = 1; C[0, 0, 1 - label] = -1; return C

def stock_tensors(net):
    return [[q.weight.detach().clone(), q.bias.detach().clone(), k.weight.detach().clone(), k.bias.detach().clone(), v.weight.detach().clone(), v.bias.detach().clone(), o.weight.detach().clone()] for q, k, v, o in net.attn_modules()]

def effective(stock, Gq, Ga, H, dh):
    out = []
    for l, (Wq, bq, Wk, bk, Wv, bv, Wo) in enumerate(stock):
        q, bq2, k, bk2, v, bv2, o = [], [], [], [], [], [], []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh); G = Gq[l, h]; Gi = torch.linalg.inv(G); A = Ga[l, h]; Ai = torch.linalg.inv(A)
            q.append(G.T @ Wq[sl]); bq2.append(G.T @ bq[sl]); k.append(Gi @ Wk[sl]); bk2.append(Gi @ bk[sl]); v.append(A.T @ Wv[sl]); bv2.append(A.T @ bv[sl]); o.append(Wo[:, sl] @ Ai.T)
        out.append([torch.cat(q, 0), torch.cat(bq2), torch.cat(k, 0), torch.cat(bk2), torch.cat(v, 0), torch.cat(bv2), torch.cat(o, 1)])
    return out

def leaves(net):
    return [t for q, k, v, o in net.attn_modules() for t in (q.weight, q.bias, k.weight, k.bias, v.weight, v.bias, o.weight)]

def load_eff(net, effs):
    with torch.no_grad():
        for p, t in zip(leaves(net), [t for e in effs for t in e]): p.copy_(t)

def eye_gauge(L, H, dh, dtype=torch.float32):
    I = torch.eye(dh, dtype=dtype).expand(L, H, dh, dh).clone(); return I, I.clone()

def crown_lb(lirpa, e, i, eps, label, dev, method="CROWN", with_ub=False, grad=False):
    """evaluation-only by default (no autograd graph: in grad mode every A matrix is retained and memory grows ~n^3)"""
    xl, xu = box(e, i, eps); bx = BoundedTensor(e.to(dev), PerturbationLpNorm(norm=np.inf, x_L=xl.to(dev), x_U=xu.to(dev)))
    with torch.set_grad_enabled(grad):
        lb, ub = lirpa.compute_bounds(x=(bx,), method=method, C=spec_C(label).to(dev), bound_lower=True, bound_upper=with_ub)
    return lb.item() if not with_ub else (lb.item(), ub.item())

def certified_radius(lirpa, e, i, label, dev, lo=0.0, hi=0.2, iters=12, method="CROWN"):
    """largest eps (bisection, DeepT-style) with CROWN lb > 0"""
    if crown_lb(lirpa, e, i, hi, label, dev, method) > 0: return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if crown_lb(lirpa, e, i, mid, label, dev, method) > 0: lo = mid
        else: hi = mid
    return lo

def build(name, dev):
    m, tok = load_deept(name); net = DeepTNet(m).to(dev).eval()
    for p in net.parameters(): p.requires_grad_(False)
    return m, tok, net

def cmd_probe(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, "cpu")
    data = load_sst("test"); print(f"# {a.name}: {len(data)} SST test sentences (binary)")
    S = sample_sentences(net, m, tok, data, n=10, seed=0)
    # fidelity: clean forward vs the DeepT model's own forward with the embeddings kwarg (float64)
    m64 = copy.deepcopy(m).double(); net64 = DeepTNet(m64); worst = 0.0
    for j, ex, e, toks in S:
        ids = torch.tensor([tok.convert_tokens_to_ids(toks)])
        with torch.no_grad(): ref = m64(ids, torch.zeros_like(ids), torch.ones_like(ids), embeddings=e.double())[0]; worst = max(worst, (ref - net64(e.double())).abs().max().item())
    print(f"# fidelity: clean forward vs DeepT forward(embeddings=) fp64 max|diff| {worst:.2e}")
    npos = [len(positions(toks)) for _, _, _, toks in S]
    print(f"# sampled sentences (seed 0): idx {[j for j, _, _, _ in S]}; lengths {[e.shape[1] for _, _, e, _ in S]}; positions {npos} (total {sum(npos)}; DeepT reference csv for small_3: 117)")
    net = net.to(dev); L, H, dh = len(net.layers), net.H, net.dh; st = stock_tensors(net)
    lirpas = {}
    def get_lirpa(n):
        if n not in lirpas: lirpas[n] = BoundedModule(net, torch.empty(1, n, net.hid, device=dev), bound_opts={"softmax": a.softmax, "sparse_intermediate_bounds": False}, device=dev)   # sparse_intermediate_bounds=True (default) makes lse-CROWN peak 18.7 GiB at 12 tokens (OOM at 16); False: 0.24 GiB, identical bound
        return lirpas[n]
    # stock: softmax interval widths + CROWN lb at a fixed eps on the first sentence's positions
    j, ex, e, toks = S[0]; lp = get_lirpa(e.shape[1]); i = positions(toks)[0]
    if dev == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); lb = crown_lb(lp, e, i, a.eps, ex["label"], dev); print(f"# sentence {j} (len {e.shape[1]}) pos {i} '{toks[i]}' eps {a.eps}: stock CROWN lb {lb:+.4f}  [{time.time()-t0:.1f}s, peak GPU {torch.cuda.max_memory_allocated()/2**30 if dev == 'cuda' else 0:.1f} GiB]")
    k = 0
    for n_, node in lp._modules.items():
        if type(node).__name__ == "BoundSoftmax" and getattr(node, "lower", None) is not None:
            print(f"   softmax L{k}: interval width mean {(node.upper - node.lower).mean().item():.4f} max {(node.upper - node.lower).max().item():.4f}; perturbed-row width mean {(node.upper - node.lower)[0, :, i].mean().item():.4f}"); k += 1
    torch.manual_seed(0)
    for scale in [0.05, 0.3]:
        I, _ = eye_gauge(L, H, dh); Gq = (I + scale * torch.randn(L, H, dh, dh)).to(dev); Ga = (I + scale * torch.randn(L, H, dh, dh)).to(dev)
        load_eff(net, effective([[t.to(dev) for t in w] for w in st], Gq, Ga, H, dh)); lb2 = crown_lb(lp, e, i, a.eps, ex["label"], dev)
        load_eff(net, effective([[t.to(dev) for t in w] for w in st], Gq, I.to(dev), H, dh)); lbq = crown_lb(lp, e, i, a.eps, ex["label"], dev)
        load_eff(net, effective([[t.to(dev) for t in w] for w in st], I.to(dev), Ga, H, dh)); lba = crown_lb(lp, e, i, a.eps, ex["label"], dev)
        print(f"# random gauge scale {scale}: lb both {lb2:+.4f} (delta {lb2 - lb:+.2e}), qk-only {lbq:+.4f}, av-only {lba:+.4f}")
    load_eff(net, effective([[t.to(dev) for t in w] for w in st], *[g.to(dev) for g in eye_gauge(L, H, dh)], H, dh))
    # stock certified radii on all positions of the first 3 sentences
    t0 = time.time(); rads = []
    for j, ex, e, toks in S[:3]:
        lp = get_lirpa(e.shape[1]); r = [certified_radius(lp, e, i, ex["label"], dev) for i in positions(toks)]; rads += r
        print(f"# sentence {j} (len {e.shape[1]}, label {ex['label']}): stock CROWN certified radii mean {np.mean(r):.4f} min {min(r):.4f} max {max(r):.4f}  [{time.time()-t0:.0f}s]")
    print(f"# stock vanilla CROWN({a.softmax}) certified radius over {len(rads)} positions: mean {np.mean(rads):.4f} (DeepT reference small_3 l_inf: 0.0327 mean over 117)")

def short_instances(net, m, tok, data, max_len, n_max=None, seed=0):
    """all correctly classified sentences with <= max_len tokens (incl. CLS/SEP), in file order (optionally a seeded subset)"""
    out = []
    for j, ex in enumerate(data):
        e, toks = g_embed(m, tok, ex)
        if e.shape[1] > max_len or not positions(toks): continue
        with torch.no_grad(): pred = net(e.to(next(net.parameters()).device)).argmax(1).item()
        if pred == ex["label"]: out.append((j, ex, e, toks))
    if n_max and len(out) > n_max: rng = random.Random(seed); out = rng.sample(out, n_max)
    return out
g_embed = embed

def cmd_radii(a):
    """stock certified radii on short test sentences, vanilla CROWN vs alpha-CROWN"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); data = load_sst("test")
    S = short_instances(net, m, tok, data, a.max_len, a.n_sent); print(f"# {a.name}: {len(S)} correctly classified test sentences with <= {a.max_len} tokens; positions {sum(len(positions(t)) for _, _, _, t in S)}")
    lirpas = {}
    def get_lirpa(n):
        if n not in lirpas: lirpas[n] = BoundedModule(net, torch.empty(1, n, net.hid, device=dev), bound_opts={"softmax": a.softmax, "sparse_intermediate_bounds": False, "optimize_bound_args": {"iteration": 20, "lr_alpha": 0.1}}, device=dev)
        return lirpas[n]
    rows = []; t0 = time.time()
    for j, ex, e, toks in S:
        lp = get_lirpa(e.shape[1])
        for i in positions(toks):
            r = certified_radius(lp, e, i, ex["label"], dev, hi=a.hi, iters=a.iters)
            ra = certified_radius(lp, e, i, ex["label"], dev, hi=a.hi, iters=a.iters, method="CROWN-Optimized") if a.alpha else float("nan")
            rows.append((j, i, e.shape[1], r, ra))
        print(f"  sentence {j} len {e.shape[1]}: CROWN radii mean {np.mean([x[3] for x in rows if x[0] == j]):.4f}" + (f", alpha-CROWN {np.mean([x[4] for x in rows if x[0] == j]):.4f}" if a.alpha else "") + f"  [{time.time()-t0:.0f}s]", flush=True)
    R = np.array([x[3] for x in rows]); print(f"# stock vanilla CROWN certified radius over {len(R)} positions: mean {R.mean():.4f} median {np.median(R):.4f} min {R.min():.4f} max {R.max():.4f}; quantiles 25/75 {np.quantile(R, .25):.4f}/{np.quantile(R, .75):.4f}")
    if a.alpha: RA = np.array([x[4] for x in rows]); print(f"# stock alpha-CROWN certified radius: mean {RA.mean():.4f} median {np.median(RA):.4f}; ratio alpha/CROWN mean {np.mean(RA / np.maximum(R, 1e-9)):.2f}")
    if a.save_json: json.dump({"rows": rows}, open(a.save_json, "w"))


def make_lirpas(net, lengths, dev, softmax="lse", alpha=False):
    """one BoundedModule per sentence length; all share the net's parameter tensors (load_eff updates every one)"""
    opts = {"softmax": softmax, "sparse_intermediate_bounds": False}
    if alpha: opts["optimize_bound_args"] = {"iteration": 20, "lr_alpha": 0.1}
    return {n: BoundedModule(net, torch.empty(1, n, net.hid, device=dev), bound_opts=opts, device=dev) for n in sorted(set(lengths))}

def fp64_gate(name, Gq, Ga, boxes, n_pts=32, seed=0):
    """max |stock - gauged| logits in float64 over random points in the boxes (list of (e, i, eps))"""
    m, tok = load_deept(name); net = DeepTNet(m).double().eval(); m2, _ = load_deept(name); net2 = DeepTNet(m2).double().eval()
    L, H, dh = len(net.layers), net.H, net.dh; load_eff(net2, effective([[t.double() for t in w] for w in stock_tensors(net2)], Gq.double(), Ga.double(), H, dh))
    gen = torch.Generator().manual_seed(seed); worst = 0.0
    with torch.no_grad():
        for e, i, eps in boxes:
            xl, xu = box(e.double(), i, eps); u = torch.rand((n_pts,) + xl.shape[1:], generator=gen, dtype=torch.float64); x = xl + u * (xu - xl)
            worst = max(worst, (net(x) - net2(x)).abs().max().item())
    return worst

def cmd_learn(a):
    torch.manual_seed(a.seed); random.seed(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, tok, net = build(a.name, dev); L, H, dh = len(net.layers), net.H, net.dh; st = [[t.to(dev) for t in w] for w in stock_tensors(net)]
    data = load_sst(a.split); S = short_instances(net, m, tok, data, a.max_len, a.n_sent, seed=a.seed)
    rng = random.Random(a.seed); boxes = []   # (e, i, label, n)
    for j, ex, e, toks in S:
        P = positions(toks); rng.shuffle(P)
        for i in P[:a.pos_per_sent]: boxes.append([e, i, ex["label"], e.shape[1], None])
    lirpas = make_lirpas(net, [b[3] for b in boxes], dev, a.softmax)
    # per-box eps = stock certified radius (bisection, no grad) scaled by eps_scale -> the stock bound sits at ~0 on every tuning box
    t0 = time.time()
    for b in boxes: b[4] = a.eps_scale * certified_radius(lirpas[b[3]], b[0], b[1], b[2], dev, hi=a.hi, iters=a.radius_iters)
    R = np.array([b[4] for b in boxes]); print(f"# {a.name}: {len(S)} {a.split} sentences <= {a.max_len} tokens -> {len(boxes)} tuning boxes; per-box eps = {a.eps_scale} x stock radius: mean {R.mean():.4f} median {np.median(R):.4f} min {R.min():.4f} max {R.max():.4f}  [{time.time()-t0:.0f}s]", flush=True)
    for p in leaves(net): p.requires_grad_(True)
    I, _ = eye_gauge(L, H, dh); Gq = nn.Parameter(I.clone().to(dev), requires_grad=a.which in ("both", "qk")); Ga = nn.Parameter(I.clone().to(dev), requires_grad=a.which in ("both", "av"))
    params = [p for p in (Gq, Ga) if p.requires_grad]; opt = torch.optim.Adam(params, lr=a.lr)
    def evaluate(bs):
        load_eff(net, effective(st, Gq.detach(), Ga.detach(), H, dh)); v = np.array([crown_lb(lirpas[b[3]], b[0], b[1], b[4], b[2], dev) for b in bs])
        return np.nanmean(v), float(np.mean(v > 0)), int(np.isnan(v).sum())
    ev_boxes = boxes[:a.n_eval]; ev = evaluate(ev_boxes); print(f"# init=id: eval on {len(ev_boxes)} tuning boxes at their eps: mean lb {ev[0]:+.4f} frac_ver {ev[1]:.3f} nan {ev[2]}", flush=True)
    # sharing check: two BoundedModules must both see a weight change
    if len(lirpas) > 1:
        b1, b2 = boxes[0], next(b for b in boxes if b[3] != boxes[0][3]); v1 = [crown_lb(lirpas[b[3]], b[0], b[1], b[4], b[2], dev) for b in (b1, b2)]
        with torch.no_grad(): leaves(net)[0].mul_(1.01)
        v2 = [crown_lb(lirpas[b[3]], b[0], b[1], b[4], b[2], dev) for b in (b1, b2)]; load_eff(net, effective(st, Gq.detach(), Ga.detach(), H, dh))
        print(f"# parameter-sharing check across BoundedModules: lb before {['%+.4f' % v for v in v1]} after perturbing a leaf {['%+.4f' % v for v in v2]} (both must change)", flush=True)
    t0 = time.time(); best = (ev[0], (Gq.detach().double().cpu().clone(), Ga.detach().double().cpu().clone()), -1); skipped = 0
    for step in range(a.steps):
        effs = effective(st, Gq, Ga, H, dh); load_eff(net, effs); lv = leaves(net)
        for p in lv: p.grad = None
        objs = []
        for b in rng.sample(boxes, a.accum):
            with torch.autograd.set_detect_anomaly(bool(a.debug)):
                lb = crown_lb_t(lirpas[b[3]], b[0], b[1], b[4], b[2], dev); objs.append(lb.item())
                o = lb - a.hinge_w * torch.relu(-lb) if a.obj == "hinge" else lb   # hinge: extra penalty on boxes that fall below 0 (protects the worst boxes)
                (-o / a.accum).backward()
        grads = [p.grad for p in lv]
        if any(g_ is None for g_ in grads): raise RuntimeError("no gradient reached the effective weights")
        if a.debug:
            for l in range(L):
                for k_, nm in enumerate(NAMES): g_ = grads[l * 7 + k_]; print(f"    L{l} {nm}: finite={bool(torch.isfinite(g_).all())} |g|={g_.norm().item():.3e}")
        torch.autograd.backward([t for e in effs for t in e], grads)
        if a.cond_pen > 0: (a.cond_pen * sum((p ** 2).sum() + (torch.linalg.inv(p) ** 2).sum() for p in params)).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip); obj = float(np.mean(objs))
        if not torch.isfinite(gn) or not np.isfinite(obj): skipped += 1; print(f"  step {step:4d}: NON-FINITE (obj {obj}, grad_norm {gn.item()}) -> skipped", flush=True); opt.zero_grad(); continue
        opt.step(); opt.zero_grad()
        if step % a.log_every == 0 or step == a.steps - 1:
            with torch.no_grad(): cond = max(torch.linalg.cond(p.reshape(-1, dh, dh)).max().item() for p in params)
            ev = evaluate(ev_boxes); print(f"  step {step:4d} batch_obj={obj:+.4f} grad_norm={gn.item():.3e} | eval mean lb {ev[0]:+.4f} frac_ver {ev[1]:.3f} nan {ev[2]} | max cond(G)={cond:.2f} | {time.time()-t0:.0f}s", flush=True)
            if ev[0] > best[0]: best = (ev[0], (Gq.detach().double().cpu().clone(), Ga.detach().double().cpu().clone()), step)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"qk": best[1][0], "av": best[1][1], "args": vars(a), "best_step": best[2], "skipped": skipped}, a.out)
    print(f"# saved best gauges (step {best[2]}, eval mean lb {best[0]:+.4f}, skipped steps {skipped}) -> {a.out}; wall {time.time()-t0:.0f}s")
    print(f"# fp64 GATE (random points in 16 tuning boxes): {fp64_gate(a.name, best[1][0], best[1][1], [(b[0], b[1], b[4]) for b in boxes[:16]]):.2e}")

def crown_lb_t(lirpa, e, i, eps, label, dev):
    """tensor-valued CROWN lb with autograd enabled (learner)"""
    xl, xu = box(e, i, eps); bx = BoundedTensor(e.to(dev), PerturbationLpNorm(norm=np.inf, x_L=xl.to(dev), x_U=xu.to(dev)))
    lb, _ = lirpa.compute_bounds(x=(bx,), method="CROWN", C=spec_C(label).to(dev), bound_lower=True, bound_upper=False); return lb.squeeze()

def cmd_eval(a):
    """paired stock vs gauged on TEST sentences: certified radii (same bisection grid) + fixed-eps verified counts; vanilla CROWN, no grad"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); L, H, dh = len(net.layers), net.H, net.dh; st = [[t.to(dev) for t in w] for w in stock_tensors(net)]
    data = load_sst("test"); S = short_instances(net, m, tok, data, a.max_len, a.n_sent, seed=a.seed); lirpas = make_lirpas(net, [e.shape[1] for _, _, e, _ in S], dev, a.softmax)
    inst = [(j, i, e, ex["label"], toks) for j, ex, e, toks in S for i in positions(toks)]; print(f"# {a.name}: {len(S)} test sentences <= {a.max_len} tokens, {len(inst)} (sentence, position) instances", flush=True)
    eps_list = [float(x) for x in a.eps_list.split(",")]; res = {}
    Gq, Ga = load_gauge(a.gauge, L, H, dh)
    for tag, (gq, ga) in [("stock", eye_gauge(L, H, dh, torch.float64)), ("gauged", (Gq, Ga))]:
        load_eff(net, effective(st, gq.float().to(dev), ga.float().to(dev), H, dh)); t0 = time.time()
        rad = np.array([certified_radius(lirpas[e.shape[1]], e, i, y, dev, hi=a.hi, iters=a.iters) for j, i, e, y, _ in inst])
        fixed = {eps: np.array([crown_lb(lirpas[e.shape[1]], e, i, eps, y, dev) for j, i, e, y, _ in inst]) for eps in eps_list}
        res[tag] = (rad, fixed)
        if a.save_json:  # partial save after each half, so a job time-out keeps the finished half (small_12 eval lost 5 h this way)
            json.dump({"inst": [(j, i, e.shape[1], y) for j, i, e, y, _ in inst], **{f"{t}_rad": r[0].tolist() for t, r in res.items()}, "fixed": {str(eps): {t: r[1][eps].tolist() for t, r in res.items()} for eps in eps_list}}, open(a.save_json, "w"))
        print(f"# {tag}: certified radius mean {rad.mean():.4f} median {np.median(rad):.4f} | " + "; ".join(f"eps {eps}: verified {(v > 0).sum()}/{len(v)} (nan {np.isnan(v).sum()}) mean lb {np.nanmean(v):+.4f}" for eps, v in fixed.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
    dr = res["gauged"][0] - res["stock"][0]
    print(f"# PAIRED radius gauged-stock over {len(dr)} instances: larger on {(dr > 0).sum()}, smaller on {(dr < 0).sum()}, equal {(dr == 0).sum()}; mean rel change {np.mean(dr / np.maximum(res['stock'][0], 1e-9)):+.3f}; mean radius {res['stock'][0].mean():.4f} -> {res['gauged'][0].mean():.4f}")
    for eps in eps_list:
        s_, g_ = res["stock"][1][eps], res["gauged"][1][eps]; d = g_ - s_
        print(f"# PAIRED eps {eps}: lb tighter on {(d > 0).sum()}/{len(d)}, looser {(d < 0).sum()}, mean delta {np.nanmean(d):+.4f}; verified {(s_ > 0).sum()} -> {(g_ > 0).sum()}; flips unverified->verified {((s_ <= 0) & (g_ > 0)).sum()}, verified->unverified {((s_ > 0) & (g_ <= 0)).sum()}")
    print(f"# fp64 GATE (random points in 24 instance boxes at eps {eps_list[0]}): {fp64_gate(a.name, Gq, Ga, [(e, i, eps_list[0]) for j, i, e, y, _ in inst[:24]]):.2e}")
    if a.save_json: json.dump({"inst": [(j, i, e.shape[1], y) for j, i, e, y, _ in inst], "stock_rad": res["stock"][0].tolist(), "gauged_rad": res["gauged"][0].tolist(), "fixed": {str(eps): {"stock": res["stock"][1][eps].tolist(), "gauged": res["gauged"][1][eps].tolist()} for eps in eps_list}}, open(a.save_json, "w"))

def cmd_eval_alpha(a):
    """paired stock vs gauged alpha-CROWN (CROWN-Optimized, 20 it) at fixed eps on test sentences <= max_len tokens (<= 8 fits the 44 GB GPU)"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); L, H, dh = len(net.layers), net.H, net.dh; st = [[t.to(dev) for t in w] for w in stock_tensors(net)]
    data = load_sst("test"); S = short_instances(net, m, tok, data, a.max_len, a.n_sent, seed=a.seed)
    inst = [(j, i, e, ex["label"], toks) for j, ex, e, toks in S for i in positions(toks)]; eps_list = [float(x) for x in a.eps_list.split(",")]
    print(f"# {a.name}: alpha-CROWN paired eval on {len(S)} test sentences <= {a.max_len} tokens, {len(inst)} instances, eps {eps_list}", flush=True)
    # Memory: alpha-CROWN's per-call peak is 36 GiB (5 tokens) / 62 GiB (6 tokens) on the 6-layer model, and every BoundedModule
    # RETAINS 4.5-8 GiB of alpha/bound state after a call (diagnostics/_alpha_mem_probe2.py), so per-length modules kept alive
    # across the loop pushed the 6-token case over 80 GB. A fresh module per call (~1 s) keeps only the peak. Weights do not
    # need requires_grad for alpha optimisation (probe 1: identical bound), so they stay frozen.
    def alpha_and_crown(e, i, eps, y):
        lp = make_lirpas(net, [e.shape[1]], dev, a.softmax, alpha=True)[e.shape[1]]
        v = crown_lb(lp, e, i, eps, y, dev, method="CROWN-Optimized", grad=True); c = crown_lb(lp, e, i, eps, y, dev); del lp; torch.cuda.empty_cache(); return v, c
    Gq, Ga = load_gauge(a.gauge, L, H, dh); res = {}
    for tag, (gq, ga) in [("stock", eye_gauge(L, H, dh, torch.float64)), ("gauged", (Gq, Ga))]:
        load_eff(net, effective(st, gq.float().to(dev), ga.float().to(dev), H, dh)); t0 = time.time(); res[tag] = {}
        for eps in eps_list:
            vc = [alpha_and_crown(e, i, eps, y) for j, i, e, y, _ in inst]; v = np.array([x[0] for x in vc]); c = np.array([x[1] for x in vc])
            res[tag][eps] = (v, c); print(f"# {tag} eps {eps}: alpha-CROWN verified {(v > 0).sum()}/{len(v)} (nan {np.isnan(v).sum()}) mean lb {np.nanmean(v):+.4f} | CROWN verified {(c > 0).sum()} mean lb {np.nanmean(c):+.4f}  [{time.time()-t0:.0f}s]", flush=True)
    for eps in eps_list:
        s_, g_ = res["stock"][eps][0], res["gauged"][eps][0]; d = g_ - s_
        print(f"# PAIRED alpha-CROWN eps {eps}: tighter on {(d > 0).sum()}/{len(d)}, looser {(d < 0).sum()}, mean delta {np.nanmean(d):+.4f}; verified {(s_ > 0).sum()} -> {(g_ > 0).sum()}; flips up {((s_ <= 0) & (g_ > 0)).sum()}, down {((s_ > 0) & (g_ <= 0)).sum()}")
    if a.save_json: json.dump({"inst": [(j, i, e.shape[1], y) for j, i, e, y, _ in inst], "res": {t: {str(e): {"alpha": r[0].tolist(), "crown": r[1].tolist()} for e, r in d.items()} for t, d in res.items()}}, open(a.save_json, "w"))

def cmd_attrib(a):
    """attention-slack attribution: CROWN lb with the attention probabilities frozen at their box-centre values (attention = constant
    linear map; QK bilinear, softmax and AV bilinear slack removed) vs the true lb, at eps = stock certified radius x {1, 1.5}"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); data = load_sst("test"); S = short_instances(net, m, tok, data, a.max_len, a.n_sent, seed=a.seed)
    lirpas = make_lirpas(net, [e.shape[1] for _, _, e, _ in S], dev, a.softmax); rows = []
    for j, ex, e, toks in S:
        n = e.shape[1]; lp = lirpas[n]
        for i in positions(toks)[:a.pos_per_sent]:
            r = certified_radius(lp, e, i, ex["label"], dev, hi=a.hi, iters=8)
            for f in (1.0, 1.5):
                eps = f * r; lb = crown_lb(lp, e, i, eps, ex["label"], dev); ub = crown_lb(lp, e, i, eps, ex["label"], dev, with_ub=True)[1]
                # frozen attention: separate BoundedModule with constant probs captured at the centre
                probs = []
                x = net.ln0(e.to(dev)); B, n_, _ = x.shape
                with torch.no_grad():
                    for l in net.layers:
                        at = l.attention.self; q = at.query(x).view(B, n_, net.H, net.dh).transpose(1, 2); k = at.key(x).view(B, n_, net.H, net.dh).transpose(1, 2); v = at.value(x).view(B, n_, net.H, net.dh).transpose(1, 2)
                        p = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(net.dh), dim=-1); probs.append(p)
                        c = torch.matmul(p, v).transpose(1, 2).reshape(B, n_, -1); h = l.attention.output.LayerNorm(l.attention.output.dense(c) + x); x = l.output.LayerNorm(l.output.dense(torch.relu(l.intermediate.dense(h))) + h)
                net.frozen_probs = nn.ParameterList([nn.Parameter(p_, requires_grad=False) for p_ in probs]); lpf = BoundedModule(net, torch.empty(1, n, net.hid, device=dev), bound_opts={"softmax": a.softmax, "sparse_intermediate_bounds": False}, device=dev)
                lbf, ubf = crown_lb(lpf, e, i, eps, ex["label"], dev, with_ub=True); net.frozen_probs = None
                rows.append((j, i, n, f, eps, lb, ub, lbf, ubf))
        print(f"  sentence {j} len {n}: " + "; ".join(f"x{f:g}: width {ub-lb:.3f} -> frozen-attn {ubf-lbf:.3f} (attn share {(1-(ubf-lbf)/max(ub-lb,1e-9))*100:.0f}%)" for (_, _, _, f, eps, lb, ub, lbf, ubf) in rows[-2:]), flush=True)
    R = np.array([[x[3], x[6] - x[5], x[8] - x[7]] for x in rows])
    for f in (1.0, 1.5):
        w = R[(R[:, 0] == f) & np.isfinite(R[:, 1]) & np.isfinite(R[:, 2])]; print(f"# eps = {f:g} x stock radius over {len(w)} finite instances: mean CROWN width {w[:, 1].mean():.3f}, frozen-attention width {w[:, 2].mean():.3f} -> attention nonlinearities account for {(1 - w[:, 2].sum() / w[:, 1].sum())*100:.1f}% of the width")
    if a.save_json: json.dump({"rows": rows}, open(a.save_json, "w"))

def load_gauge(path, L, H, dh):
    if path is None: return eye_gauge(L, H, dh, torch.float64)
    g_ = torch.load(path); return g_["qk"], g_["av"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["probe", "radii", "learn", "eval", "eval_alpha", "attrib"])
    ap.add_argument("--split", default="dev"); ap.add_argument("--pos_per_sent", type=int, default=3); ap.add_argument("--eps_scale", type=float, default=1.0); ap.add_argument("--radius_iters", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150); ap.add_argument("--accum", type=int, default=4); ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--cond_pen", type=float, default=1e-4); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--which", default="both"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log_every", type=int, default=10); ap.add_argument("--debug", type=int, default=0); ap.add_argument("--n_eval", type=int, default=48)
    ap.add_argument("--out", default=None); ap.add_argument("--gauge", default=None); ap.add_argument("--eps_list", default="0.01,0.02,0.03")
    ap.add_argument("--obj", default="mean", help="mean | hinge"); ap.add_argument("--hinge_w", type=float, default=4.0)
    ap.add_argument("--max_len", type=int, default=12); ap.add_argument("--n_sent", type=int, default=20); ap.add_argument("--alpha", type=int, default=1); ap.add_argument("--hi", type=float, default=0.1); ap.add_argument("--iters", type=int, default=10); ap.add_argument("--save_json", default=None); ap.add_argument("--name", default="sst_bert_small_3"); ap.add_argument("--eps", type=float, default=0.03); ap.add_argument("--softmax", default="lse"); ap.add_argument("--crown_batch", type=int, default=512)
    a = ap.parse_args(); {"probe": cmd_probe, "radii": cmd_radii, "learn": cmd_learn, "eval": cmd_eval, "eval_alpha": cmd_eval_alpha, "attrib": cmd_attrib}[a.cmd](a)
