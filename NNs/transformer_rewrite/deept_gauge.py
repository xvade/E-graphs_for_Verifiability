#!/usr/bin/env python
"""Attention-gauge rewrite for the DeepT (Bonaert et al., PLDI 2021) pretrained SST sentiment transformers
(deept_benchmarks/DeepT/Robustness-Verification-for-Transformers/sst_bert_*; BERT-style, hidden 128, 4 heads, 3/6/12 layers,
'no_var' LayerNorm = mean-subtraction (linear), ReLU MLP, tanh pooler).  The same release ships further separately trained
checkpoints that load unchanged: yelp_bert_small_{3,6,12} (Yelp polarity, own vocab; --data yelp or auto), sst_bert_big_*
(hidden 256), sst_bert_smaller_* (hidden 64), sst_bert_standard_layer_norm_* (full LayerNorm).

Verification spec (DeepT / Shi et al. 2020): the input is the token embedding sequence (word+position+type, BEFORE the
embedding LayerNorm) of a test sentence [CLS] w_1..w_k [SEP]; ONE word position i is perturbed in an l_inf ball of radius eps
(CLS/SEP and word-piece '#' tokens are never perturbed); the property is that the true-label logit stays larger.  DeepT reports
the max certified eps per (sentence, position) via binary search on 10 test sentences sampled with seed 0 (117 positions for
the 3-layer model); we replicate that sampling and evaluate both certified radii and fixed-eps verified counts.

Rewrite: per-head G (query/key) and Ga (value/out.dense) exactly as in genbab_gauge.py (nn.Linear convention).
"""
import sys, os, re, math, json, copy, time, random, argparse, numpy as np, torch, torch.nn as nn
import os as _o; REPO = _o.environ.get("REPO") or _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..")); DT = os.path.join(REPO, "deept_benchmarks/DeepT/Robustness-Verification-for-Transformers")
sys.path.insert(0, os.path.join(REPO, "alpha-beta-CROWN/complete_verifier")); sys.path.insert(0, DT)
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
NAMES = ["query.weight", "query.bias", "key.weight", "key.bias", "value.weight", "value.bias", "out.weight"]

def run_meta():
    """Provenance stamp written into every saved results JSON (added 2026-09-12): the card, Slurm job, host, date and the
    fp32 matmul precision the numbers were produced under. See PROVENANCE.md for why the card matters per claim."""
    import socket, datetime
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"device": dev, "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
            "date": datetime.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__,
            "matmul_precision": torch.get_float32_matmul_precision(), "tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE")}

def dump_json(d, f):
    """json.dump with the run_meta() stamp merged into a top-level dict (partial per-instance files are not stamped)."""
    if isinstance(d, dict): d = {**d, "meta": {**(d.get("meta") or {}), "run": run_meta()}}
    json.dump(d, f)

def load_yelp(split, max_words=14):
    """Yelp Review Polarity (DeepT's yelp_bert_* models; csv label 1 -> 0 negative, 2 -> 1 positive).  Yelp has no dev split, so
    split 'dev' = train.csv and 'test' = test.csv (disjoint).  Only reviews with <= max_words words are kept (the pipeline needs
    <= 12 word pieces anyway; 856 of the 38000 test reviews).  Word splitting is a regex stand-in for DeepT's nltk.word_tokenize
    (nltk is not in the venv); BERT's basic tokenizer re-splits punctuation, so the word-piece sequence is the same."""
    import csv
    def words(t): return re.findall(r"\w+|[^\w\s]", t.replace("\\n", " ").replace('\\"', '"'))
    f = {"test": "test.csv", "dev": "train.csv"}[split]; data = []
    for l, t in csv.reader(open(os.path.join(DT, "..", "data", "yelp", f))):
        if len(t.split()) <= max_words:
            w = words(t)
            if len(w) <= max_words: data.append({"label": int(l) - 1, "sent_a": w})
    return data

def load_random(a, n=400, min_words=3, max_words=6):
    """Out-of-distribution tuning data with NO dataset: sequences of random whole-word entries of the model's own vocabulary
    (alphabetic, not word-piece continuations, not special tokens), 3-6 words each; label None -> short_instances() uses the
    model's own prediction as the label, so every sequence passes the 'correctly classified' filter.  Seeded by --seed."""
    d = os.path.join(DT, a.name); ck = os.path.join(d, "ckpt-%d" % int(open(os.path.join(d, "checkpoint")).readline()))
    vocab = [w.strip() for w in open(os.path.join(ck, "vocab.txt"), encoding="utf-8")]
    words = [w for w in vocab if w.isalpha() and w.islower() and len(w) >= 2]
    rng = random.Random(1000 + a.seed)
    return [{"label": None, "sent_a": [rng.choice(words) for _ in range(rng.randint(min_words, max_words))]} for _ in range(n)]

def load_data(a, split):
    """--data sst|yelp|random|auto (auto = from the model name prefix). 'yelp' with an sst_* model = cross-dataset tuning text
    (tokenised with the model's own tokenizer, Yelp's true sentiment labels); 'random' = random vocabulary sequences (no dataset)."""
    d = a.data if a.data != "auto" else ("yelp" if a.name.startswith("yelp") else "sst")
    return load_random(a) if d == "random" else load_yelp(split) if d == "yelp" else load_sst(split)

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
        # split attribution (diagnostic, inexact): lin_mode in {None, "qk", "sm", "av", "all"} linearises ONE attention nonlinearity
        # (or all three) at the box centre; the centre tensors per layer are ParameterLists so BoundedModule traces them as constants.
        self.lin_mode = None; self.c_q0 = self.c_k0 = self.c_s0 = self.c_p0 = self.c_v0 = None
    def attn_modules(self):   # per layer: (query, key, value, out_dense)
        return [(l.attention.self.query, l.attention.self.key, l.attention.self.value, l.attention.output.dense) for l in self.layers]
    # diagnostic: self.frozen_probs (nn.ParameterList per layer, or None) -> attention becomes a fixed linear map.  Must NOT be a class
    # attribute: nn.Module attribute lookup would then return the class None instead of the registered ParameterList.
    def forward(self, e):
        x = self.ln0(e); B, n, _ = x.shape
        for li, l in enumerate(self.layers):
            a = l.attention.self
            v = a.value(x).view(B, n, self.H, self.dh).transpose(1, 2)
            lm = self.lin_mode
            if self.frozen_probs is not None: p = self.frozen_probs[li]   # constant attention (ParameterList so BoundedModule traces it on the right device)
            else:
                q = a.query(x).view(B, n, self.H, self.dh).transpose(1, 2); k = a.key(x).view(B, n, self.H, self.dh).transpose(1, 2)
                if lm in ("qk", "all"):   # QK^T -> first-order expansion at the centre: q0 k^T + q k0^T - q0 k0^T (linear in q, k)
                    q0, k0 = self.c_q0[li], self.c_k0[li]
                    sc = (torch.matmul(q0, k.transpose(-1, -2)) + torch.matmul(q, k0.transpose(-1, -2)) - torch.matmul(q0, k0.transpose(-1, -2))) / math.sqrt(self.dh)
                else: sc = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)
                if lm in ("sm", "all"):   # softmax -> Jacobian at the centre: p0 + p0*(s-s0) - p0*<p0, s-s0>  (linear in s)
                    p0, s0 = self.c_p0[li], self.c_s0[li]; d = sc - s0; p = p0 + p0 * d - p0 * (p0 * d).sum(-1, keepdim=True)
                else: p = torch.softmax(sc, dim=-1)
            if lm in ("av", "all") and self.frozen_probs is None:   # P V -> p0 v + p v0 - p0 v0 (linear in p, v)
                p0, v0 = self.c_p0[li], self.c_v0[li]; cv = torch.matmul(p0, v) + torch.matmul(p, v0) - torch.matmul(p0, v0)
            else: cv = torch.matmul(p, v)
            c = cv.transpose(1, 2).reshape(B, n, self.H * self.dh)
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
    """l_inf box of radius eps on one embedding row i (DeepT's spec) or on several rows (i = tuple/list of positions: k-word perturbation)"""
    xl = e.clone(); xu = e.clone()
    for j in (list(i) if isinstance(i, (list, tuple)) else [i]): xl[0, j] -= eps; xu[0, j] += eps
    return xl, xu

def pos_sets(toks, k, rng, cap):
    """k-word position sets of a sentence: k = 1 -> single valid positions (shuffled, first `cap`; identical rng usage to the original
    one-word code path); k = 2 -> all pairs of valid positions, shuffled, first `cap`."""
    P = positions(toks)
    if k == 1: rng.shuffle(P); return P[:cap]
    from itertools import combinations
    Q = [tuple(c) for c in combinations(P, k)]; rng.shuffle(Q); return Q[:cap]

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

def fold64(st, gq, ga, H, dh):
    """certificate-side fold: invert and multiply in fp64, round ONCE to fp32 (the fp32 fold's inverse error is ~kappa*u per entry,
    6 ulps on small_6, and moves the logits 15x more than the single rounding: 6e-8 vs 4e-9, measured 2026-09-10)"""
    return [[t.float() for t in w] for w in effective([[t.double() for t in w] for w in st], gq.double().to(st[0][0].device), ga.double().to(st[0][0].device), H, dh)]

def leaves(net):
    return [t for q, k, v, o in net.attn_modules() for t in (q.weight, q.bias, k.weight, k.bias, v.weight, v.bias, o.weight)]

def load_eff(net, effs):
    with torch.no_grad():
        for p, t in zip(leaves(net), [t for e in effs for t in e]): p.copy_(t)

class IntervalLinear(nn.Module):
    """x @ Wt + b with Wt (1 x in x out) and b (1 x 1 x out) as auto_LiRPA BoundedParameters.  nn.Linear traces to MatMul(x, Transpose(W)), and
    auto_LiRPA's weight-perturbation path cannot push A matrices through that Transpose (4-D A vs 3-D permutation), so the
    interval-weight pass swaps each attention nn.Linear for this module, whose parameter feeds the MatMul directly."""
    def __init__(self, Wt, b): super().__init__(); self.Wt = Wt; self.b = b
    def forward(self, x):
        y = torch.matmul(x, self.Wt); return y + self.b if self.b is not None else y

def install_weight_intervals(net, effs64):
    """Rigorous transfer of the certificate to the ORIGINAL network.  The gauged network equals the original in real arithmetic; the
    only difference is that the folded attention weights (computed here in fp64) are stored in fp32.  Declare each stored entry as
    the interval between its two fp32 neighbours (it contains the exact real product: the fp32 rounding is <= half an ulp and the
    fp64 product error is ~1e-16 relative, far below one ulp), so the bound holds for every network in the family, including the
    exact rewrite, i.e. for the original function.  auto_LiRPA then treats those layers as bilinear (weight x activation) with its
    McCormick relaxation.  Returns the largest fp32 rounding actually incurred."""
    from auto_LiRPA import BoundedParameter
    def bp(t64):
        w = t64.float(); lo = torch.nextafter(w, torch.full_like(w, -float("inf"))); hi = torch.nextafter(w, torch.full_like(w, float("inf")))
        return BoundedParameter(w.clone(), PerturbationLpNorm(norm=np.inf, x_L=lo, x_U=hi), requires_grad=False), (w.double() - t64).abs().max().item()
    worst = 0.0; net._orig_attn = []
    for l, (Wq, bq, Wk, bk, Wv, bv, Wo) in zip(net.layers, effs64):
        at = l.attention.self; od = l.attention.output; net._orig_attn.append((at.query, at.key, at.value, od.dense))
        for parent, nm, W, b in ((at, "query", Wq, bq), (at, "key", Wk, bk), (at, "value", Wv, bv), (od, "dense", Wo, None)):
            # leading size-1 "batch" dim: auto_LiRPA's concretisation takes dim 0 of a perturbed root as the batch dimension
            Wt, e1 = bp(W.T.contiguous().unsqueeze(0)); worst = max(worst, e1)
            if b is not None: bb, e2 = bp(b.reshape(1, 1, -1)); worst = max(worst, e2)
            else: bb = nn.Parameter(getattr(parent, nm).bias.detach().clone().reshape(1, 1, -1), requires_grad=False)   # out-projection bias is not transformed by the gauge
            setattr(parent, nm, IntervalLinear(Wt, bb))
    return worst

def remove_weight_intervals(net):
    for l, (q, k, v, o) in zip(net.layers, net._orig_attn): l.attention.self.query, l.attention.self.key, l.attention.self.value, l.attention.output.dense = q, k, v, o
    net._orig_attn = []

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
    data = load_data(a, "test"); print(f"# {a.name}: {len(data)} test sentences (binary)")
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
        if ex["label"] is None: ex["label"] = pred   # label-free tuning data (--data random): the model's own prediction is the label
        if pred == ex["label"]: out.append((j, ex, e, toks))
    if n_max and len(out) > n_max: rng = random.Random(seed); out = rng.sample(out, n_max)
    return out
g_embed = embed

def cmd_radii(a):
    """stock certified radii on short test sentences, vanilla CROWN vs alpha-CROWN"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); data = load_data(a, "test")
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
    if a.save_json: dump_json({"rows": rows}, open(a.save_json, "w"))


def make_lirpas(net, lengths, dev, softmax="lse", alpha=False, alpha_iters=20, alpha_lr=0.1, alpha_shared=False):
    """one BoundedModule per sentence length; all share the net's parameter tensors (load_eff updates every one)"""
    opts = {"softmax": softmax, "sparse_intermediate_bounds": False}
    if alpha: opts["optimize_bound_args"] = {"iteration": alpha_iters, "lr_alpha": alpha_lr, "use_shared_alpha": bool(alpha_shared)}
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
    data = load_data(a, a.split); S = short_instances(net, m, tok, data, a.max_len, a.n_sent, seed=a.seed)
    rng = random.Random(a.seed); boxes = []   # (e, i, label, n)
    for j, ex, e, toks in S:
        for i in pos_sets(toks, a.k_words, rng, a.pos_per_sent): boxes.append([e, i, ex["label"], e.shape[1], None])
    if a.k_words > 1: print(f"# k-word perturbation: every tuning box widens {a.k_words} embedding rows at once", flush=True)
    if getattr(a, "label", -1) >= 0: boxes = [b for b in boxes if b[2] == a.label]; print(f"# per-class learner: keeping the {len(boxes)} boxes with label {a.label}", flush=True)   # label-restricted tuning set (per-class gauge)
    lirpas = make_lirpas(net, [b[3] for b in boxes], dev, a.softmax)
    # per-box eps = stock certified radius (bisection, no grad) scaled by eps_scale -> the stock bound sits at ~0 on every tuning box
    t0 = time.time()
    for b in boxes: b[4] = a.eps_scale * certified_radius(lirpas[b[3]], b[0], b[1], b[2], dev, hi=a.hi, iters=a.radius_iters)
    # --alpha_iters K > 0: ALTERNATING joint optimisation of the gauge and alpha-CROWN's relaxation parameters.  Per box: (1) inner loop, weights
    # frozen: CROWN-Optimized (K Adam iterations on alpha at the current gauge; best alpha is left in the module's nodes); (2) outer step, alpha
    # frozen (nodes set to 'reuse'): one plain backward pass whose graph reaches the effective weights -> gradient to G.  By Danskin's theorem
    # d/dG max_alpha lb(G, alpha) = d lb/dG at the optimal alpha, so this is the exact envelope gradient without differentiating through the
    # alpha loop.  A fresh BoundedModule per call (each retains 4.5-8 GiB of alpha state after a call, see cmd_eval_alpha); measured peaks on
    # small_6: 36 GiB at 5 tokens, 62 GiB at 6 -> tuning boxes must be <= 6 tokens on an 80 GB card.
    alt = a.alpha_iters > 0
    if alt: print(f"# alternating alpha/gauge optimisation: {a.alpha_iters} alpha iterations (lr {a.alpha_lr}, shared {a.alpha_shared}) per box, fresh module per call", flush=True)
    def alpha_module(n): return make_lirpas(net, [n], dev, a.softmax, alpha=True, alpha_iters=a.alpha_iters, alpha_lr=a.alpha_lr, alpha_shared=a.alpha_shared)[n]
    def alpha_lb(b, grad_weights):
        """alpha-optimised lower bound on box b; if grad_weights, also returns the 'reuse' pass tensor whose graph reaches the weights"""
        lp = alpha_module(b[3]); lv_ = leaves(net)
        for p in lv_: p.requires_grad_(False)
        v = crown_lb(lp, b[0], b[1], b[4], b[2], dev, method="CROWN-Optimized", grad=True)   # inner loop (alpha only)
        for p in lv_: p.requires_grad_(True)
        if not grad_weights: del lp; torch.cuda.empty_cache(); return v, None
        for node in lp.get_enabled_opt_act(): node.opt_reuse()
        lb = crown_lb_t(lp, b[0], b[1], b[4], b[2], dev); return v, (lb, lp)
    R = np.array([b[4] for b in boxes]); print(f"# {a.name}: {len(S)} {a.split} sentences <= {a.max_len} tokens -> {len(boxes)} tuning boxes; per-box eps = {a.eps_scale} x stock radius: mean {R.mean():.4f} median {np.median(R):.4f} min {R.min():.4f} max {R.max():.4f}  [{time.time()-t0:.0f}s]", flush=True)
    for p in leaves(net): p.requires_grad_(True)
    I, _ = eye_gauge(L, H, dh); q0, a0 = (I.clone(), I.clone()) if a.gauge is None else [t.float() for t in load_gauge(a.gauge, L, H, dh)]   # learn --gauge: warm start (e.g. the closed-form gauge) instead of identity
    Gq = nn.Parameter(q0.to(dev), requires_grad=a.which in ("both", "qk")); Ga = nn.Parameter(a0.to(dev), requires_grad=a.which in ("both", "av"))
    params = [p for p in (Gq, Ga) if p.requires_grad]; opt = torch.optim.Adam(params, lr=a.lr)
    def evaluate(bs):
        load_eff(net, effective(st, Gq.detach(), Ga.detach(), H, dh))
        v = np.array([alpha_lb(b, False)[0] if alt else crown_lb(lirpas[b[3]], b[0], b[1], b[4], b[2], dev) for b in bs])
        return np.nanmean(v), float(np.mean(v > 0)), int(np.isnan(v).sum())
    ev_boxes = boxes[:a.n_eval]; ev = evaluate(ev_boxes); print(f"# init={'id' if a.gauge is None else a.gauge}: eval on {len(ev_boxes)} tuning boxes at their eps: mean lb {ev[0]:+.4f} frac_ver {ev[1]:.3f} nan {ev[2]}", flush=True)
    # sharing check: two BoundedModules must both see a weight change
    if len(lirpas) > 1 and not alt:
        b1, b2 = boxes[0], next(b for b in boxes if b[3] != boxes[0][3]); v1 = [crown_lb(lirpas[b[3]], b[0], b[1], b[4], b[2], dev) for b in (b1, b2)]
        with torch.no_grad(): leaves(net)[0].mul_(1.01)
        v2 = [crown_lb(lirpas[b[3]], b[0], b[1], b[4], b[2], dev) for b in (b1, b2)]; load_eff(net, effective(st, Gq.detach(), Ga.detach(), H, dh))
        print(f"# parameter-sharing check across BoundedModules: lb before {['%+.4f' % v for v in v1]} after perturbing a leaf {['%+.4f' % v for v in v2]} (both must change)", flush=True)
    t0 = time.time(); best = (ev[0], (Gq.detach().double().cpu().clone(), Ga.detach().double().cpu().clone()), -1); skipped = 0; start_step = 0
    ck = (a.out + ".ckpt") if a.out else None
    if ck and a.resume and os.path.exists(ck):   # preemptible partitions: continue from the last checkpoint (current gauge, optimiser state, best so far)
        c = torch.load(ck); start_step = c["step"] + 1
        with torch.no_grad(): Gq.copy_(c["gq_cur"].float().to(dev)); Ga.copy_(c["ga_cur"].float().to(dev))
        opt.load_state_dict(c["opt"]); best = (c["best_val"], (c["qk"], c["av"]), c["best_step"]); skipped = c.get("skipped", 0)
        print(f"# resumed from {ck}: continuing at step {start_step} (best so far {best[0]:+.4f} at step {best[2]})", flush=True)
    def save_ckpt(step):
        if ck: torch.save({"step": step, "gq_cur": Gq.detach().double().cpu(), "ga_cur": Ga.detach().double().cpu(), "opt": opt.state_dict(), "qk": best[1][0], "av": best[1][1], "best_val": best[0], "best_step": best[2], "skipped": skipped, "args": vars(a)}, ck + ".tmp"); os.replace(ck + ".tmp", ck)
    for step in range(start_step, a.steps):
        effs = effective(st, Gq, Ga, H, dh); load_eff(net, effs); lv = leaves(net)
        for p in lv: p.grad = None
        objs = []
        for b in rng.sample(boxes, a.accum):
            with torch.autograd.set_detect_anomaly(bool(a.debug)):
                if alt:
                    v_alpha, (lb, lp_) = alpha_lb(b, True)
                    if step == 0: print(f"    alt check: alpha-optimised lb {v_alpha:+.4f} | reuse-pass lb {lb.item():+.4f} (graph to the weights)", flush=True)
                else: lb = crown_lb_t(lirpas[b[3]], b[0], b[1], b[4], b[2], dev)
                objs.append(lb.item())
                o = lb - a.hinge_w * torch.relu(-lb) if a.obj == "hinge" else lb   # hinge: extra penalty on boxes that fall below 0 (protects the worst boxes)
                (-o / a.accum).backward()
                if alt: del lp_, lb, o; torch.cuda.empty_cache()
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
        if ck and a.ckpt_every and (step % a.ckpt_every == 0 or step == a.steps - 1): save_ckpt(step)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"qk": best[1][0], "av": best[1][1], "args": vars(a), "best_step": best[2], "skipped": skipped, "partial": False}, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
    print(f"# saved best gauges (step {best[2]}, eval mean lb {best[0]:+.4f}, skipped steps {skipped}) -> {a.out}; wall {time.time()-t0:.0f}s")
    print(f"# fp64 GATE (random points in 16 tuning boxes): {fp64_gate(a.name, best[1][0], best[1][1], [(b[0], b[1], b[4]) for b in boxes[:16]]):.2e}")

def crown_lb_t(lirpa, e, i, eps, label, dev):
    """tensor-valued CROWN lb with autograd enabled (learner)"""
    xl, xu = box(e, i, eps); bx = BoundedTensor(e.to(dev), PerturbationLpNorm(norm=np.inf, x_L=xl.to(dev), x_U=xu.to(dev)))
    lb, _ = lirpa.compute_bounds(x=(bx,), method="CROWN", C=spec_C(label).to(dev), bound_lower=True, bound_upper=False); return lb.squeeze()

def cmd_eval(a):
    """paired stock vs gauged on TEST sentences: certified radii (same bisection grid) + fixed-eps verified counts; vanilla CROWN, no grad"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); L, H, dh = len(net.layers), net.H, net.dh; st = [[t.to(dev) for t in w] for w in stock_tensors(net)]
    data = load_data(a, "test"); S = short_instances(net, m, tok, data, a.max_len, a.n_sent, seed=a.seed); lirpas = make_lirpas(net, [e.shape[1] for _, _, e, _ in S], dev, a.softmax)
    if a.k_words == 1: inst = [(j, i, e, ex["label"], toks) for j, ex, e, toks in S for i in positions(toks)]
    else: prng = random.Random(a.seed); inst = [(j, i, e, ex["label"], toks) for j, ex, e, toks in S for i in pos_sets(toks, a.k_words, prng, a.pairs_per_sent)]
    print(f"# {a.name}: {len(S)} test sentences <= {a.max_len} tokens, {len(inst)} (sentence, position{'-set' if a.k_words > 1 else ''}) instances; k_words = {a.k_words}", flush=True)
    eps_list = [float(x) for x in a.eps_list.split(",")]; res = {}
    gauges = [load_gauge(g_, L, H, dh) for g_ in a.gauge.split(",")]; Gq, Ga = gauges[0]   # --gauge a.pt[,b.pt,...] -> tags gauged, gauged2, ...
    tags = ["gauged" if k == 0 else f"gauged{k + 1}" for k in range(len(gauges))]
    inst_key = json.loads(json.dumps([(j, i, e.shape[1], y) for j, i, e, y, _ in inst]))
    if a.save_json and os.path.exists(a.save_json):   # resume: weight sets already in the JSON (same instances) are not recomputed (ckpt pre-emptions restart the job)
        prev = json.load(open(a.save_json))
        if prev.get("inst") == inst_key:
            for t in ["stock"] + tags + ["stock_wint", "gauged_wint"]:
                if f"{t}_rad" in prev and all(t in prev["fixed"].get(str(eps), {}) for eps in eps_list):
                    res[t] = (np.array(prev[f"{t}_rad"]), {eps: np.array(prev["fixed"][str(eps)][t]) for eps in eps_list}); print(f"# resumed weight set '{t}' from {a.save_json}", flush=True)
        else: print(f"# {a.save_json} exists but its instances differ -> recomputing everything", flush=True)
    for tag, (gq, ga) in [("stock", eye_gauge(L, H, dh, torch.float64))] + list(zip(tags, gauges)):
        if tag in res: continue
        load_eff(net, fold64(st, gq, ga, H, dh)); t0 = time.time()
        rad = np.array([certified_radius(lirpas[e.shape[1]], e, i, y, dev, hi=a.hi, iters=a.iters) for j, i, e, y, _ in inst])
        fixed = {eps: np.array([crown_lb(lirpas[e.shape[1]], e, i, eps, y, dev) for j, i, e, y, _ in inst]) for eps in eps_list}
        res[tag] = (rad, fixed)
        if a.save_json:  # partial save after each half, so a job time-out keeps the finished half (small_12 eval lost 5 h this way)
            dump_json({"inst": [(j, i, e.shape[1], y) for j, i, e, y, _ in inst], **{f"{t}_rad": r[0].tolist() for t, r in res.items()}, "fixed": {str(eps): {t: r[1][eps].tolist() for t, r in res.items()} for eps in eps_list}}, open(a.save_json, "w"))
        print(f"# {tag}: certified radius mean {rad.mean():.4f} median {np.median(rad):.4f} | " + "; ".join(f"eps {eps}: verified {(v > 0).sum()}/{len(v)} (nan {np.isnan(v).sum()}) mean lb {np.nanmean(v):+.4f}" for eps, v in fixed.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
    if a.weight_intervals:   # rigorous-transfer tier: fp32-neighbour intervals on the folded attention weights (see install_weight_intervals)
        st64 = [[t.double() for t in w] for w in st]; lengths = [e.shape[1] for _, _, e, _ in S]
        for tag, (gq, ga) in [("stock_wint", eye_gauge(L, H, dh, torch.float64)), ("gauged_wint", gauges[0])]:
            if tag in res: continue
            effs64 = effective(st64, gq.double().to(dev), ga.double().to(dev), H, dh); worst = install_weight_intervals(net, effs64)
            lw = make_lirpas(net, lengths, dev, a.softmax); t0 = time.time()
            print(f"# {tag}: attention weights as 2-ulp fp32 intervals (largest fp32 rounding of the folded weights {worst:.2e})", flush=True)
            rad = np.array([certified_radius(lw[e.shape[1]], e, i, y, dev, hi=a.hi, iters=a.iters) for j, i, e, y, _ in inst])
            fixed = {eps: np.array([crown_lb(lw[e.shape[1]], e, i, eps, y, dev) for j, i, e, y, _ in inst]) for eps in eps_list}
            res[tag] = (rad, fixed); remove_weight_intervals(net); del lw; torch.cuda.empty_cache()
            if a.save_json: dump_json({"inst": [(j, i, e.shape[1], y) for j, i, e, y, _ in inst], **{f"{t}_rad": r[0].tolist() for t, r in res.items()}, "fixed": {str(eps): {t: r[1][eps].tolist() for t, r in res.items()} for eps in eps_list}}, open(a.save_json, "w"))
            print(f"# {tag}: certified radius mean {rad.mean():.4f} median {np.median(rad):.4f} | " + "; ".join(f"eps {eps}: verified {(v > 0).sum()}/{len(v)} (nan {np.isnan(v).sum()}) mean lb {np.nanmean(v):+.4f}" for eps, v in fixed.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
    pairs = [(t, "stock") for t in tags] + [(t, "gauged") for t in tags[1:]]
    if a.weight_intervals: pairs += [("stock_wint", "stock"), ("gauged_wint", "gauged"), ("gauged_wint", "stock")]
    for t, base in pairs:
        dr = res[t][0] - res[base][0]
        print(f"# PAIRED radius {t}-{base} over {len(dr)} instances: larger on {(dr > 0).sum()}, smaller on {(dr < 0).sum()}, equal {(dr == 0).sum()}; mean rel change {np.mean(dr / np.maximum(res[base][0], 1e-9)):+.3f}; mean radius {res[base][0].mean():.4f} -> {res[t][0].mean():.4f}")
        for eps in eps_list:
            s_, g_ = res[base][1][eps], res[t][1][eps]; d = g_ - s_
            print(f"# PAIRED {t}-{base} eps {eps}: lb tighter on {(d > 0).sum()}/{len(d)}, looser {(d < 0).sum()}, mean delta {np.nanmean(d):+.4f}; verified {(s_ > 0).sum()} -> {(g_ > 0).sum()}; flips unverified->verified {((s_ <= 0) & (g_ > 0)).sum()}, verified->unverified {((s_ > 0) & (g_ <= 0)).sum()}")
    for t, (gq, ga) in zip(tags, gauges): print(f"# fp64 GATE {t} (random points in 24 instance boxes at eps {eps_list[0]}): {fp64_gate(a.name, gq, ga, [(e, i, eps_list[0]) for j, i, e, y, _ in inst[:24]]):.2e}")
    if a.save_json: dump_json({"inst": [(j, i, e.shape[1], y) for j, i, e, y, _ in inst], "k_words": a.k_words, "gauges": a.gauge.split(","), **{f"{t}_rad": r[0].tolist() for t, r in res.items()}, "fixed": {str(eps): {t: r[1][eps].tolist() for t, r in res.items()} for eps in eps_list}}, open(a.save_json, "w"))

def cmd_eval_alpha(a):
    """paired stock vs gauged alpha-CROWN (CROWN-Optimized, 20 it) at fixed eps on test sentences <= max_len tokens (<= 8 fits the 44 GB GPU)"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); L, H, dh = len(net.layers), net.H, net.dh; st = [[t.to(dev) for t in w] for w in stock_tensors(net)]
    data = load_data(a, "test"); S = short_instances(net, m, tok, data, a.max_len, a.n_sent, seed=a.seed)
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
        if tag == "stock" and a.skip_stock: continue
        load_eff(net, fold64(st, gq, ga, H, dh)); t0 = time.time(); res[tag] = {}
        for eps in eps_list:
            part = (a.save_json + f".part_{tag}_{eps}") if a.save_json else None; done = json.load(open(part)) if part and os.path.exists(part) else []
            if done: print(f"# resumed {len(done)} of {len(inst)} instances for {tag} eps {eps} from {part}", flush=True)
            for k, (j, i, e, y, _) in enumerate(inst):
                if k < len(done): continue
                done.append(list(alpha_and_crown(e, i, eps, y)))
                if part: json.dump(done, open(part + ".tmp", "w")); os.replace(part + ".tmp", part)
            vc = done; v = np.array([x[0] for x in vc], dtype=float); c = np.array([x[1] for x in vc], dtype=float)
            res[tag][eps] = (v, c); print(f"# {tag} eps {eps}: alpha-CROWN verified {(v > 0).sum()}/{len(v)} (nan {np.isnan(v).sum()}) mean lb {np.nanmean(v):+.4f} | CROWN verified {(c > 0).sum()} mean lb {np.nanmean(c):+.4f}  [{time.time()-t0:.0f}s]", flush=True)
    for eps in (eps_list if "stock" in res else []):
        s_, g_ = res["stock"][eps][0], res["gauged"][eps][0]; d = g_ - s_
        print(f"# PAIRED alpha-CROWN eps {eps}: tighter on {(d > 0).sum()}/{len(d)}, looser {(d < 0).sum()}, mean delta {np.nanmean(d):+.4f}; verified {(s_ > 0).sum()} -> {(g_ > 0).sum()}; flips up {((s_ <= 0) & (g_ > 0)).sum()}, down {((s_ > 0) & (g_ <= 0)).sum()}")
    if a.save_json: dump_json({"inst": [(j, i, e.shape[1], y) for j, i, e, y, _ in inst], "res": {t: {str(e): {"alpha": r[0].tolist(), "crown": r[1].tolist()} for e, r in d.items()} for t, d in res.items()}}, open(a.save_json, "w"))

def cmd_attrib(a):
    """attention-slack attribution: CROWN lb with the attention probabilities frozen at their box-centre values (attention = constant
    linear map; QK bilinear, softmax and AV bilinear slack removed) vs the true lb, at eps = stock certified radius x {1, 1.5}"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); data = load_data(a, "test"); S = short_instances(net, m, tok, data, a.max_len, a.n_sent, seed=a.seed)
    lirpas = make_lirpas(net, [e.shape[1] for _, _, e, _ in S], dev, a.softmax); rows = []
    for j, ex, e, toks in S:
        n = e.shape[1]; lp = lirpas[n]
        for i in positions(toks)[:a.pos_per_sent]:
            r = certified_radius(lp, e, i, ex["label"], dev, hi=a.hi, iters=8)
        factors = [float(x) for x in a.factors.split(",")]; modes = ["qk", "sm", "av", "all"] if a.split_attrib else []
        for i in positions(toks)[:a.pos_per_sent]:
            r = certified_radius(lp, e, i, ex["label"], dev, hi=a.hi, iters=8)
            for f in factors:
                eps = f * r; lb = crown_lb(lp, e, i, eps, ex["label"], dev); ub = crown_lb(lp, e, i, eps, ex["label"], dev, with_ub=True)[1]
                # centre activations per layer (identical for every linearisation, which is exact at the centre)
                probs, C = [], {"q": [], "k": [], "s": [], "p": [], "v": []}
                x = net.ln0(e.to(dev)); B, n_, _ = x.shape
                with torch.no_grad():
                    for l in net.layers:
                        at = l.attention.self; q = at.query(x).view(B, n_, net.H, net.dh).transpose(1, 2); k = at.key(x).view(B, n_, net.H, net.dh).transpose(1, 2); v = at.value(x).view(B, n_, net.H, net.dh).transpose(1, 2)
                        sc = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(net.dh); p = torch.softmax(sc, dim=-1); probs.append(p)
                        for key, t in zip("qksp v".replace(" ", ""), (q, k, sc, p, v)): C[key].append(t)
                        c = torch.matmul(p, v).transpose(1, 2).reshape(B, n_, -1); h = l.attention.output.LayerNorm(l.attention.output.dense(c) + x); x = l.output.LayerNorm(l.output.dense(torch.relu(l.intermediate.dense(h))) + h)
                bo = {"softmax": a.softmax, "sparse_intermediate_bounds": False}
                # frozen attention: separate BoundedModule with constant probs captured at the centre
                net.frozen_probs = nn.ParameterList([nn.Parameter(p_, requires_grad=False) for p_ in probs]); lpf = BoundedModule(net, torch.empty(1, n, net.hid, device=dev), bound_opts=bo, device=dev)
                lbf, ubf = crown_lb(lpf, e, i, eps, ex["label"], dev, with_ub=True); net.frozen_probs = None; del lpf
                split = {}
                if modes:
                    PL = lambda ts: nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in ts])
                    net.c_q0, net.c_k0, net.c_s0, net.c_p0, net.c_v0 = PL(C["q"]), PL(C["k"]), PL(C["s"]), PL(C["p"]), PL(C["v"])
                    for md in modes:
                        net.lin_mode = md; lpm = BoundedModule(net, torch.empty(1, n, net.hid, device=dev), bound_opts=bo, device=dev)
                        lbm, ubm = crown_lb(lpm, e, i, eps, ex["label"], dev, with_ub=True); split[md] = ubm - lbm; del lpm
                    net.lin_mode = None; net.c_q0 = net.c_k0 = net.c_s0 = net.c_p0 = net.c_v0 = None
                rows.append((j, i, n, f, eps, lb, ub, lbf, ubf, split))
        for (_, i_, _, f, eps, lb, ub, lbf, ubf, sp) in rows[-len(factors) * len(positions(toks)[:a.pos_per_sent]):]:
            w = ub - lb; sh = lambda v: f"{(1 - v / max(w, 1e-9)) * 100:.0f}%"
            print(f"  sentence {j} len {n} pos {i_} x{f:g}: width {w:.3f}; frozen-attn {sh(ubf - lbf)}" + "".join(f"; lin-{md} {sh(v)}" for md, v in sp.items()), flush=True)
    for f in factors:
        rf = [x for x in rows if x[3] == f and np.isfinite(x[6] - x[5]) and np.isfinite(x[8] - x[7]) and all(np.isfinite(v) for v in x[9].values())]
        if not rf: print(f"# eps = {f:g} x stock radius: no finite instances"); continue
        W = sum(x[6] - x[5] for x in rf); Wf = sum(x[8] - x[7] for x in rf)
        line = f"# eps = {f:g} x stock radius over {len(rf)} finite instances: mean CROWN width {W/len(rf):.3f}; share of width removed by: frozen attention {(1 - Wf / W)*100:.1f}%"
        for md in modes: line += f"; lin-{md} {(1 - sum(x[9][md] for x in rf) / W)*100:.1f}%"
        print(line, flush=True)
    if a.save_json: dump_json({"rows": rows, "modes": modes, "factors": factors}, open(a.save_json, "w"))

def pq_optimise(net, st, lirpa, e, i, eps, label, dev, Gq0, Ga0, H, dh, steps, lr, cond_pen, clip, stop_at=None):
    """Per-query gauge (route A): Adam on (Gq, Ga) from the given init, maximising THIS box's CROWN margin lb at eps; best-of over
    iterates (step 0 = the init, so the result is never below the fixed gauge); optional early stop once lb > stop_at.  Sound for
    any iterate because every gauge is an exact rewrite.  Returns (best lb, (Gq, Ga) fp32 on dev, best step, steps run)."""
    Gq = nn.Parameter(Gq0.float().to(dev).clone()); Ga = nn.Parameter(Ga0.float().to(dev).clone()); opt = torch.optim.Adam([Gq, Ga], lr=lr)
    best = (-float("inf"), None, -1); run = 0
    for step in range(steps + 1):
        effs = effective(st, Gq, Ga, H, dh); load_eff(net, effs); lv = leaves(net)
        for p_ in lv: p_.grad = None
        lb = crown_lb_t(lirpa, e, i, eps, label, dev); v = lb.item()
        if np.isfinite(v) and v > best[0]: best = (v, (Gq.detach().clone(), Ga.detach().clone()), step)
        if step == steps or (stop_at is not None and v > stop_at): break
        (-lb).backward(); grads = [p_.grad for p_ in lv]
        if any(g_ is None for g_ in grads): break
        for g_ in grads: g_[~torch.isfinite(g_)] = 0.0      # near the lse NaN cliff a few entries are non-finite: mask them (as pbv_learn does) instead of giving up
        torch.autograd.backward([t for ee in effs for t in ee], grads)
        if cond_pen > 0: (cond_pen * sum((p_ ** 2).sum() + (torch.linalg.inv(p_) ** 2).sum() for p_ in (Gq, Ga))).backward()
        for p_ in (Gq, Ga): p_.grad[~torch.isfinite(p_.grad)] = 0.0
        gn = torch.nn.utils.clip_grad_norm_([Gq, Ga], clip)
        if not torch.isfinite(gn) or gn.item() == 0.0: opt.zero_grad(); break
        opt.step(); opt.zero_grad(); run = step + 1
    return best[0], best[1], best[2], run

def pq_safe(*args, **kw):
    """pq_optimise, or None when grad-mode CROWN runs out of GPU memory (retained A matrices grow ~n^3: 80 GB holds 11 tokens of small_6)"""
    net, lirpa, e = args[0], args[2], args[3]; cap = kw.pop("max_tokens", 0)
    if cap and e.shape[1] > cap: return None                      # pre-emptive: the retained grad-mode graph would not fit (see --pq_max_tokens)
    global PQ_OOM
    torch.cuda.empty_cache()
    try: r = pq_optimise(*args, **kw)
    except (torch.OutOfMemoryError, RuntimeError) as ex:   # auto_LiRPA's scripted ops re-raise CUDA OOM as a plain RuntimeError
        if "out of memory" not in str(ex): raise
        print(f"    per-query optimisation OOM ({str(ex)[:60]}...) -> recorded as the fixed gauge; the process will exit 3 after saving so the chain restarts it with a clean GPU", flush=True); r = None; PQ_OOM = True
    for p_ in leaves(net): p_.grad = None
    # a grad-mode pass leaves its node bounds / A matrices (and so the whole retained graph) on the BoundedModule of that sentence length;
    # with several lengths in play these add up, so drop them after EVERY optimisation, not only after a failure
    lirpa._clear_and_set_new(None); purge_graph_tensors(lirpa); import gc; gc.collect(); torch.cuda.empty_cache(); return r
PQ_OOM = False

def purge_graph_tensors(lirpa):
    """drop every graph-attached tensor a grad-mode pass left on the BoundedModule or its nodes (attributes _clear_and_set_new does not
    know about, e.g. relaxation caches): anything with a grad_fn is a product of a pass and is recomputed by the next compute_bounds"""
    def purge(obj):
        n = 0
        for k, v in list(vars(obj).items()):
            if k.startswith("_") or isinstance(v, (nn.Parameter, nn.Module)): continue
            if isinstance(v, torch.Tensor) and v.grad_fn is not None: delattr(obj, k); n += 1
            elif isinstance(v, (list, tuple)) and v and all(isinstance(t, torch.Tensor) for t in v) and any(t.grad_fn is not None for t in v): delattr(obj, k); n += 1
            elif isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()) and any(t.grad_fn is not None for t in v.values()): delattr(obj, k); n += 1
        return n
    n = purge(lirpa) + sum(purge(node) for node in lirpa.nodes())
    return n

def cmd_eval_pq(a):
    """Route A: per-query gauge optimisation on TEST instances.  For each instance and each eps in --eps_list: stock lb, fixed
    learned-gauge lb, and the per-query lb (Adam from the learned gauge, --pq_steps steps, early stop once verified unless
    --pq_stop_verified 0).  With --pq_radius 1 also: stock / fixed-gauge certified radius, then the gauge optimised at the fixed
    gauge's radius r_f and a bisection on [r_f, hi] with that gauge frozen (sound: its lb at r_f is >= the fixed gauge's > 0).
    Progressive JSON save + resume (--save_json), so a pre-empted job continues where it stopped."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m, tok, net = build(a.name, dev); L, H, dh = len(net.layers), net.H, net.dh; st = [[t.to(dev) for t in w] for w in stock_tensors(net)]
    data = load_data(a, "test"); S = short_instances(net, m, tok, data, a.max_len, a.n_sent, seed=a.seed); lirpas = make_lirpas(net, [e.shape[1] for _, _, e, _ in S], dev, a.softmax)
    inst = [(j, i, e, ex["label"], toks) for j, ex, e, toks in S for i in positions(toks)]; print(f"# {a.name}: {len(S)} test sentences <= {a.max_len} tokens, {len(inst)} instances; per-query steps {a.pq_steps}, lr {a.pq_lr}", flush=True)
    for p_ in leaves(net): p_.requires_grad_(True)
    Gq0, Ga0 = load_gauge(a.gauge, L, H, dh); I64 = eye_gauge(L, H, dh, torch.float64); eps_list = [float(x) for x in a.eps_list.split(",")]
    out = {"inst": [(j, i, e.shape[1], y) for j, i, e, y, _ in inst], "eps_list": eps_list, "args": vars(a), "rec": {}}
    if a.save_json and os.path.exists(a.save_json):
        prev = json.load(open(a.save_json)); out["rec"] = prev.get("rec", {}); print(f"# resuming: {len(out['rec'])} instances already done", flush=True)
        oom_len = [out["inst"][int(k)][2] for k, r in out["rec"].items() if any(isinstance(v, dict) and v.get("oom") for v in r.values())]
        if oom_len and min(oom_len) - 1 < a.pq_max_tokens:   # adaptive cap: a length that ran out of memory once will again; stop trying it (and longer)
            a.pq_max_tokens = min(oom_len) - 1; print(f"# per-query token cap lowered to {a.pq_max_tokens} (an instance of length {min(oom_len)} ran out of memory earlier)", flush=True)
    def fixed_lb(gq, ga, lp, e, i, eps, y): load_eff(net, fold64(st, gq, ga, H, dh)); return crown_lb(lp, e, i, eps, y, dev)
    t_all = time.time()
    for idx, (j, i, e, y, toks) in enumerate(inst):
        if str(idx) in out["rec"]: continue
        lp = lirpas[e.shape[1]]; rec = {}; t0 = time.time()
        for eps in eps_list:
            s_ = fixed_lb(I64[0], I64[1], lp, e, i, eps, y); f_ = fixed_lb(Gq0, Ga0, lp, e, i, eps, y)
            if f_ > 0 and a.pq_stop_verified: pq = (f_, 0, 0)
            else:
                r_ = pq_safe(net, st, lp, e, i, eps, y, dev, Gq0, Ga0, H, dh, a.pq_steps, a.pq_lr, a.cond_pen, a.clip, stop_at=0.0 if a.pq_stop_verified else None, max_tokens=a.pq_max_tokens)
                if r_ is None: pq = (f_, -1, 0)                                              # OOM / token cap: per-query := fixed
                elif not np.isfinite(r_[0]): pq = (f_, -2, r_[3])                            # NaN cliff: no finite iterate at all (fixed is NaN too) -- not an OOM
                else: pq = (r_[0], r_[2], r_[3])
            rec[str(eps)] = {"stock": s_, "fixed": f_, "pq": pq[0], "pq_best_step": pq[1], "pq_steps_run": pq[2], "oom": pq[1] == -1}
        if a.pq_radius:
            load_eff(net, fold64(st, I64[0], I64[1], H, dh)); r_s = certified_radius(lp, e, i, y, dev, hi=a.hi, iters=a.iters)
            load_eff(net, fold64(st, Gq0, Ga0, H, dh)); r_f = certified_radius(lp, e, i, y, dev, hi=a.hi, iters=a.iters)
            m0 = crown_lb(lp, e, i, r_f, y, dev)
            r_ = pq_safe(net, st, lp, e, i, r_f, y, dev, Gq0, Ga0, H, dh, a.pq_steps, a.pq_lr, a.cond_pen, a.clip, stop_at=None, max_tokens=a.pq_max_tokens)
            if r_ is None: v, bs, r_pq = m0, -1, r_f
            elif not np.isfinite(r_[0]): v, bs, r_pq = m0, -2, r_f
            else:
                v, G, bs, run = r_; load_eff(net, effective(st, G[0], G[1], H, dh)); r_pq = certified_radius(lp, e, i, y, dev, lo=r_f, hi=a.hi, iters=a.iters) if r_f < a.hi else r_f
            rec["radius"] = {"stock": r_s, "fixed": r_f, "pq": r_pq, "margin_at_rf_fixed": m0, "margin_at_rf_pq": v, "pq_best_step": bs, "oom": bs == -1}
        out["rec"][str(idx)] = rec
        if a.save_json: dump_json(out, open(a.save_json, "w"))
        if PQ_OOM:
            print(f"  inst {idx:3d} saved (per-query := fixed after OOM); exiting 3 for a clean restart (resume from the JSON)", flush=True); sys.exit(3)
        ran_heavy = e.shape[1] >= a.pq_restart_len and any(isinstance(v, dict) and v.get("pq_steps_run", 0) > 0 or (isinstance(v, dict) and v.get("pq_best_step", 0) not in (-1, 0)) for v in rec.values())
        if a.pq_restart_len and ran_heavy:   # a grad-mode optimisation on a long instance leaves ~20 GB behind that no cache/bound clearing recovers: restart the process (resume from the JSON)
            print(f"  inst {idx:3d} saved; exiting 3 for a clean restart after a {e.shape[1]}-token optimisation (--pq_restart_len {a.pq_restart_len})", flush=True); sys.exit(3)
        msg = "; ".join(f"eps {eps}: {r['stock']:+.3f} / {r['fixed']:+.3f} / {r['pq']:+.3f} ({r['pq_steps_run']} st)" for eps, r in rec.items() if eps != "radius")
        if "radius" in rec: r = rec["radius"]; msg += f"; radius {r['stock']:.4f} / {r['fixed']:.4f} / {r['pq']:.4f} (margin at r_f {r['margin_at_rf_fixed']:+.3f} -> {r['margin_at_rf_pq']:+.3f})"
        mem = f", peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if torch.cuda.is_available() else ""; torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        print(f"  inst {idx:3d} (sent {j} pos {i} len {e.shape[1]}): stock / fixed / per-query -- {msg}  [{time.time()-t0:.0f}s, total {time.time()-t_all:.0f}s{mem}]", flush=True)
    R = out["rec"]; n = len(R); oom = sorted({int(x) for x in R for k in R[x] if R[x][k].get("oom")})
    if oom: print(f"# per-query optimisation skipped (per-query := fixed gauge) on {len(oom)} of {n} instances (lengths {sorted({out['inst'][x][2] for x in oom})}): --pq_max_tokens {a.pq_max_tokens} or CUDA OOM; grad-mode CROWN memory grows ~n^3 and 80 GB holds 11 tokens of this model", flush=True)
    for eps in eps_list:
        k = str(eps); s_ = np.array([R[x][k]["stock"] for x in R]); f_ = np.array([R[x][k]["fixed"] for x in R]); q_ = np.array([R[x][k]["pq"] for x in R])
        print(f"# eps {eps} over {n}: verified stock {(s_ > 0).sum()} / fixed gauge {(f_ > 0).sum()} / per-query {(q_ > 0).sum()}; per-query lb > fixed on {(q_ > f_).sum()}, mean lb {np.nanmean(s_):+.3f} / {np.nanmean(f_):+.3f} / {np.nanmean(q_):+.3f}", flush=True)
    if a.pq_radius and n:
        rs = np.array([R[x]["radius"]["stock"] for x in R]); rf = np.array([R[x]["radius"]["fixed"] for x in R]); rq = np.array([R[x]["radius"]["pq"] for x in R])
        print(f"# radius over {n}: mean stock {rs.mean():.4f} / fixed {rf.mean():.4f} ({100*(rf.mean()/rs.mean()-1):+.1f}%) / per-query {rq.mean():.4f} ({100*(rq.mean()/rs.mean()-1):+.1f}% vs stock, {100*(rq.mean()/rf.mean()-1):+.1f}% vs fixed); per-query > fixed on {(rq > rf).sum()}, equal {(rq == rf).sum()}", flush=True)

def load_gauge(path, L, H, dh):
    if path is None: return eye_gauge(L, H, dh, torch.float64)
    g_ = torch.load(path); return g_["qk"], g_["av"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["probe", "radii", "learn", "eval", "eval_alpha", "attrib", "eval_pq"])
    ap.add_argument("--split", default="dev"); ap.add_argument("--pos_per_sent", type=int, default=3); ap.add_argument("--eps_scale", type=float, default=1.0); ap.add_argument("--radius_iters", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150); ap.add_argument("--accum", type=int, default=4); ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--cond_pen", type=float, default=1e-4); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--which", default="both"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log_every", type=int, default=10); ap.add_argument("--debug", type=int, default=0); ap.add_argument("--n_eval", type=int, default=48); ap.add_argument("--label", type=int, default=-1, help="learn: keep only tuning boxes with this label (-1 = all)")
    ap.add_argument("--out", default=None); ap.add_argument("--gauge", default=None); ap.add_argument("--alpha_iters", type=int, default=0, help="learn: >0 = alternating alpha-CROWN / gauge optimisation with this many alpha iterations per box"); ap.add_argument("--alpha_lr", type=float, default=0.1); ap.add_argument("--alpha_shared", type=int, default=0); ap.add_argument("--skip_stock", type=int, default=0, help="eval_alpha: only the gauged weight set (join stock from an earlier file)"); ap.add_argument("--resume", type=int, default=1, help="learn: continue from <out>.ckpt if present"); ap.add_argument("--ckpt_every", type=int, default=5); ap.add_argument("--eps_list", default="0.01,0.02,0.03")
    ap.add_argument("--obj", default="mean", help="mean | hinge"); ap.add_argument("--hinge_w", type=float, default=4.0)
    ap.add_argument("--max_len", type=int, default=12); ap.add_argument("--n_sent", type=int, default=20); ap.add_argument("--alpha", type=int, default=1); ap.add_argument("--hi", type=float, default=0.1); ap.add_argument("--iters", type=int, default=10); ap.add_argument("--save_json", default=None); ap.add_argument("--name", default="sst_bert_small_3"); ap.add_argument("--data", default="auto", help="sst | yelp | auto (from --name prefix)"); ap.add_argument("--eps", type=float, default=0.03); ap.add_argument("--softmax", default="lse"); ap.add_argument("--crown_batch", type=int, default=512)
    ap.add_argument("--split_attrib", type=int, default=0, help="attrib: also linearise ONE nonlinearity at a time (qk / softmax / av) and all three"); ap.add_argument("--factors", default="1,1.5", help="attrib: eps as multiples of the stock radius")
    ap.add_argument("--pq_steps", type=int, default=20); ap.add_argument("--pq_lr", type=float, default=0.01); ap.add_argument("--pq_stop_verified", type=int, default=1); ap.add_argument("--pq_radius", type=int, default=0); ap.add_argument("--pq_restart_len", type=int, default=10, help="eval_pq: exit 3 (chain restarts, resume) after any optimisation on an instance this long or longer; 0 = never"); ap.add_argument("--pq_max_tokens", type=int, default=11, help="eval_pq: skip the per-query optimisation (per-query := fixed gauge) on longer instances; 0 = no cap")
    ap.add_argument("--weight_intervals", type=int, default=0, help="eval: also verify stock and gauged weights declared as 2-ulp fp32 intervals (rigorous transfer to the original network)")
    ap.add_argument("--k_words", type=int, default=1, help="perturb k embedding rows at once (1 = DeepT's one-word spec; 2 = two-word)"); ap.add_argument("--pairs_per_sent", type=int, default=7, help="eval: position sets per sentence when k_words > 1")
    a = ap.parse_args(); {"probe": cmd_probe, "radii": cmd_radii, "learn": cmd_learn, "eval": cmd_eval, "eval_alpha": cmd_eval_alpha, "attrib": cmd_attrib, "eval_pq": cmd_eval_pq}[a.cmd](a)
