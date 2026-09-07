"""Learn an attention gauge AGAINST HUANG ET AL.'S VERIFIER (PBVerification, Shi-style backward bounds) instead of auto_LiRPA.

The gauge is the same exact per-head reparametrisation as in deept_gauge.py (W_q <- G^T W_q, W_k <- G^-1 W_k, W_v <- Ga^T W_v,
W_o <- W_o Ga^-T; their attention has no Q/K/V biases).  The objective is the margin lower bound produced by THEIR code
(`--version origin` = their Baseline = Shi et al. 2020; `--version inner` = midpoint tangent planes, the unoptimised centre of
their PBverifierT family) on tuning boxes at each box's stock certified radius under that same bound, differentiated through
their bound computation w.r.t. the folded weights (a copy of their package with the `final_lb` detach removed:
deept_benchmarks/PBVerification_grad/).  Output: gauges/<out>.pt in the same {"qk","av"} format, so export_gauged_ckpt.py folds it.

    python pbv_learn.py --ckpt <shi-layout dir> --version origin --split dev --max_len 8 --n_sent 40 --pos_per_sent 3 \
        --steps 120 --accum 4 --lr 0.01 --seed 0 --out gauges/pbvtrained_small6_origin_seed0.pt
"""
import argparse, copy, os, random, sys, time
import numpy as np, torch
REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; PBV = os.path.join(REPO, "deept_benchmarks/PBVerification_grad")
DATA = os.path.join(REPO, "deept_benchmarks/data")
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--version", default="origin", choices=["origin", "inner"])
ap.add_argument("--split", default="dev"); ap.add_argument("--max_len", type=int, default=8); ap.add_argument("--n_sent", type=int, default=40); ap.add_argument("--pos_per_sent", type=int, default=3)
ap.add_argument("--steps", type=int, default=120); ap.add_argument("--accum", type=int, default=4); ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--cond_pen", type=float, default=1e-4)
ap.add_argument("--clip", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_eval", type=int, default=48); ap.add_argument("--log_every", type=int, default=10)
ap.add_argument("--anomaly", type=int, default=0, help="1: run one box under torch.autograd.detect_anomaly and report the op that produces NaN grads, then exit"); ap.add_argument("--hi", type=float, default=0.1); ap.add_argument("--radius_iters", type=int, default=8); ap.add_argument("--out", required=True); ap.add_argument("--which", default="both")
a = ap.parse_args()
os.chdir(PBV); sys.path.insert(0, PBV); sys.path.insert(0, os.path.join(PBV, "pylib"))
from Parser import Parser, update_arguments
from Models import Transformer
from Verifiers import VerifierBackward
from Verifiers.Edge import EdgeDense

args, _ = Parser().getParser().parse_known_args(["--verify", "--dir", a.ckpt, "--data", "sst", "--method", "backward", "--p", "10", "--perturbed_words", "1", "--version", a.version, "--seed", str(a.seed)])
args = update_arguments(args); dev = args.device
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
target = Transformer(args, []); model = target.model.eval()
for p in model.parameters(): p.requires_grad_(False)
V = VerifierBackward(args, target, None)
layers = model.bert.encoder.layer; L = len(layers); H = layers[0].attention.self.num_attention_heads; dh = layers[0].attention.self.attention_head_size
print(f"# {a.ckpt}: {L} layers, {H} heads, d_head {dh}, layer_norm {V.layer_norm}, version {a.version}, device {dev}", flush=True)

# ---- data: DeepT's SST loader + their get_embeddings (identical tokenisation to their verify run) ----
def load_sst(split):
    data = []
    for line in open(os.path.join(DATA, "sst", f"{split}.txt")):
        segs = line[:-1].split(" "); label = int(segs[0][1])
        if label == 2: continue
        label = 0 if label < 2 else 1; tokens = []
        for i in range(len(segs) - 1):
            if segs[i][0] == "(" and segs[i][1] in "01234" and segs[i + 1][0] != "(": tokens.append(segs[i + 1][:segs[i + 1].find(")")])
        tokens = ["(" if t == "-LRB-" else ")" if t == "-RRB-" else t for t in tokens]
        data.append({"label": label, "sent_a": tokens})
    return data
def positions(toks): return [i for i in range(1, len(toks) - 1) if not (toks[i][0] == "#" or toks[i + 1][0] == "#")]

# ---- gauge parametrisation folded into their modules (plain tensors replace the Parameters) ----
lins = [(l.attention.self.query, l.attention.self.key, l.attention.self.value, l.attention.output.dense) for l in layers]
stock = [[q.weight.detach().clone(), k.weight.detach().clone(), v.weight.detach().clone(), o.weight.detach().clone()] for q, k, v, o in lins]
assert all(q.bias is None and k.bias is None and v.bias is None for q, k, v, o in lins), "expected bias-free Q/K/V (their modeling.py)"
for q, k, v, o in lins:
    for lin in (q, k, v, o): del lin._parameters["weight"]
Gq = torch.eye(dh, device=dev).expand(L, H, dh, dh).clone().requires_grad_(a.which in ("both", "qk"))
Ga = torch.eye(dh, device=dev).expand(L, H, dh, dh).clone().requires_grad_(a.which in ("both", "av"))
def apply_gauge(Gq, Ga):
    for l, (q, k, v, o) in enumerate(lins):
        Wq, Wk, Wv, Wo = stock[l]; qs, ks, vs, os_ = [], [], [], []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh); G = Gq[l, h]; A = Ga[l, h]
            qs.append(G.T @ Wq[sl]); ks.append(torch.linalg.inv(G) @ Wk[sl]); vs.append(A.T @ Wv[sl]); os_.append(Wo[:, sl] @ torch.linalg.inv(A).T)
        q.weight = torch.cat(qs, 0); k.weight = torch.cat(ks, 0); v.weight = torch.cat(vs, 0); o.weight = torch.cat(os_, 1)
apply_gauge(Gq, Ga)

# ---- their bound, without the no_grad / attack wrapper, returning the margin lower bound ----
def margin(emb, index, eps, label):
    bounds = V._bound_input(emb, index=[index], eps=eps)
    for layer in V.encoding_layers: _, _, bounds = V._bound_layer(bounds, layer, QK_output=None)
    bounds = V._bound_pooling(bounds, V.pooler)
    with torch.no_grad():
        cls = copy.deepcopy(V.classifier)
        for p in cls.parameters(): p.requires_grad_(False)
        cls.weight[0, :] -= cls.weight[1, :]; cls.bias[0] -= cls.bias[1]
    bounds = bounds.next(EdgeDense(args, V.controller, bounds, dense=cls), dim=cls.weight.shape[0])
    m = bounds.l[0][0] if label == 0 else -bounds.u[0][0]
    for lay in V.controller.layers:
        for attr in ("lw", "uw", "final_lw", "final_uw", "final_lb", "final_ub"):
            if hasattr(lay, attr): delattr(lay, attr)
    return m
def margin_ng(emb, index, eps, label):
    with torch.no_grad(): return margin(emb, index, eps, label).item()
def radius(emb, index, label):
    if margin_ng(emb, index, a.hi, label) > 0: return a.hi
    lo, hi = 0.0, a.hi
    for _ in range(a.radius_iters):
        mid = (lo + hi) / 2
        if margin_ng(emb, index, mid, label) > 0: lo = mid
        else: hi = mid
    return lo

# ---- exactness check of the fold on random gauges (their forward) ----
data = load_sst(a.split); boxes = []; t0 = time.time()
for j, ex in enumerate(data):
    emb, toks = target.get_embeddings([ex]); toks = toks[0]
    if emb.shape[1] > a.max_len or not positions(toks): continue
    pred = target.step([ex])[-1]["pred_labels"][0]
    if pred != ex["label"]: continue
    for i in positions(toks)[:a.pos_per_sent]: boxes.append([emb[0].detach().to(dev), i, ex["label"], emb.shape[1], None, j])
    if len({b[5] for b in boxes}) >= a.n_sent: break
print(f"# {len({b[5] for b in boxes})} {a.split} sentences <= {a.max_len} tokens -> {len(boxes)} tuning boxes  [{time.time()-t0:.0f}s]", flush=True)
t0 = time.time()
for b in boxes: b[4] = radius(b[0], b[1], b[2])
R = np.array([b[4] for b in boxes]); print(f"# per-box eps = stock certified radius under THEIR {a.version} bound: mean {R.mean():.4f} median {np.median(R):.4f} min {R.min():.4f} max {R.max():.4f}  [{time.time()-t0:.0f}s]", flush=True)
ev_boxes = boxes[:a.n_eval]
if a.anomaly:
    import traceback
    for bi, b in enumerate(boxes[:3]):
        apply_gauge(Gq, Ga)
        try:
            with torch.autograd.detect_anomaly():
                lb = margin(b[0], b[1], b[4], b[2]); print(f"# anomaly box {bi}: lb {lb.item():+.4f}, len {b[3]}, pos {b[1]}, eps {b[4]:.4f}", flush=True); lb.backward()
            gq = Gq.grad; print(f"# anomaly box {bi}: grad finite entries {int(torch.isfinite(gq).sum())}/{gq.numel()}, grad norm {gq[torch.isfinite(gq)].norm().item():.3e}", flush=True)
        except Exception as e:
            print(f"# anomaly box {bi}: {type(e).__name__}: {str(e)[:300]}", flush=True); traceback.print_exc()
        Gq.grad = None; Ga.grad = None
    sys.exit(0)
def evaluate():
    vals = [margin_ng(b[0], b[1], b[4], b[2]) for b in ev_boxes]; v = np.array(vals); return float(np.nanmean(v)), float(np.mean(v > 0)), int(np.isnan(v).sum())
ev = evaluate(); print(f"# init=id: eval on {len(ev_boxes)} tuning boxes at their eps: mean lb {ev[0]:+.4f} frac_ver {ev[1]:.3f} nan {ev[2]}", flush=True)
params = [p for p in (Gq, Ga) if p.requires_grad]; opt = torch.optim.Adam(params, lr=a.lr); rng = random.Random(a.seed)
best = (ev[0], (Gq.detach().double().cpu().clone(), Ga.detach().double().cpu().clone()), -1); skipped = 0; t0 = time.time()
for step in range(a.steps):
    opt.zero_grad(); objs = []
    for b in rng.sample(boxes, a.accum):
        apply_gauge(Gq, Ga); lb = margin(b[0], b[1], b[4], b[2]); objs.append(lb.item()); (-lb / a.accum).backward()
    if a.cond_pen > 0: (a.cond_pen * sum((p ** 2).sum() + (torch.linalg.inv(p) ** 2).sum() for p in params)).backward()
    # their relaxations divide by box widths; degenerate coordinates give NaN/inf gradient entries -> zero those entries (masked gradient)
    n_bad = 0
    for p_ in params:
        if p_.grad is not None:
            bad = ~torch.isfinite(p_.grad); n_bad += int(bad.sum()); p_.grad[bad] = 0.0
    gn = torch.nn.utils.clip_grad_norm_(params, a.clip); obj = float(np.mean(objs))
    if n_bad and (step % a.log_every == 0): print(f"  step {step:4d}: zeroed {n_bad} non-finite gradient entries", flush=True)
    if not torch.isfinite(gn) or not np.isfinite(obj): skipped += 1; print(f"  step {step:4d}: NON-FINITE (obj {obj}, grad_norm {gn.item()}) -> skipped", flush=True); opt.zero_grad(); continue
    opt.step()
    if step % a.log_every == 0 or step == a.steps - 1:
        with torch.no_grad(): apply_gauge(Gq, Ga); cond = max(torch.linalg.cond(p.reshape(-1, dh, dh)).max().item() for p in params)
        ev = evaluate(); print(f"  step {step:4d} batch_obj={obj:+.4f} grad_norm={gn.item():.3e} | eval mean lb {ev[0]:+.4f} frac_ver {ev[1]:.3f} nan {ev[2]} | max cond(G)={cond:.2f} | {time.time()-t0:.0f}s", flush=True)
        if ev[0] > best[0]: best = (ev[0], (Gq.detach().double().cpu().clone(), Ga.detach().double().cpu().clone()), step)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
torch.save({"qk": best[1][0], "av": best[1][1], "args": vars(a), "best_step": best[2], "skipped": skipped, "trained_against": f"PBVerification {a.version}"}, a.out)
print(f"# saved best gauges (step {best[2]}, eval mean lb {best[0]:+.4f}, skipped {skipped}) -> {a.out}; wall {time.time()-t0:.0f}s", flush=True)
# exactness of the fold under THEIR model forward (fp32): logits stock vs best gauge on the first 8 tuning sentences
def scores(ex):
    v = target.step([ex])[-1]["pred_scores"]; return np.asarray(v.detach().cpu() if torch.is_tensor(v) else v, dtype=np.float64)
with torch.no_grad():
    exs = [data[j] for j in list(dict.fromkeys(b[5] for b in boxes))[:8]]
    apply_gauge(torch.eye(dh, device=dev).expand(L, H, dh, dh).clone(), torch.eye(dh, device=dev).expand(L, H, dh, dh).clone()); ref = [scores(ex) for ex in exs]
    apply_gauge(best[1][0].float().to(dev), best[1][1].float().to(dev)); new = [scores(ex) for ex in exs]
    print(f"# fp32 fold check under their forward: max |logit change| = {max(float(np.abs(np.asarray(r) - np.asarray(n)).max()) for r, n in zip(ref, new)):.2e}", flush=True)
