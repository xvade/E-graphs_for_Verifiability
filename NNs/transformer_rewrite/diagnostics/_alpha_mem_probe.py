"""Why does alpha-CROWN on small_6 OOM at 6 tokens?  Hypothesis: eval_alpha sets requires_grad on the WEIGHTS (copied from the
learner), which makes autograd save every CROWN A-matrix for weight gradients that alpha optimisation never uses.
Measures peak GPU memory and the bound with weights requires_grad True vs False (small_3, 6 tokens: bounds must agree), then
small_6 at 6 and 7 tokens with weights frozen.  Run from NNs/transformer_rewrite on a GPU node.
RESULT (2026-09-06 03:50, L40S): hypothesis REJECTED — small_3 6 tokens: 14.6 GiB (weights grad) vs 16.4 GiB (frozen), identical
bounds; small_6 OOMs (>43.6 GiB) at 6, 7 and 8 tokens either way.  alpha-CROWN on the 6-layer model needs a bigger GPU or <= 5 tokens."""
import sys, time, torch, numpy as np; sys.argv = ["x"]; sys.path.insert(0, "."); import deept_gauge as g
dev = "cuda"
def run(name, n_tok, wgrad, eps):
    m, tok, net = g.build(name, dev); data = g.load_sst("test"); S = [s for s in g.short_instances(net, m, tok, data, n_tok) if s[2].shape[1] == n_tok][:1]
    if not S: print(f"# {name} {n_tok} tokens: no test sentence of exactly that length"); return
    j, ex, e, toks = S[0]; i = g.positions(toks)[0]; lp = g.make_lirpas(net, [n_tok], dev, "lse", alpha=True)[n_tok]
    for p in g.leaves(net): p.requires_grad_(wgrad)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache(); t0 = time.time()
    try: lb = g.crown_lb(lp, e, i, eps, ex["label"], dev, method="CROWN-Optimized", grad=True); c = g.crown_lb(lp, e, i, eps, ex["label"], dev)
    except torch.OutOfMemoryError: print(f"# {name} {n_tok} tokens wgrad={wgrad}: OOM (peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB)", flush=True); return
    print(f"# {name} {n_tok} tokens wgrad={wgrad}: alpha lb {lb:+.5f} (CROWN {c:+.5f}) peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB [{time.time()-t0:.0f}s]", flush=True)
run("sst_bert_small_3", 6, True, 0.03); run("sst_bert_small_3", 6, False, 0.03)
run("sst_bert_small_6", 6, False, 0.02); run("sst_bert_small_6", 7, False, 0.02); run("sst_bert_small_6", 8, False, 0.02)
print("PROBE_DONE")
