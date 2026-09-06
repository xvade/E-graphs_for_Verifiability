"""alpha-CROWN on small_6 OOM'd even on an 80 GB A100 at 6 tokens (job 39662797).  Is the peak per instance really > 80 GB, or does
memory GROW across consecutive instances (references kept inside the BoundedModule between calls)?  Runs 4 consecutive
instances at 5 and 6 tokens, printing allocated memory before/after each call and the per-call peak, reusing one BoundedModule
(as eval_alpha does) and then rebuilding it per instance.  Run from NNs/transformer_rewrite on an 80 GB GPU."""
import sys, time, gc, torch, numpy as np; sys.argv = ["x"]; sys.path.insert(0, "."); import deept_gauge as g
dev = "cuda"; GiB = 2**30
def probe(name, n_tok, eps, rebuild):
    m, tok, net = g.build(name, dev); data = g.load_sst("test"); S = [s for s in g.short_instances(net, m, tok, data, n_tok) if s[2].shape[1] == n_tok][:2]
    inst = [(e, i, ex["label"]) for j, ex, e, toks in S for i in g.positions(toks)][:4]
    lp = None if rebuild else g.make_lirpas(net, [n_tok], dev, "lse", alpha=True)[n_tok]
    for k, (e, i, y) in enumerate(inst):
        if rebuild: lp = g.make_lirpas(net, [n_tok], dev, "lse", alpha=True)[n_tok]
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); before = torch.cuda.memory_allocated() / GiB; t0 = time.time()
        try: lb = g.crown_lb(lp, e, i, eps, y, dev, method="CROWN-Optimized", grad=True); msg = f"lb {lb:+.4f}"
        except torch.OutOfMemoryError: msg = "OOM"
        print(f"# {name} {n_tok} tok rebuild={rebuild} inst {k}: {msg}; allocated before {before:.2f} GiB, after {torch.cuda.memory_allocated()/GiB:.2f} GiB, peak {torch.cuda.max_memory_allocated()/GiB:.1f} GiB [{time.time()-t0:.0f}s]", flush=True)
        if msg == "OOM": break
    del lp, net; gc.collect(); torch.cuda.empty_cache()
probe("sst_bert_small_6", 5, 0.02, False); probe("sst_bert_small_6", 5, 0.02, True); probe("sst_bert_small_6", 6, 0.02, False)
print("PROBE_DONE")
