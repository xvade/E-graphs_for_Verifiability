"""alpha-CROWN (CROWN-Optimized, 20 it) peak memory and time per call on sst_bert_big_3 (hidden 256, 3 layers) at 6/7/8 tokens,
one fresh BoundedModule per call as cmd_eval_alpha does.  Decides the max_len of the single alpha-tier comparison and whether it
fits a 48 GB L40S.  Run from NNs/transformer_rewrite on an 80 GB GPU:  python diagnostics/_alpha_mem_probe_big3.py [name] [tokens...]"""
import sys, time, gc, torch; args = sys.argv[1:]; sys.argv = ["x"]; sys.path.insert(0, "."); import deept_gauge as g
name = args[0] if args else "sst_bert_big_3"; toks = [int(t) for t in args[1:]] or [6, 7, 8]; dev = "cuda"; GiB = 2**30
m, tok, net = g.build(name, dev); data = g.load_sst("test")
for n_tok in toks:
    S = [s for s in g.short_instances(net, m, tok, data, n_tok) if s[2].shape[1] == n_tok][:1]
    inst = [(e, i, ex["label"]) for j, ex, e, toks_ in S for i in g.positions(toks_)][:2]
    for k, (e, i, y) in enumerate(inst):
        lp = g.make_lirpas(net, [n_tok], dev, "lse", alpha=True)[n_tok]
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        try: lb = g.crown_lb(lp, e, i, 0.03, y, dev, method="CROWN-Optimized", grad=True); msg = f"lb {lb:+.4f}"
        except torch.OutOfMemoryError: msg = "OOM"
        print(f"# {name} {n_tok} tok inst {k}: {msg}; peak {torch.cuda.max_memory_allocated()/GiB:.1f} GiB [{time.time()-t0:.0f}s]", flush=True)
        del lp; gc.collect(); torch.cuda.empty_cache()
        if msg == "OOM": break
print("PROBE_DONE")
