import sys, time, torch, numpy as np; sys.argv = ["x"]; sys.path.insert(0, ".")  # run from NNs/transformer_rewrite (see run_*.sh); import deept_gauge as g
from auto_LiRPA import BoundedModule
dev = "cuda"; m, tok, net = g.build("sst_bert_small_3", dev); data = g.load_sst("test")
S = g.short_instances(net, m, tok, data, 12, None); bylen = {}
for s in S: bylen.setdefault(s[2].shape[1], s)
print("available lengths:", sorted(bylen), flush=True)
# (1) NaN cliff check, no grad, lse vs complex
for mode in []:
    opts = {"softmax": mode, "sparse_intermediate_bounds": False}
    if mode == "complex": opts["fixed_reducemax_index"] = True
    for n in [9, 11]:
        j, ex, e, toks = bylen[n]; lp = BoundedModule(net, torch.empty(1, n, 128, device=dev), bound_opts=opts, device=dev)
        for i in g.positions(toks)[:2]:
            r = g.certified_radius(lp, e, i, ex["label"], dev, hi=0.1, iters=10)
            lbs = [g.crown_lb(lp, e, i, f * r, ex["label"], dev) for f in (0.9, 1.0, 1.02, 1.5, 2.0, 4.0)]
            print(f"{mode} sent {j} len {n} pos {i}: radius {r:.4f}; lb at 0.9r/r/1.02r/1.5r/2r/4r = {['%+.3f' % v for v in lbs]}", flush=True)
        del lp; torch.cuda.empty_cache()
# (2) alpha-CROWN with grad: memory vs n
for n in [6, 8, 10]:
    if n not in bylen: continue
    j, ex, e, toks = bylen[n]; i = g.positions(toks)[0]
    lp = BoundedModule(net, torch.empty(1, n, 128, device=dev), bound_opts={"softmax": "lse", "sparse_intermediate_bounds": False, "optimize_bound_args": {"iteration": 20, "lr_alpha": 0.1}}, device=dev)
    r = g.certified_radius(lp, e, i, ex["label"], dev, hi=0.1, iters=10); lb = g.crown_lb(lp, e, i, 0.9 * r, ex["label"], dev)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    try:
        for p in g.leaves(net): p.requires_grad_(True)
        la = g.crown_lb(lp, e, i, 0.9 * r, ex["label"], dev, method="CROWN-Optimized", grad=True)
        print(f"alpha n={n}: CROWN {lb:+.4f} -> alpha-CROWN {la:+.4f}  [{time.time()-t0:.1f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB]", flush=True)
    except Exception as ex_: print(f"alpha n={n}: FAILED {type(ex_).__name__} peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    for p in g.leaves(net): p.requires_grad_(False)
    del lp; torch.cuda.empty_cache()
