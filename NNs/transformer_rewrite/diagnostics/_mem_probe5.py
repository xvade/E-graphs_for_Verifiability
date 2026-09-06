import sys, time, torch, numpy as np; sys.argv = ["x"]; sys.path.insert(0, ".")  # run from NNs/transformer_rewrite (see run_*.sh); import deept_gauge as g
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
dev = "cuda"; m, tok, net = g.build("sst_bert_small_3", dev); data = g.load_sst("test")
bylen = {}
for ex in data:
    e, toks = g.embed(m, tok, ex); bylen.setdefault(e.shape[1], (ex, e, toks))
n = 12; ex, e, toks = bylen[n]; i = g.positions(toks)[0]
for tag, opts, ng in [("lse grad", {"softmax": "lse"}, False), ("lse no_grad", {"softmax": "lse"}, True), ("complex no_grad", {"softmax": "complex", "fixed_reducemax_index": True}, True), ("lse no_grad sparse_interm=False", {"softmax": "lse", "sparse_intermediate_bounds": False}, True)]:
    torch.cuda.empty_cache(); lp = BoundedModule(net, torch.empty(1, n, 128, device=dev), bound_opts=opts, device=dev); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    try:
        if ng:
            with torch.no_grad(): lb = g.crown_lb(lp, e, i, 0.03, ex["label"], dev, method="CROWN")
        else: lb = g.crown_lb(lp, e, i, 0.03, ex["label"], dev, method="CROWN")
        print(f"n={n} {tag:32s}: lb {lb:+.4f} {time.time()-t0:5.1f}s peak {torch.cuda.max_memory_allocated()/2**30:6.2f} GiB", flush=True)
    except Exception as ex_: print(f"n={n} {tag:32s}: FAILED {type(ex_).__name__} {str(ex_)[:100]}", flush=True)
    del lp
