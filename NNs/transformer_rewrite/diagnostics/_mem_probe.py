import sys, time, torch, numpy as np; sys.argv = ["x"]; sys.path.insert(0, ".")  # run from NNs/transformer_rewrite (see run_*.sh); import deept_gauge as g
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
dev = "cuda"; m, tok, net = g.build("sst_bert_small_3", dev); data = g.load_sst("test")
# pick test sentences of specific lengths
bylen = {}
for ex in data:
    e, toks = g.embed(m, tok, ex); bylen.setdefault(e.shape[1], (ex, e, toks))
for n in [6, 8, 12, 16, 20]:
    ex, e, toks = bylen[n]; i = g.positions(toks)[0]
    for method in ["IBP+backward", "forward+backward", "CROWN"]:
        for cb in ([1 << 30, 256] if method == "CROWN" else [1 << 30]):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            try:
                lp = BoundedModule(net, torch.empty(1, n, 128, device=dev), bound_opts={"softmax": "lse", "crown_batch_size": cb, "batched_crown_max_vram_ratio": 0.9}, device=dev)
                lb = g.crown_lb(lp, e, i, 0.03, ex["label"], dev, method=method)
                print(f"n={n:2d} {method:17s} cb={cb:>10d}: lb {lb:+.4f}  {time.time()-t0:5.1f}s  peak {torch.cuda.max_memory_allocated()/2**30:6.2f} GiB", flush=True)
            except Exception as ex_:
                print(f"n={n:2d} {method:17s} cb={cb:>10d}: FAILED {type(ex_).__name__} {str(ex_)[:80]}  peak {torch.cuda.max_memory_allocated()/2**30:6.2f} GiB", flush=True)
            del lp; torch.cuda.empty_cache()
