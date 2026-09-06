import sys, time, torch, numpy as np, collections; sys.argv = ["x"]; sys.path.insert(0, ".")  # run from NNs/transformer_rewrite (see run_*.sh); import deept_gauge as g
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
dev = "cuda"; m, tok, net = g.build("sst_bert_small_3", dev); data = g.load_sst("test")
bylen = {}
for ex in data:
    e, toks = g.embed(m, tok, ex); bylen.setdefault(e.shape[1], (ex, e, toks))
n = 12; ex, e, toks = bylen[n]; i = g.positions(toks)[0]
lp = BoundedModule(net, torch.empty(1, n, 128, device=dev), bound_opts={"softmax": "lse"}, device=dev)
stats = collections.defaultdict(lambda: [0, 0.0, None])   # node -> [calls, max peak delta GiB, shapes]
def wrap(node):
    f = node.bound_backward
    def w(last_lA, last_uA, *args, **kw):
        base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        out = f(last_lA, last_uA, *args, **kw)
        d = (torch.cuda.max_memory_allocated() - base) / 2**30; s = stats[f"{node.name} {type(node).__name__}"]
        if d > s[1]: s[1] = d; s[2] = (tuple(last_lA.shape) if isinstance(last_lA, torch.Tensor) else type(last_lA).__name__, [tuple(x.output_shape) for x in node.inputs[:2]])
        s[0] += 1; return out
    node.bound_backward = w
for nd in lp._modules.values(): wrap(nd)
torch.cuda.reset_peak_memory_stats(); lb = g.crown_lb(lp, e, i, 0.03, ex["label"], dev, method="CROWN")
print(f"lb {lb:+.4f}")
for k, (c, d, sh) in sorted(stats.items(), key=lambda kv: -kv[1][1])[:10]: print(f"  peak delta {d:6.2f} GiB  calls {c:4d}  {k}  last_lA {sh[0]} inputs {sh[1]}")
