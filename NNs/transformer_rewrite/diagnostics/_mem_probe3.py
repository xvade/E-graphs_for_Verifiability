import sys, time, torch, numpy as np; sys.argv = ["x"]; sys.path.insert(0, ".")  # run from NNs/transformer_rewrite (see run_*.sh); import deept_gauge as g
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.operators import linear as L
orig = L.BoundLinear.propagate_A_xy.__func__ if hasattr(L.BoundLinear.propagate_A_xy, "__func__") else L.BoundLinear.propagate_A_xy
worst = [0, None]
def logged(last_A, alpha_pos, alpha_neg, beta_pos, beta_neg, dim_y):
    prod = int(np.prod(last_A.shape)) * beta_pos.shape[-2] if beta_pos.dim() == last_A.dim() else int(np.prod(torch.broadcast_shapes(last_A.unsqueeze(-1).shape, beta_pos.shape)))
    if prod > worst[0]: worst[0] = prod; worst[1] = (tuple(last_A.shape), tuple(alpha_pos.shape), tuple(beta_pos.shape), dim_y)
    return orig(last_A, alpha_pos, alpha_neg, beta_pos, beta_neg, dim_y)
L.BoundLinear.propagate_A_xy = staticmethod(logged)
dev = "cuda"; m, tok, net = g.build("sst_bert_small_3", dev); data = g.load_sst("test")
bylen = {}
for ex in data:
    e, toks = g.embed(m, tok, ex); bylen.setdefault(e.shape[1], (ex, e, toks))
n = 12; ex, e, toks = bylen[n]; i = g.positions(toks)[0]
lp = BoundedModule(net, torch.empty(1, n, 128, device=dev), bound_opts={"softmax": "lse"}, device=dev)
torch.cuda.reset_peak_memory_stats(); lb = g.crown_lb(lp, e, i, 0.03, ex["label"], dev, method="CROWN")
print(f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB lb {lb:+.4f}; worst bilinear A_y product: {worst[0]/2**28:.2f} GiB (fp32) shapes last_A {worst[1][0]} alpha {worst[1][1]} beta {worst[1][2]} dim_y {worst[1][3]}")
# which node feeds these shapes: list MatMul nodes with input shapes
for nm, nd in lp._modules.items():
    if type(nd).__name__ in ("BoundMatMul", "BoundLinear") and all(getattr(i_, "perturbed", False) for i_ in nd.inputs[:2]): print("  bilinear", nm, [tuple(i_.output_shape) for i_ in nd.inputs[:2]], "->", tuple(nd.output_shape))
