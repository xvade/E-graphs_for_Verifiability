import sys, math, torch, collections, numpy as np; sys.argv=["x"]; sys.path.insert(0, ".")  # run from NNs/transformer_rewrite (see run_*.sh); import deept_gauge as g, torch.nn as nn
from auto_LiRPA import BoundedModule
torch.set_num_threads(3); m, tok, net = g.build("sst_bert_small_3", "cpu"); data = g.load_sst("test"); S = g.short_instances(net, m, tok, data, 8, 3, seed=3)
j, ex, e, toks = S[0]; n = e.shape[1]; i = g.positions(toks)[0]
lp = BoundedModule(net, torch.empty(1, n, 128), bound_opts={"softmax": "lse", "sparse_intermediate_bounds": False})
r = g.certified_radius(lp, e, i, ex["label"], "cpu", hi=0.1, iters=8); lb, ub = g.crown_lb(lp, e, i, 2 * r, ex["label"], "cpu", with_ub=True)
x = net.ln0(e); probs = []
with torch.no_grad():
    for l in net.layers:
        at = l.attention.self; q = at.query(x).view(1, n, net.H, net.dh).transpose(1, 2); k = at.key(x).view(1, n, net.H, net.dh).transpose(1, 2); v = at.value(x).view(1, n, net.H, net.dh).transpose(1, 2)
        p = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(net.dh), dim=-1); probs.append(p); c = torch.matmul(p, v).transpose(1, 2).reshape(1, n, -1)
        h = l.attention.output.LayerNorm(l.attention.output.dense(c) + x); x = l.output.LayerNorm(l.output.dense(torch.relu(l.intermediate.dense(h))) + h)
net.frozen_probs = nn.ParameterList([nn.Parameter(p_, requires_grad=False) for p_ in probs]); lpf = BoundedModule(net, torch.empty(1, n, 128), bound_opts={"softmax": "lse", "sparse_intermediate_bounds": False})
lbf, ubf = g.crown_lb(lpf, e, i, 2 * r, ex["label"], "cpu", with_ub=True); net.frozen_probs = None
c1 = collections.Counter(type(v).__name__ for v in lp._modules.values()); c2 = collections.Counter(type(v).__name__ for v in lpf._modules.values())
print(f"len {n} pos {i} eps {2*r:.4f}: normal lb {lb:+.6f} ub {ub:+.6f} width {ub-lb:.6f} | frozen lb {lbf:+.6f} ub {ubf:+.6f} width {ubf-lbf:.6f}")
print("normal graph:", {k: c1[k] for k in ("BoundSoftmax", "BoundMatMul", "BoundRelu", "BoundTanh", "BoundExp")}); print("frozen graph:", {k: c2[k] for k in ("BoundSoftmax", "BoundMatMul", "BoundRelu", "BoundTanh", "BoundExp")})
# forward agreement at a random point in the box
with torch.no_grad(): xl, xu = g.box(e, i, 2 * r); xs = xl + torch.rand_like(xl) * (xu - xl); out1 = net(xs); net.frozen_probs = nn.ParameterList([nn.Parameter(p_, requires_grad=False) for p_ in probs]); out2 = net(xs); net.frozen_probs = None
print(f"forward at a random box point: normal {out1.numpy().round(4)} frozen {out2.numpy().round(4)} (should differ slightly)")
