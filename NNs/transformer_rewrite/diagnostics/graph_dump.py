import sys, os, torch, re
sys.path.insert(0, "."); from deept_gauge import *
m, tok, net = build("sst_bert_small_6", "cuda"); D = load_sst("dev"); S = short_instances(net, m, tok, D, 6, 2, seed=0)
j, ex, e, toks = S[0]; lp = make_lirpas(net, [e.shape[1]], "cuda")[e.shape[1]]
lb = crown_lb(lp, e, 1, 0.01, ex["label"], "cuda"); print("lb", lb, "n", e.shape[1])
for node in lp.nodes():
    low = getattr(node, "lower", None); sh = tuple(low.shape) if isinstance(low, torch.Tensor) else None
    par = [getattr(i, "ori_name", "") for i in node.inputs if type(i).__name__ == "BoundParams"]
    print(f"{node.name:12s} {type(node).__name__:18s} in={[i.name for i in node.inputs]} out={node.output_name} bounds={sh} params={par}")
