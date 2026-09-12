import sys, time, torch, argparse, random; sys.path.insert(0, "."); torch.set_num_threads(4)
from gauge_formula import *
a = argparse.Namespace(name="yelp_bert_small_3", data="random", seed=0); dev = "cpu"; t0 = time.time()
m, tok, net = build(a.name, dev); L, H, dh, hid = len(net.layers), net.H, net.dh, net.hid; print("built", time.time() - t0)
st = [[t.to(dev) for t in w] for w in stock_tensors(net)]; st64 = [[t.double() for t in w] for w in st]
R = load_data(a, "dev"); Sr = short_instances(net, m, tok, R, 8, 2, seed=0); rng = random.Random(0)
boxes = [(e, i, ex["label"], e.shape[1], 0.01) for j, ex, e, toks in Sr for i in pos_sets(toks, 1, rng, 1)]; print("boxes", [(b[3], b[1]) for b in boxes])
Ms = box_shapes(net, boxes, dev); print("M", [tuple(M.shape) for M in Ms], time.time() - t0)
sens = sens_shapes(net, boxes, dev); print("sens", [[tuple(t.shape) for t in w] for w in sens[0]], time.time() - t0)
for l in range(L):
    al, be, ga = sens[l][0]; print(f" layer {l} box0 head0: alpha {al[0].numpy().round(3)} beta {be[0].numpy().round(3)} gamma {ga[0].numpy().round(4)}")
No = out_shapes(net, boxes, dev); Nf, Nn = weight_shapes(net, st64, dev); Na = [combine_N([Nf[l], Nn[l], No[l]]) for l in range(L)]
G0 = candidate_svd(st64, Ms, H, dh, Na); G1 = candidate_svd_w(st64, Ms, H, dh, Na, sens, hid); G2 = candidate_svd_w(st64, Ms, H, dh, Na, None, hid, rho=[torch.ones(M.shape[1], dtype=torch.float64) for M in Ms])
print("shapes", tuple(G1[0].shape), tuple(G1[1].shape), "rho=1 reproduces svd_jacN_all:", all(torch.allclose(x, y) for x, y in zip(G0, G2)))
print("sens vs plain rel diff per layer QK", [f"{((G1[0][l]-G0[0][l]).norm()/G0[0][l].norm()).item():.2f}" for l in range(L)], "AV", [f"{((G1[1][l]-G0[1][l]).norm()/G0[1][l].norm()).item():.2f}" for l in range(L)])
print("cond", max(torch.linalg.cond(G1[0].reshape(-1, dh, dh)).max().item(), torch.linalg.cond(G1[1].reshape(-1, dh, dh)).max().item()), "total", time.time() - t0)
