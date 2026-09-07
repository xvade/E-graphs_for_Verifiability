"""Sanity check before running the gauge pipeline on a Shi-style model that was NOT trained with DeepT's module classes
(e.g. Huang et al.'s model_sst_3, whose attention has no Q/K/V biases): the harness forward must reproduce the loaded
model's own logits, and a CROWN call must work.  Exit 1 on mismatch.   python pbv_harness_check.py <name> [--no_crown]"""
import copy, sys, time, torch
from deept_gauge import build, load_sst, embed, make_lirpas, crown_lb, positions
name = sys.argv[1]; dev = "cuda" if torch.cuda.is_available() else "cpu"
m, tok, net = build(name, dev)            # net (on dev) shares the encoder modules with m; m's embeddings stay on CPU (embed() needs that)
m_cpu = copy.deepcopy(m).cpu().eval()     # an all-CPU copy of the raw model for the reference logits
print(f"# {name}: hidden {net.hid}, heads {net.H}, d_head {net.dh}, layers {len(net.layers)}; max |q/k/v bias| = {max(float(l.attention.self.query.bias.abs().max()) for l in net.layers):.2e}", flush=True)
data = load_sst("test"); worst = 0.0
for ex in data[:8]:
    e, toks = embed(m, tok, ex); ids = torch.tensor([tok.convert_tokens_to_ids(toks)])
    with torch.no_grad():
        ref = m_cpu(ids, torch.zeros_like(ids), torch.ones_like(ids)); ref = ref[0] if isinstance(ref, (tuple, list)) else ref; out = net(e.to(dev))
    worst = max(worst, (ref.flatten().cpu() - out.flatten().cpu()).abs().max().item())
print(f"# harness vs model logits on 8 test sentences: max |diff| = {worst:.2e}", flush=True)
if "--no_crown" not in sys.argv:
    e, toks = embed(m, tok, data[0]); lp = make_lirpas(net, [e.shape[1]], dev)[e.shape[1]]; t0 = time.time(); i = positions(toks)[0]
    print(f"# CROWN lb at eps 0.01 / 0.03 on sentence 0 pos {i} ({e.shape[1]} tokens): {crown_lb(lp, e, i, 0.01, data[0]['label'], dev):+.4f} / {crown_lb(lp, e, i, 0.03, data[0]['label'], dev):+.4f}  [{time.time()-t0:.1f}s]", flush=True)
sys.exit(0 if worst < 1e-4 else 1)
