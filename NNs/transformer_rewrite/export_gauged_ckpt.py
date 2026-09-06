"""Export a DeepT checkpoint with a learned attention gauge folded into its weights, in the directory layout that
Shi et al. 2020-style verifiers (DeepT, PBVerification) load with `--dir`:

    <out>/checkpoint            (contains the epoch number N)
    <out>/ckpt-N/config.json, vocab.txt, pytorch_model.bin

`--gauge none` writes the stock weights through the same code path (identity gauge), so the stock/gauged pair differs
only in the folded gauge.  The gauge is exact (per-head G G^-1 insertions); the fp64 gate is reported on 16 random points.

Usage (from NNs/transformer_rewrite, venv python):
    python export_gauged_ckpt.py --name sst_bert_small_6 --gauge gauges/deept_small6_seed0.pt --out ../../deept_benchmarks/gauged_ckpts/sst_bert_small_6_gauged
    python export_gauged_ckpt.py --name sst_bert_small_6 --gauge none --out ../../deept_benchmarks/gauged_ckpts/sst_bert_small_6_stock
"""
import argparse, os, shutil, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deept_gauge import DT, load_deept, DeepTNet, stock_tensors, effective, load_eff, eye_gauge, load_gauge

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--gauge", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.manual_seed(0)
m, tok = load_deept(a.name); net = DeepTNet(m); L, H, dh = len(net.layers), net.H, net.dh
src = os.path.join(DT, a.name); n_ck = int(open(os.path.join(src, "checkpoint")).readline()); ck = os.path.join(src, "ckpt-%d" % n_ck)
st = stock_tensors(net)
# reference outputs on random embeddings before the rewrite
x = torch.randn(16, 9, m.config.hidden_size)
with torch.no_grad(): y0 = net(x).double()
if a.gauge.lower() == "none": Gq, Ga = eye_gauge(L, H, dh, torch.float64)
else: Gq, Ga = load_gauge(a.gauge, L, H, dh)
st64 = [[t.double() for t in w] for w in st]
load_eff(net, [[t.float() for t in w] for w in effective(st64, Gq.double(), Ga.double(), H, dh)])
with torch.no_grad(): y1 = net(x).double()
print(f"# {a.name} gauge={a.gauge}: max |logit change| on 16 random inputs = {(y1 - y0).abs().max().item():.3e} (fp32 weights)")
out_ck = os.path.join(a.out, "ckpt-%d" % n_ck); os.makedirs(out_ck, exist_ok=True)
for f in ("config.json", "vocab.txt"): shutil.copy(os.path.join(ck, f), out_ck)
for extra in ("vocab_base.txt",):
    if os.path.exists(os.path.join(src, extra)): shutil.copy(os.path.join(src, extra), a.out)
torch.save(m.state_dict(), os.path.join(out_ck, "pytorch_model.bin"))
open(os.path.join(a.out, "checkpoint"), "w").write("%d\n" % n_ck)
print(f"# wrote {out_ck}/pytorch_model.bin ({len(m.state_dict())} tensors)")
