"""Reconstruct the labels of the held-out screen boxes (same draws as gauge_formula.py main(): rngd = Random(seed+7), test split,
max_len from the learner's args, n_dev sentences x dev_pos positions) and split every gauge's dev_radius by label."""
import os, sys, json, random, argparse, numpy as np, torch
sys.path.insert(0, "."); from deept_gauge import build, load_data, short_instances, pos_sets
jf = sys.argv[1]; d = json.load(open(jf)); a = argparse.Namespace(**d["args"]); rows = d["rows"]
torch.manual_seed(a.seed); random.seed(a.seed); m, tok, net = build(a.name, "cpu")
gp = [p for p in a.gauges.split(",") if p]; ga = torch.load(gp[0] if os.path.isabs(gp[0]) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", gp[0]) if "diagnostics" in os.path.abspath(__file__) else os.path.join(os.path.dirname(os.path.abspath(__file__)), gp[0])).get("args", {})
la = argparse.Namespace(name=a.name, data=ga.get("data", "auto"), seed=ga.get("seed", 0)); Dv = load_data(la, ga.get("split", "dev"))
Ss = short_instances(net, m, tok, Dv, ga.get("max_len", 8), ga.get("n_sent", 60), seed=la.seed); used = {j for j, ex, e, toks in Ss}
rngd = random.Random(a.seed + 7); dsplit = a.dev_split or ga.get("split", "dev"); same = dsplit == ga.get("split", "dev")
rest = [s_ for s_ in short_instances(net, m, tok, Dv if same else load_data(la, dsplit), ga.get("max_len", 8), None) if not same or s_[0] not in used]; rngd.shuffle(rest)
Sd = rest[:a.n_dev]; boxes = [(j, i, ex["label"], e.shape[1]) for j, ex, e, toks in Sd for i in pos_sets(toks, 1, rngd, a.dev_pos)]
names = [k for k in rows if "dev_radius" in rows[k]]; n = len(rows[names[0]]["dev_radius"]); assert n == len(boxes), (n, len(boxes))
lab = np.array([b[2] for b in boxes]); print(f"# {jf}: {len(Sd)} {dsplit} sentences -> {n} boxes; labels 0: {(lab==0).sum()}  1: {(lab==1).sum()}  (sentence ids {[b[0] for b in boxes[:6]]}...)")
R = {k: np.array(rows[k]["dev_radius"]) for k in names}; st = R["identity"]; ref = sys.argv[2] if len(sys.argv) > 2 else names[1]
for sel, nm in ((lab >= 0, "all"), (lab == 0, "label 0"), (lab == 1, "label 1")):
    s = st[sel]; print(f"\n== {nm}: n={sel.sum()}  mean stock radius {s.mean():.5f}"); gl = R[ref][sel].mean() / s.mean() - 1
    for k in names:
        r = R[k][sel]; g = r.mean() / s.mean() - 1; L = R[ref][sel]
        print(f"  {k:22s} mean {r.mean():.5f}  gain {100*g:+5.1f}%  share {g/gl if gl else float('nan'):5.2f}  vs stock {int((r>s+1e-12).sum()):3d}/{int((r<s-1e-12).sum()):3d}  vs {ref} {int((r>L+1e-12).sum()):3d}/{int((r<L-1e-12).sum()):3d}")
