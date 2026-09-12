"""sentence-id overlap between the round-7 screens' held-out boxes (formula_<m>_sgn<y>.json) and the paired protocol's instances"""
import sys, json, random, argparse, numpy as np, torch
sys.path.insert(0, "."); from deept_gauge import build, load_data, short_instances, pos_sets
for m, paired in (("small6", "results/deept_small6_eval_short_seed0.json"), ("big3", "results/deept_big3_eval_short_seed0.json")):
    P = json.load(open(paired)); pj = {x[0] for x in P["inst"]}; pn = {}
    for x in P["inst"]: pn[x[0]] = pn.get(x[0], 0) + 1
    for y in (0, 1):
        jf = f"results/formula_{m}_sgn{y}.json"; d = json.load(open(jf)); a = argparse.Namespace(**d["args"])
        torch.manual_seed(a.seed); random.seed(a.seed); net_m, tok, net = build(a.name, "cpu")
        gp = [p for p in a.gauges.split(",") if p]; ga = torch.load(gp[0]).get("args", {})
        la = argparse.Namespace(name=a.name, data=ga.get("data", "auto"), seed=ga.get("seed", 0)); Dv = load_data(la, ga.get("split", "dev"))
        Ss = short_instances(net, net_m, tok, Dv, ga.get("max_len", 8), ga.get("n_sent", 60), seed=la.seed); used = {j for j, ex, e, toks in Ss}
        rngd = random.Random(a.seed + 7); dsplit = a.dev_split or ga.get("split", "dev"); same = dsplit == ga.get("split", "dev")
        rest = [s_ for s_ in short_instances(net, net_m, tok, Dv if same else load_data(la, dsplit), ga.get("max_len", 8), None) if not same or s_[0] not in used]; rngd.shuffle(rest)
        Sd = rest[:a.n_dev]; sj = {j for j, ex, e, toks in Sd}; ov = sj & pj
        print(f"{m} screen label {y} ({jf}): {len(sj)} {dsplit} sentences; overlap with the paired protocol's {len(pj)} sentences: {len(ov)} sentences = {sum(pn[j] for j in ov)} of {len(P['inst'])} paired instances; ids {sorted(ov)}", flush=True)
