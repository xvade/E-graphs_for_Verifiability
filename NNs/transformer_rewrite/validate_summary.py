"""Summarise a gauge_formula.py validate JSON: held-in screen (mean CROWN lb at the stock radius on random-token boxes / the learner's
boxes, share of the learned gain) and, when present, the held-out certified-radius screen (ratio of means vs stock, larger/smaller
counts, share of the learned gain).   python validate_summary.py results/formula_yelp3_hyb.json [learned_name]"""
import json, sys, numpy as np
d = json.load(open(sys.argv[1])); rows = d["rows"]; names = list(rows)
L = sys.argv[2] if len(sys.argv) > 2 else next((k for k in names if k != "identity" and ":" not in k), None)
i0 = rows["identity"]; l0 = rows.get(L, i0)
print(f"{sys.argv[1]}: random boxes {d.get('boxes_random')}, learner boxes {d.get('boxes_sst')}, dev boxes {d.get('boxes_dev', 0)}, text-probe boxes {d.get('boxes_probe', 0)}; learned = {L}")
print(f"  {'gauge':28s} | held-in lb random (share)  learner (share) | held-out radius: gain (share)  larger/smaller vs stock  vs learned | cond")
r0 = np.array(i0.get("dev_radius", [])); rl = np.array(l0.get("dev_radius", []))
for k in names:
    r = rows[k]; sr = (r["random_lb"] - i0["random_lb"]) / (l0["random_lb"] - i0["random_lb"]); ss = (r["sst_lb"] - i0["sst_lb"]) / (l0["sst_lb"] - i0["sst_lb"])
    line = f"  {k:28s} | {r['random_lb']:+.3f} ({sr:5.2f})  {r['sst_lb']:+.3f} ({ss:5.2f}) |"
    if "dev_radius" in r and len(r0):
        rr = np.array(r["dev_radius"]); g = rr.mean() / r0.mean() - 1; gl = rl.mean() / r0.mean() - 1
        line += f" {g:+6.1%} ({g / gl if gl else float('nan'):5.2f})  {int((rr > r0 + 1e-9).sum()):3d}/{int((rr < r0 - 1e-9).sum()):3d}  {int((rr > rl + 1e-9).sum()):3d}/{int((rr < rl - 1e-9).sum()):3d} |"
    else: line += " " * 34 + "|"
    print(line + f" {r.get('cond', float('nan')):6.1f}")
