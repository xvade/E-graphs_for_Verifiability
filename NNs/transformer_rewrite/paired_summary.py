"""Summarise a paired eval JSON (deept_gauge.py eval --save_json): radius gain (ratio of means), larger/smaller counts vs stock,
share of the learned gauge's gain, fixed-eps verified counts and flips, head-to-head vs the learned gauge.
    python paired_summary.py results/deept_formula_big3_eval_short_seed0.json [learned_tag=gauged3]"""
import json, sys, numpy as np
d = json.load(open(sys.argv[1])); L = sys.argv[2] if len(sys.argv) > 2 else "gauged3"
tags = [k[:-4] for k in d if k.endswith("_rad")]; R = {t: np.array(d[f"{t}_rad"]) for t in tags}; n = len(R["stock"])
if len(sys.argv) > 4:   # reference JSON holding the learned set (same instances, e.g. an earlier paired eval): joined as tag "ref"
    ref = json.load(open(sys.argv[3])); assert [tuple(x) for x in ref["inst"]] == [tuple(x) for x in d["inst"]], "instances differ"
    R["ref"] = np.array(ref[f"{sys.argv[4]}_rad"]); tags.append("ref"); L = "ref"
    for eps in d["fixed"]: d["fixed"][eps]["ref"] = ref["fixed"][eps][sys.argv[4]]
print(f"{sys.argv[1]}: {n} instances; gauges {d.get('gauges')}; learned tag = {L}")
g_l = R[L].mean() / R["stock"].mean() - 1 if L in R else float("nan")
for t in tags:
    r, s = R[t], R["stock"]; g = r.mean() / s.mean() - 1
    line = f"  {t:8s} mean radius {r.mean():.5f}  vs stock {g:+6.1%}  larger {int((r > s + 1e-12).sum()):3d} / smaller {int((r < s - 1e-12).sum()):3d} / equal {int((abs(r - s) <= 1e-12).sum()):3d}"
    if t not in ("stock", L) and L in R: line += f"  share of learned gain {g / g_l:5.2f}  vs learned: larger {int((r > R[L] + 1e-12).sum()):3d} / smaller {int((r < R[L] - 1e-12).sum()):3d}"
    print(line)
for eps, F in d["fixed"].items():
    st = np.array(F["stock"]); out = [f"  eps {eps}: verified stock {int((st > 0).sum()):3d}"]
    for t in tags:
        if t == "stock": continue
        v = np.array(F[t]); out.append(f"{t} {int((v > 0).sum()):3d} (flips +{int(((v > 0) & (st <= 0)).sum())} / -{int(((v <= 0) & (st > 0)).sum())}, mean lb {np.nanmean(v - st):+.3f})")
    print("; ".join(out))
if L in R:
    for t in tags:
        if t in ("stock", L): continue
        for eps, F in d["fixed"].items(): dv = np.array(F[L]) - np.array(F[t]); print(f"  learned - {t} at eps {eps}: lb tighter on {int((dv > 1e-9).sum())}/{n} (mean {dv.mean():+.3f})")
