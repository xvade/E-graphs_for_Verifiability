"""Per-class gauges at the alpha-CROWN tier: combine per-label alpha eval JSONs (one gauge per file, all instances) by the instance's
label and compare with stock and the once-trained single gauge (and optionally a second per-class pair).
  python perclass_alpha.py <single_alpha.json> <eps> <name>=<lab0.json>,<lab1.json> [<name2>=<lab0.json>,<lab1.json> ...]"""
import sys, json, numpy as np
ref = json.load(open(sys.argv[1])); eps = sys.argv[2]; inst = ref["inst"]; lab = np.array([x[3] for x in inst])
s = np.array(ref["res"]["stock"][eps]["alpha"]); g = np.array(ref["res"]["gauged"][eps]["alpha"])
print(f"# {sys.argv[1]} eps {eps}: {len(inst)} instances (label 0 / 1: {(lab==0).sum()} / {(lab==1).sum()})")
def row(nm, v, base=None):
    line = f"  {nm:34s} verified {(v>0).sum():2d}/{len(v)} (nan {np.isnan(v).sum()}) mean lb {np.nanmean(v):+.4f} | per label {(v[lab==0]>0).sum()}/{(lab==0).sum()}, {(v[lab==1]>0).sum()}/{(lab==1).sum()}"
    if nm != "stock": line += f" | vs stock tighter {(v>s).sum()} looser {(v<s).sum()}"
    if nm not in ("stock", "single"): line += f" | vs single tighter {(v>g).sum()} looser {(v<g).sum()}, mean d {np.nanmean(v-g):+.4f}"
    if base is not None: line += f" | vs {base[0]} tighter {(v>base[1]).sum()} looser {(v<base[1]).sum()}, mean d {np.nanmean(v-base[1]):+.4f}"
    print(line)
row("stock", s); row("single", g); chosen = {}
for spec in sys.argv[3:]:
    nm, files = spec.split("="); f0, f1 = files.split(","); d0 = json.load(open(f0)); d1 = json.load(open(f1)); assert d0["inst"] == inst and d1["inst"] == inst
    m0 = np.array(d0["res"]["gauged"][eps]["alpha"]); m1 = np.array(d1["res"]["gauged"][eps]["alpha"]); ch = np.where(lab == 0, m0, m1); chosen[nm] = ch
    row(f"{nm} label-0 gauge on all", m0); row(f"{nm} label-1 gauge on all", m1)
    base = next(iter(chosen.items())) if len(chosen) > 1 else None; row(f"{nm} CHOSEN BY LABEL", ch, base)
