"""Per-class gauges on the PAIRED protocol: combine two per-class paired evals (label-0 gauge, label-1 gauge; one file with tags
gauged / gauged2 from deept_formula_eval.sh) by choosing, for every instance, the gauge of its label, and compare with the single
learned gauge's paired file on the same instances (ratio of means vs stock, larger / smaller counts, head to head, per label).
  python perclass_paired.py <single_paired.json> <perclass_paired.json> [<perclass_paired_2.json> ...]"""
import sys, json, numpy as np
single = json.load(open(sys.argv[1])); inst = single["inst"]; st = np.array(single["stock_rad"], float); lr = np.array(single["gauged_rad"], float)
lab = np.array([x[3] for x in inst]); print(f"# {len(inst)} instances (label 0: {(lab==0).sum()}, label 1: {(lab==1).sum()}); stock mean radius {st.mean():.5f}; single learned {lr.mean():.5f} ({lr.mean()/st.mean()-1:+.1%}; larger {(lr>st+1e-9).sum()} / smaller {(lr<st-1e-9).sum()})")
def report(name, r):
    gain = r.mean() / st.mean() - 1; ref = lr.mean() / st.mean() - 1
    line = f"  {name:34s} mean {r.mean():.5f} vs stock {gain:+.1%} (larger {(r>st+1e-9).sum()} / smaller {(r<st-1e-9).sum()}) share {gain/ref:.2f} | vs single learned: larger {(r>lr+1e-9).sum()} / smaller {(r<lr-1e-9).sum()}"
    for y in (0, 1):
        m = lab == y; line += f" | label {y}: {r[m].mean()/st[m].mean()-1:+.1%} (single {lr[m].mean()/st[m].mean()-1:+.1%})"
    print(line)
for f in sys.argv[2:]:
    d = json.load(open(f)); assert d["inst"] == inst, "instance sets differ"
    g0 = np.array(d["gauged_rad"], float); g1 = np.array(d["gauged2_rad"], float)
    comb = np.where(lab == 0, g0, g1); print(f"# {f}"); report("label-0 gauge on all", g0); report("label-1 gauge on all", g1); report("CHOSEN BY LABEL", comb)
    if "fixed" in d:
        for eps, v in d["fixed"].items():
            s_ = np.array(v["stock"], float); c_ = np.where(lab == 0, np.array(v["gauged"], float), np.array(v["gauged2"], float)); l_ = np.array(single["fixed"][eps]["gauged"], float) if eps in single.get("fixed", {}) else None
            print(f"    eps {eps}: verified stock {(s_>0).sum()} -> chosen-by-label {(c_>0).sum()}" + (f" (single learned {(l_>0).sum()})" if l_ is not None else ""))
