"""Bucket the paired-protocol results by token count T and by position; manual vs learned gauge."""
import json, sys, numpy as np
from collections import defaultdict
R = "results/"
def load(path, tag):
    d = json.load(open(R + path)); return d, np.array(d["stock_rad"]), np.array(d[tag + "_rad"]), {e: (np.array(F["stock"]), np.array(F[tag])) for e, F in d["fixed"].items()}
def bucket_report(name, inst, s, m, l, fs, fm, fl, key, label):
    groups = defaultdict(list)
    for k, x in enumerate(inst): groups[key(x)].append(k)
    eps_max = max(fs, key=float)
    print(f"\n[{name}] bucket by {label}: n | stock mean rad | manual gain | learned gain | share | manual vs learned larger/smaller/equal | eps {eps_max}: mean lb manual/learned, verified stock/manual/learned, manual-vs-learned tighter/looser")
    rows = []
    for g in sorted(groups):
        I = np.array(groups[g]); ss, mm, ll = s[I], m[I], l[I]
        gm, gl = mm.mean() / ss.mean() - 1, ll.mean() / ss.mean() - 1
        sh = gm / gl if abs(gl) > 1e-12 else float("nan")
        big, sml, eq = int((mm > ll + 1e-12).sum()), int((mm < ll - 1e-12).sum()), int((abs(mm - ll) <= 1e-12).sum())
        S0, M0, L0 = fs[eps_max][I], fm[eps_max][I], fl[eps_max][I]
        dv = L0 - M0
        print(f"  {label}={g:>3}: n={len(I):3d} | {ss.mean():.5f} | {gm:+6.1%} | {gl:+6.1%} | {sh:5.2f} | {big:3d}/{sml:3d}/{eq:3d} | lb {np.nanmean(M0-S0):+.3f}/{np.nanmean(L0-S0):+.3f}, ver {int((S0>0).sum()):3d}/{int((M0>0).sum()):3d}/{int((L0>0).sum()):3d}, {int((dv>1e-9).sum()):3d}/{int((dv<-1e-9).sum()):3d}")
        rows.append((g, len(I), ss.mean(), gm, gl, sh, big, sml, eq))
    return rows
def per_eps(name, fs, fm, fl):
    print(f"[{name}] per eps: verified stock / manual / learned; mean lb gain manual / learned; share of lb gain; learned tighter / manual tighter")
    for e in sorted(fs, key=float):
        S0, M0, L0 = fs[e], fm[e], fl[e]; dv = L0 - M0
        print(f"  eps {e}: {int((S0>0).sum()):3d} / {int((M0>0).sum()):3d} / {int((L0>0).sum()):3d}; {np.nanmean(M0-S0):+.4f} / {np.nanmean(L0-S0):+.4f}; {np.nanmean(M0-S0)/np.nanmean(L0-S0):5.2f}; {int((dv>1e-9).sum()):3d} / {int((dv<-1e-9).sum()):3d}")
def run(name, path_m, tag_m, path_l, tag_l):
    dm, s, m, Fm = load(path_m, tag_m); dl, s2, l, Fl = load(path_l, tag_l)
    assert [tuple(x) for x in dm["inst"]] == [tuple(x) for x in dl["inst"]], "instances differ"; assert np.allclose(s, s2)
    inst = dm["inst"]; fs = {e: v[0] for e, v in Fm.items()}; fm = {e: v[1] for e, v in Fm.items()}; fl = {e: Fl[e][1] for e in Fm}
    n = len(s); gm, gl = m.mean() / s.mean() - 1, l.mean() / s.mean() - 1
    print(f"\n===== {name}: {n} instances; manual {tag_m}@{path_m}, learned {tag_l}@{path_l}; overall manual {gm:+.1%} learned {gl:+.1%} share {gm/gl:.2f}; manual vs learned larger {int((m>l+1e-12).sum())} / smaller {int((m<l-1e-12).sum())} / equal {int((abs(m-l)<=1e-12).sum())}")
    Ts = np.array([x[2] for x in inst]); print(f"  T distribution: {dict(zip(*np.unique(Ts, return_counts=True)))}; position distribution: {dict(zip(*np.unique([x[1] for x in inst], return_counts=True)))}")
    per_eps(name, fs, fm, fl)
    bucket_report(name, inst, s, m, l, fs, fm, fl, lambda x: x[2], "T")
    bucket_report(name, inst, s, m, l, fs, fm, fl, lambda x: "<=8" if x[2] <= 8 else ">8", "Tgrp")
    bucket_report(name, inst, s, m, l, fs, fm, fl, lambda x: x[1], "pos")
    bucket_report(name, inst, s, m, l, fs, fm, fl, lambda x: "first" if x[1] == 1 else ("last" if x[1] == x[2] - 2 else "mid"), "posgrp")
    bucket_report(name, inst, s, m, l, fs, fm, fl, lambda x: x[3], "label")
    # per-instance relative difference: where are the biggest learned wins / manual wins?
    rel = (l - m) / s; order = np.argsort(-rel)
    print(f"  per-instance (learned-manual)/stock: mean {rel.mean():+.4f}, median {np.median(rel):+.4f}, p10 {np.percentile(rel,10):+.4f}, p90 {np.percentile(rel,90):+.4f}")
    print("  top-8 learned wins (inst, stock, manual, learned):", [(inst[k], round(float(s[k]),5), round(float(m[k]),5), round(float(l[k]),5)) for k in order[:8]])
    print("  top-8 manual wins:", [(inst[k], round(float(s[k]),5), round(float(m[k]),5), round(float(l[k]),5)) for k in order[-8:][::-1]])
    # is the gap in the stock-radius-large or small instances?
    q = np.quantile(s, [0.25, 0.5, 0.75]); print("  by stock-radius quartile:")
    bucket_report(name, inst, s, m, l, fs, fm, fl, lambda x: 0, "all")
    key = lambda k: int(np.searchsorted(q, s[k], side="right"))
    groups = defaultdict(list)
    for k in range(n): groups[key(k)].append(k)
    for g in sorted(groups):
        I = np.array(groups[g]); print(f"    quartile {g}: n={len(I)} stock {s[I].mean():.5f} manual {m[I].mean()/s[I].mean()-1:+.1%} learned {l[I].mean()/s[I].mean()-1:+.1%} share {(m[I].mean()/s[I].mean()-1)/(l[I].mean()/s[I].mean()-1):.2f} m>l {int((m[I]>l[I]+1e-12).sum())} m<l {int((m[I]<l[I]-1e-12).sum())}")
run("big_3 unified (l1N_qk+av0) vs learned", "deept_formula_big3_r3b_eval_short_seed0.json", "gauged", "deept_formula_big3_eval_short_seed0.json", "gauged3")
run("big_3 closed form (svd_jacN_all) vs learned", "deept_formula_big3_eval_short_seed0.json", "gauged", "deept_formula_big3_eval_short_seed0.json", "gauged3")
run("big_3 QK-only (l1N_qk) vs learned", "deept_formula_big3_r3b_eval_short_seed0.json", "gauged2", "deept_formula_big3_eval_short_seed0.json", "gauged3")
run("yelp3 l1N vs learned", "deept_formula_yelp3_r3_eval_short_seed0.json", "gauged", "deept_formula_yelp3_r3_eval_short_seed0.json", "gauged3")
run("yelp3 closed form (svd_jacN_all) vs learned", "deept_formula_yelp3_eval_short_seed0.json", "gauged", "deept_formula_yelp3_eval_short_seed0.json", "gauged3")
run("small_6 closed form (svd_jacN_all 24-box) vs learned", "deept_formula_small6_eval_short_seed0.json", "gauged", "deept_small6_eval_short_seed0.json", "gauged")
