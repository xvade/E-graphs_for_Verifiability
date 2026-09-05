#!/usr/bin/env python
"""Parse official abcrown logs (vit.yaml pipeline) into per-instance records and pair two of them.

Per instance we extract: (a) the initial CROWN bounds (complex-mode VANILLA CROWN, deterministic), (b) the initial
alpha-CROWN bounds (full CROWN), (c) #specs verified by alpha-CROWN, (d) the final Result (safe / safe-incomplete /
unknown / timeout).  (a) and (b) are contention-free; the BaB verdict is time-capped and reported with that caveat.

  python vit_official_parse.py A.log [B.log]
"""
import sys, re, json, numpy as np
def parse(path):
    recs, cur = {}, None
    for line in open(path, errors="replace"):
        m = re.search(r"Using vnnlib vnnlib/(\S+?)_(\d+)\.vnnlib", line)
        if m: cur = int(m.group(2)); recs[cur] = dict(id=cur); continue
        if cur is None: continue
        r = recs[cur]
        m = re.match(r"initial CROWN bounds \(first 10 items\): \[(.*)\]", line)
        if m and "crown" not in r: r["crown"] = [float(v) for v in m.group(1).split(",") if v.strip()]
        m = re.match(r"initial alpha-crown bounds \(first 10 items\): \[(.*)\]", line)
        if m and "alpha" not in r: r["alpha"] = [float(v) for v in m.group(1).split(",") if v.strip()]
        if line.startswith("Using alpha-CROWN to initialize bounds"): r["_await_alpha_t"] = True
        m = re.match(r"alpha/beta optimization time: ([\d.]+)", line)
        if m and r.pop("_await_alpha_t", False): r["alpha_time"] = float(m.group(1))   # the initial alpha-CROWN pass (cap: max_time 0.3 x 100 s)
        m = re.match(r"(\d+) / (\d+) OR specs are verified", line)
        if m and "n_alpha_ver" not in r: r["n_alpha_ver"] = int(m.group(1)); r["n_specs"] = int(m.group(2))
        m = re.match(r"Global lower bound: (\S+)", line)
        if m and "alpha" in r and "alpha_glb" not in r: r["alpha_glb"] = float(m.group(1))   # min alpha-CROWN bound over specs not already verified by CROWN
        m = re.match(r"max lb (\S+) min lb (\S+)", line)
        if m: r["bab_last_min_lb"] = float(m.group(2))   # last BaB batch's min domain lb (proxy for the proof's margin)
        m = re.match(r"Result: (\S+) in ([\d.]+) seconds", line)
        if m:
            r["result"] = m.group(1); r["time"] = float(m.group(2))
            # 'safe-incomplete' = all specs verified by the incomplete (alpha-CROWN) bound; no 'X / 9' line is printed then
            if r["result"] == "safe-incomplete": r["n_alpha_ver"] = r["n_specs"] = 9
    return recs
def summarize(recs, name):
    done = [r for r in recs.values() if "result" in r]
    res = {}
    for r in done: res[r["result"]] = res.get(r["result"], 0) + 1
    cmin = [min(r["crown"]) for r in done if "crown" in r]
    nver_crown = sum(1 for r in done if "crown" in r and min(r["crown"]) > 0)
    # alpha: printed list only covers the specs still unverified after CROWN; #verified specs is the reliable stat
    nver_alpha = sum(1 for r in done if r.get("n_alpha_ver", 0) == r.get("n_specs", 9))
    amin = [min(r["alpha"]) if r.get("alpha") else np.inf for r in done]
    at = np.array([r["alpha_time"] for r in done if "alpha_time" in r])
    print(f"# {name}: {len(done)} instances finished")
    if len(at): print(f"  initial alpha-CROWN optimization time: n={len(at)} mean {at.mean():.1f}s max {at.max():.1f}s; >=29s (time-capped, 30 s cap): {int((at >= 29).sum())}; 'all verified early' skipped (no alpha pass): {len(done) - len(at)}")
    print(f"  final Result counts: {res}")
    print(f"  initial CROWN (complex, vanilla): mean min-lb {np.mean(cmin):+.4f}, median {np.median(cmin):+.4f}, all-9-verified {nver_crown}/{len(cmin)}")
    print(f"  alpha-CROWN: all-9-verified {nver_alpha}/{len(done)}; mean #specs verified {np.mean([r.get('n_alpha_ver', 0) for r in done]):.2f}/9;"
          f" mean min-lb over unverified specs {np.mean([a for a in amin if np.isfinite(a)]):+.4f}")
    return done
def pair(A, B, na, nb):
    ids = sorted(set(A) & set(B)); ids = [i for i in ids if "result" in A[i] and "result" in B[i]]
    dc = np.array([min(B[i]["crown"]) - min(A[i]["crown"]) for i in ids if "crown" in A[i] and "crown" in B[i]])
    print(f"# PAIRED ({len(ids)} common finished instances) {na} -> {nb}")
    print(f"  initial CROWN min-lb: tighter on {int((dc > 0).sum())}/{len(dc)}, looser {int((dc < 0).sum())}, delta mean {dc.mean():+.4f} median {np.median(dc):+.4f} worst {dc.min():+.4f}")
    da = np.array([B[i].get("n_alpha_ver", 0) - A[i].get("n_alpha_ver", 0) for i in ids])
    print(f"  alpha-CROWN #specs verified: more on {int((da > 0).sum())}, fewer on {int((da < 0).sum())}, same {int((da == 0).sum())}; net {int(da.sum()):+d} specs")
    rank = {"safe": 2, "safe-incomplete": 2, "unknown": 0, "timeout": 0, "unsafe-pgd": -1}
    flips_up = [i for i in ids if rank.get(A[i]["result"], 0) < rank.get(B[i]["result"], 0)]
    flips_dn = [i for i in ids if rank.get(A[i]["result"], 0) > rank.get(B[i]["result"], 0)]
    print(f"  final verdict flips (time-capped BaB): unknown->verified {len(flips_up)} {flips_up[:20]}; verified->unknown {len(flips_dn)} {flips_dn[:20]}")
    # margins of B's newly verified instances (fp32-storage caveat check): safe-incomplete -> min(alpha GLB, positive CROWN bounds);
    # safe (BaB) -> last BaB batch's min domain lb (proxy).  Storage discrepancy of the rewritten weights is ~3e-6.
    newly = [i for i in ids if rank.get(A[i]["result"], 0) < 2 <= rank.get(B[i]["result"], 0)]
    new_inc = [i for i in ids if A[i]["result"] != "safe-incomplete" and B[i]["result"] == "safe-incomplete"]
    def margin(r):
        if r["result"] == "safe-incomplete":
            c = min(r["crown"]) if "crown" in r else np.inf
            return min(r.get("alpha_glb", np.inf), c) if c > 0 else r.get("alpha_glb", np.nan)
        return r.get("bab_last_min_lb", np.nan)
    for lab, S in (("unknown->verified", newly), ("newly safe-incomplete (alpha-CROWN alone)", new_inc)):
        ms = [(i, B[i]["result"], margin(B[i])) for i in S]
        if ms: print(f"  margins of B's {lab} ({len(ms)}): min {min(m for _, _, m in ms):.2e}; " + ", ".join(f"{i}:{res[:4]}:{m:.1e}" for i, res, m in ms))
    ta = np.mean([A[i]["time"] for i in ids]); tb = np.mean([B[i]["time"] for i in ids])
    print(f"  mean time/instance {ta:.1f}s -> {tb:.1f}s")
if __name__ == "__main__":
    A = parse(sys.argv[1]); summarize(A, sys.argv[1].split("/")[-1])
    if len(sys.argv) > 2:
        B = parse(sys.argv[2]); summarize(B, sys.argv[2].split("/")[-1]); pair(A, B, sys.argv[1].split("/")[-1], sys.argv[2].split("/")[-1])
