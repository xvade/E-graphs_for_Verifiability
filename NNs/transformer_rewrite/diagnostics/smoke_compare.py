"""Golden-smoke comparison: python diagnostics/smoke_compare.py <reference.json> <new.json> [--lb_tol 1e-4] [--grid <step>]
--grid is the bisection step of the run = hi / 2**iters (deept_unfolded.py defaults: 0.1 / 2**4 = 0.00625 for the 7-instance smoke;
the full 294-instance protocol used a 1e-4 grid, the default here).
Compares every `<set>_rad` list and every fixed-eps lower bound shared by the two deept_gauge.py / deept_unfolded.py
result files. Pass = identical NaN pattern, identical verified counts per eps, |lb delta| <= lb_tol, radii equal except
one bisection grid step (--grid) on at most one instance per set. Exit code 1 on failure.
See ../PROVENANCE.md (sensitivity class B) for why these tolerances."""
import sys, json, math
ref, new = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
tol = float(sys.argv[sys.argv.index("--lb_tol") + 1]) if "--lb_tol" in sys.argv else 1e-4
grid = float(sys.argv[sys.argv.index("--grid") + 1]) if "--grid" in sys.argv else 1e-4
step_tol = 1.5 * grid   # one grid step, with slack for fp rounding of the grid itself
ok = True
if ref.get("inst") != new.get("inst"): print("instance lists differ"); ok = False
for k in sorted(set(ref) & set(new)):
    if not k.endswith("_rad"): continue
    r, n = ref[k], new[k]
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(r, n)) if abs(a - b) > step_tol]
    steps = sum(1 for a, b in zip(r, n) if 1e-6 < abs(a - b) <= step_tol)
    print(f"{k}: {len(r)} radii, {steps} off by one grid step ({grid:g}), {len(diffs)} larger:", diffs[:5])
    if diffs or steps > 1: ok = False
for eps in sorted(set(ref.get("fixed", {})) & set(new.get("fixed", {}))):
    for t in sorted(set(ref["fixed"][eps]) & set(new["fixed"][eps])):
        r, n = ref["fixed"][eps][t], new["fixed"][eps][t]
        nan_r, nan_n = [x is None or x != x for x in r], [x is None or x != x for x in n]
        ver_r, ver_n = sum(1 for x in r if x is not None and x == x and x > 0), sum(1 for x in n if x is not None and x == x and x > 0)
        d = [abs(a - b) for a, b, na, nb in zip(r, n, nan_r, nan_n) if not (na or nb)]
        print(f"eps {eps} {t}: verified {ver_r} vs {ver_n}, nan {sum(nan_r)} vs {sum(nan_n)}, max |lb delta| {max(d) if d else 0:.2e}")
        if nan_r != nan_n or ver_r != ver_n or (d and max(d) > tol): ok = False
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
