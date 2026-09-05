#!/bin/bash
# Verify DRY-RUN: run z3_verify_egg.py on head -N of the subst dedup set to get a
# per-rule rate before committing to the full verify. Same env as
# run_rule_gen_commute_fixed.sh stages 4/5 (runs OUTSIDE the container: conda taso_py).
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
OUT="$REPO/NNs/reassoc_results"
CONDA=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/miniconda3
TASOPY="$CONDA/envs/taso_py/bin/python3"
Z3PKG=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/z3pkg
export LD_LIBRARY_PATH="$REPO/taso/build:$CONDA/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$OUT:$REPO/NNs:$Z3PKG"
N=${1:-30}
head -n "$N" "$OUT/fullop_subst_d3_dedup.txt" > "$OUT/dry_in.txt"
echo "== import check =="
$TASOPY -c "import z3, tensor_axioms; print('z3 + tensor_axioms import OK')" || { echo "IMPORT FAILED"; exit 2; }
echo "== verify head -$N (timeout_ms 10000) =="
S=$(date +%s)
$TASOPY "$REPO/NNs/z3_verify_egg.py" "$OUT/dry_in.txt" "$OUT/dry_verified.txt" --timeout_ms 10000
E=$(date +%s)
echo "== result =="
echo "verified lines: $(wc -l < "$OUT/dry_verified.txt" 2>/dev/null || echo 0) / $N"
echo "wall: $((E-S)) s  => $(awk "BEGIN{printf \"%.2f\", ($E-$S)/$N}") s/rule"
echo "extrapolated full 4432: ~$(awk "BEGIN{printf \"%.0f\", ($E-$S)/$N*4432/60}") min"
