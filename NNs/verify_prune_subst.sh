#!/bin/bash
# Stage 4: Z3-verify the canonical subst dedup corpus (S = fullop_subst_d3_dedup.txt) ->
# fullop_subst_d3_verified.txt. Runs OUTSIDE the container (conda taso_py + z3). The prune
# (stage 5) is NOT here: tensat's debug binary needs the container glibc, so pruning lives in
# prune_subst.sh (split SAFE/STRUCT + prune SAFE inside tensat.sif). See that script.
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
OUT="$REPO/NNs/reassoc_results"
CONDA=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/miniconda3
TASOPY="$CONDA/envs/taso_py/bin/python3"
Z3PKG=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/z3pkg
export LD_LIBRARY_PATH="$REPO/taso/build:$CONDA/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$OUT:$REPO/NNs:$Z3PKG"
P=fullop_subst_d3
ts(){ date +%H:%M:%S; }

echo "[$(ts)] [4/5] Z3 verify -> ${P}_verified.txt"
S=$(date +%s)
$TASOPY "$REPO/NNs/z3_verify_egg.py" "$OUT/${P}_dedup.txt" "$OUT/${P}_verified.txt" \
    --timeout_ms 10000 | sed 's/^/    /'
E=$(date +%s); echo "[$(ts)] verify wall: $((E-S)) s"
echo "    verified lines: $(wc -l < "$OUT/${P}_verified.txt")  (in: $(wc -l < "$OUT/${P}_dedup.txt"))"
echo "[$(ts)] verify DONE. Prune stage: run prune_subst.sh (needs tensat.sif)."
