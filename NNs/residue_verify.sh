#!/bin/bash
# Close the subset-check hedge: dump the truly-novel residue (rules in baseline B' whose
# equivalence S carries in NEITHER orientation) and Z3-verify them, so "truly-novel" gets a
# MEASURED sound count instead of a "likely-unsound" hand-wave.
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
OUT="$REPO/NNs/reassoc_results"
CONDA=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/miniconda3
TASOPY="$CONDA/envs/taso_py/bin/python3"
Z3PKG=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/z3pkg
export LD_LIBRARY_PATH="$REPO/taso/build:$CONDA/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$OUT:$REPO/NNs:$Z3PKG"
ts(){ date +%H:%M:%S; }

echo "[$(ts)] dump truly-novel residue"
$TASOPY "$REPO/NNs/subset_orient.py" \
    "$OUT/fullop_subst_d3_dedup.txt" "$OUT/fullop_d3_dedup.CURRENT.txt" \
    "$OUT/residue_novel.txt" | sed 's/^/    /'

echo "[$(ts)] Z3-verify the truly-novel residue -> residue_novel_verified.txt"
$TASOPY "$REPO/NNs/z3_verify_egg.py" \
    "$OUT/residue_novel.txt" "$OUT/residue_novel_verified.txt" --timeout_ms 10000 | sed 's/^/    /'
echo "[$(ts)] novel sound: $(wc -l < "$OUT/residue_novel_verified.txt")  (of $(wc -l < "$OUT/residue_novel.txt"))"
echo "[$(ts)] novel-sound min/max: $(grep -cE 'ewmax|ewmin' "$OUT/residue_novel_verified.txt" || true)"
echo "[$(ts)] DONE"
