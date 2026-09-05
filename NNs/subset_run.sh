#!/bin/bash
# Subset check driver (runs on the compute node). The on-disk fullop_d3_dedup.txt was
# produced by pb2egg BEFORE transpose was un-gated (0 transpose vs subst's 1306), so it is
# NOT comparable. Regenerate B' with CURRENT pb2egg+prededup from the baseline .pb (same
# tooling that made S), then run the var-to-var instance check.
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
OUT="$REPO/NNs/reassoc_results"
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH="$OUT:$REPO/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/z3pkg:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages"
ts(){ date +%H:%M:%S; }

echo "[$(ts)] protoc bindings"
protoc -I "$REPO/taso/src/core" --python_out="$OUT" "$REPO/taso/src/core/rules.proto"

echo "[$(ts)] regen B' : pb2egg (current) on baseline fullop_d3_graph_subst.pb"
python3 "$REPO/NNs/pb2egg.py" "$OUT/fullop_d3_graph_subst.pb" "$OUT/fullop_d3_egg.CURRENT.txt" \
    --multi-out "$OUT/fullop_d3_egg.CURRENT.multi.pb" | sed 's/^/    /'

echo "[$(ts)] regen B' : prededup (alpha)"
python3 "$REPO/NNs/prededup.py" "$OUT/fullop_d3_egg.CURRENT.txt" "$OUT/fullop_d3_dedup.CURRENT.txt" | sed 's/^/    /'
echo "[$(ts)] B' lines: $(wc -l < "$OUT/fullop_d3_dedup.CURRENT.txt")   (S: $(wc -l < "$OUT/fullop_subst_d3_dedup.txt"))"

echo "[$(ts)] === SUBSET CHECK ==="
python3 "$REPO/NNs/subset_check.py" \
    "$OUT/fullop_subst_d3_dedup.txt" "$OUT/fullop_d3_dedup.CURRENT.txt"
echo "[$(ts)] DONE"
