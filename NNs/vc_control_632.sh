#!/bin/bash
# CONTROL: maxout verif_cost extraction with the ORIGINAL 632 pwl_rules_ac.txt (the rule set
# that produced the 9.65 win), through the CURRENT binary + fixed CPU reconstruct path.
# Discriminates: 1097-core-lacks-AC vs verif_cost-binary-regression.
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; cd "$REPO"
R=NNs/reassoc_results; RULES=$R/pwl_rules_ac.txt; INT=$R/maxout_intervals.json
PFX=maxout_ac632
# stage 1: extraction (CPU taso)
cd "$REPO/tensat"; export LD_LIBRARY_PATH="$PWD/../taso/build:/opt/conda/lib"
rm -f tmp/${PFX}_verif.model*
./target/debug/tensat -r "../$RULES" -s none --model_file "../NNs/maxout.taso" \
  --n_iter 25 --n_sec 150 --n_nodes 800000 --no_cycle --no_runtime_report \
  --verif_cost --interval_file "../$INT" --export_model tmp/${PFX} 2>&1 \
  | grep -iaE 'verif-cost|gap-cost|leaf interval|Stopped|iterations|panic' | tail -6
cd "$REPO"
# stage 2: reconstruct (CPU, RPATH-fixed taso python)
export LD_LIBRARY_PATH="$PWD/taso/build:/opt/conda/lib"
export PYTHONPATH="$PWD/taso/python:$PWD/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages"
PY=/opt/conda/bin/python3
f=tensat/tmp/${PFX}_verif.model
SUB=$R/${PFX}_forms; mkdir -p "$SUB"
if [ -f "$f" ]; then
  d=$("$PY" -c "import structural_signature as ss; print(ss.analyze('$f')['max_depth'])" 2>/dev/null)
  "$PY" NNs/reconstruct_generic.py "$f" "$R/maxout_wN.npz" "$f.weight_names.json" "$SUB/recon_vc.onnx" 2>&1 | tail -1
  echo "0 ${d:-0} NNs/reassoc_results/${PFX}_forms/recon_vc.onnx" > "$SUB/manifest.txt"
  echo "CONTROL verif form depth ${d:-?}"
else echo "NO _verif.model exported"; fi
