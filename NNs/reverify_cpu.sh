#!/bin/bash
# Reverify one model with a rule set, ALL ON CPU (no GPU): stage1 CPU-taso saturation+diverse
# extraction, stage2 reconstruct, stage3 alpha-CROWN on CPU (bound_forms falls back; toy nets).
# Usage: reverify_cpu.sh <name> <taso> <wN> <ref_relpath> <wbx_relpath> <rules> <n_diverse> <n_sec> <tag>
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
NAME=$1; TASO=$2; WN=$3; REF=$4; WBX=$5; RULES=$6; NDIV=$7; NSEC=$8; TAG=$9
SUB="${NAME}_${TAG}_forms"; PFX="${NAME}_${TAG}"
echo "### $NAME [$TAG] rules=$(basename "$RULES") n_diverse=$NDIV n_sec=$NSEC ###"

apptainer exec "$REPO/tensat.sif" bash -lc "
  cd '$REPO/tensat'
  export LD_LIBRARY_PATH=\$PWD/../taso/build:/opt/conda/lib
  rm -f tmp/${PFX}_div*.model*
  ./target/debug/tensat -r '$RULES' -s none --model_file '$TASO' \
    --n_iter 30 --n_sec $NSEC --n_nodes 800000 --no_cycle --no_runtime_report \
    --n_diverse $NDIV --export_model tmp/${PFX}_div 2>&1 | grep -iaE 'Stopped|iterations|Diverse sample 0|Assertion|panic' | tail -4
  ls tmp/${PFX}_div_diverse*.model 2>/dev/null | grep -vc json | xargs echo '  diverse forms:'
"
apptainer exec "$REPO/tensat.sif" bash "$REPO/NNs/recon_forms.sh" "$REPO" "${PFX}_div" "$WN" "$SUB" 2>&1 \
  | grep -iE "reconstructed|depths"
echo "  --- CPU alpha-CROWN ---"
cd "$REPO"; CUDA_VISIBLE_DEVICES="" "$REPO/alpha-beta-CROWN/.venv/bin/python" NNs/bound_forms.py "$SUB" "$REF" "$WBX" 2>&1 \
  | grep -vE "Warning|warn|Deprecation|onnx2pytorch|torch.jit|tracemalloc|experimental|Early stop" \
  | grep -iE "tensat forms:|best form|improvement|numeric-gate" | tail -4
