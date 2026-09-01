#!/bin/bash
# Reverify ONE model with a given rule set, end to end, on a GPU node:
#   [1] tensat forms  (diverse sampling + optional verif_cost-steered form)   [container, GPU taso]
#   [2] reconstruct    each form -> ONNX (ewmax/ewmin -> relu)                 [container]
#   [3] alpha-CROWN    certified upper bound + unstable-ReLU count per form    [abcrown venv, GPU]
# Usage: reverify_model.sh <name> <taso> <wN.npz> <ref_onnx_relpath> <wbx.npz_relpath> <rules.txt> [intervals.json]
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
NAME=$1; TASO=$2; WN=$3; REF=$4; WBX=$5; RULES=$6; INTERVALS=${7:-}
SIF="$REPO/tensat.sif"
VENV="$REPO/alpha-beta-CROWN/.venv/bin/python"
SUB="${NAME}_core_forms"
FORMS="$REPO/NNs/reassoc_results/$SUB"
echo "############### REVERIFY: $NAME  (rules=$(basename "$RULES")) ###############"

echo "=== [1] tensat forms (CPU taso) ==="
apptainer exec --nv "$SIF" bash -lc "
  cd '$REPO/tensat'
  export LD_LIBRARY_PATH=\$PWD/../taso/build:/opt/conda/lib
  export CUDA_VISIBLE_DEVICES=0
  rm -f tmp/${NAME}_core_div*.model* tmp/${NAME}_core_vc*.model*
  echo '--- n_diverse 20 ---'
  ./target/debug/tensat -r '$RULES' -s none --model_file '$TASO' \
    --n_iter 12 --n_sec 120 --n_nodes 500000 --no_cycle --no_runtime_report \
    --n_diverse 20 --export_model tmp/${NAME}_core_div 2>&1 | grep -iaE 'panic|error|abort|diverse|export|saturat' | tail -5
  ls tmp/${NAME}_core_div_diverse*.model 2>/dev/null | grep -vc json | xargs echo 'diverse forms exported:'
  if [ -n '$INTERVALS' ]; then
    echo '--- verif_cost steered ---'
    ./target/debug/tensat -r '$RULES' -s none --model_file '$TASO' \
      --n_iter 12 --n_sec 120 --n_nodes 500000 --no_cycle --no_runtime_report \
      --verif_cost --interval_file '$INTERVALS' --export_model tmp/${NAME}_core_vc 2>&1 | grep -iaE 'verif-cost|leaf intervals|gap-cost|panic|error' | tail -4
  fi
"

echo "=== [2] reconstruct forms -> ONNX ==="
apptainer exec --nv "$SIF" bash -lc "bash '$REPO/NNs/recon_forms.sh' '$REPO' '${NAME}_core_div' '$WN' '$SUB'"
# also reconstruct the verif_cost form (if any) into the same subdir + manifest
if [ -n "$INTERVALS" ]; then
  apptainer exec --nv "$SIF" bash -lc "
    cd '$REPO'
    export LD_LIBRARY_PATH=\$PWD/taso/build:/opt/conda/lib
    export PYTHONPATH=\$PWD/taso/python:\$PWD/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages
    f=tensat/tmp/${NAME}_core_vc_verif.model
    if [ -f \"\$f\" ]; then
      d=\$(/opt/conda/bin/python3 -c \"import structural_signature as ss; print(ss.analyze('\$f')['max_depth'])\" 2>/dev/null)
      /opt/conda/bin/python3 NNs/reconstruct_generic.py \"\$f\" '$WN' \"\$f.weight_names.json\" '$FORMS/recon_vc.onnx' >/dev/null 2>&1 \
        && echo \"vc \${d:-?} $FORMS/recon_vc.onnx\" >> '$FORMS/manifest.txt' && echo 'verif_cost form reconstructed'
    fi
  "
fi

echo "=== [3] alpha-CROWN bounds (abcrown venv, GPU) ==="
cd "$REPO"
"$VENV" NNs/bound_forms.py "$SUB" "$REF" "$WBX"
echo "############### $NAME DONE ###############"
