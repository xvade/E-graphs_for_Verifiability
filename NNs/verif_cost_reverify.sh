#!/bin/bash
# verif_cost (deterministic, IBP-gap-targeted) reverify -- the RIGHT extraction for a
# verifiability WIN, immune to the --n_diverse collapse. Per model: tensat --verif_cost ->
# reconstruct the single steered form -> alpha-CROWN. Run on a GPU node (reconstruct's taso
# python binding is GPU-linked). Results tee'd to a durable table.
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
R="$REPO/NNs/reassoc_results"; SIF="$REPO/tensat.sif"
VENV="$REPO/alpha-beta-CROWN/.venv/bin/python"
CORE="$R/relaxed_d3_core.txt"
RES="$R/verif_cost_reverify_results.txt"
echo "==== verif_cost reverify (1097 core) $(date) ====" | tee "$RES"

run() {  # name taso wN ref wbx intervals
  local NAME=$1 TASO=$2 WN=$3 REF=$4 WBX=$5 INT=$6
  local SUB=${NAME}_vc_forms FORMS=$R/${NAME}_vc_forms
  echo "### $NAME (verif_cost, 1097 core) ###" | tee -a "$RES"
  # stage 1: CPU-taso saturation + verif_cost steered extraction
  apptainer exec --nv "$SIF" bash -lc "
    cd '$REPO/tensat'; export LD_LIBRARY_PATH=\$PWD/../taso/build:/opt/conda/lib
    rm -f tmp/${NAME}_vconly*.model*
    ./target/debug/tensat -r '$CORE' -s none --model_file '$TASO' \
      --n_iter 25 --n_sec 150 --n_nodes 800000 --no_cycle --no_runtime_report \
      --verif_cost --interval_file '$INT' --export_model tmp/${NAME}_vconly 2>&1 \
      | grep -iaE 'verif-cost|gap-cost|leaf interval|Stopped|iterations|Assertion|panic' | tail -5
  " 2>&1 | grep -vE "WARNING|Terminating|Timeouts|libtinfo|no version" | tee -a "$RES"
  # stage 2: reconstruct the single _verif form (GPU node -> taso python works)
  apptainer exec --nv "$SIF" bash -lc "
    cd '$REPO'; export LD_LIBRARY_PATH=\$PWD/taso/build:/opt/conda/lib
    export PYTHONPATH=\$PWD/taso/python:\$PWD/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages
    mkdir -p '$FORMS'; f=tensat/tmp/${NAME}_vconly_verif.model
    if [ -f \"\$f\" ]; then
      d=\$(/opt/conda/bin/python3 -c \"import structural_signature as ss; print(ss.analyze('\$f')['max_depth'])\" 2>/dev/null)
      /opt/conda/bin/python3 NNs/reconstruct_generic.py \"\$f\" '$WN' \"\$f.weight_names.json\" '$FORMS/recon_vc.onnx' 2>&1 | tail -3
      echo \"vc \${d:-0} $FORMS/recon_vc.onnx\" > '$FORMS/manifest.txt'
      echo \"verif form depth \${d:-?}, reconstructed\"
    else echo 'NO _verif.model exported'; fi
  " 2>&1 | grep -vE "WARNING|Terminating|Timeouts|libtinfo|no version" | tee -a "$RES"
  # stage 3: alpha-CROWN bound of the single steered form
  cd "$REPO"; "$VENV" NNs/bound_forms.py "$SUB" "$REF" "$WBX" 2>&1 \
    | grep -vE "Warning|warn|Deprecation|onnx2pytorch|torch.jit|tracemalloc|experimental|Early stop" \
    | grep -iE "form depth|^ *vc |^ *[0-9]+ +[0-9]|tensat forms:|numeric-gate" | tee -a "$RES"
}

run maxout  "$REPO/NNs/maxout.taso"  "$R/maxout_wN.npz"  NNs/maxout.onnx  NNs/reassoc_results/maxout_wbx.npz  "$R/maxout_intervals.json"
run lattice "$REPO/NNs/lattice.taso" "$R/lattice_wN.npz" NNs/lattice.onnx NNs/reassoc_results/lattice_wbx.npz "$R/lattice_intervals.json"
echo "==== DONE $(date). baselines: maxout input 12.0257 (prior verif_cost 9.65); lattice plateau 8.50 ====" | tee -a "$RES"
