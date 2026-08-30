#!/bin/bash
# Verifiability-aware extraction on a model, then reconstruct the single steered form.
# Usage: verif_run.sh <repo> <model.taso> <rules.txt> <intervals.json> <wN.npz> <prefix>
set -u
REPO="$1"; MODEL="$2"; RULES="$3"; INTERVALS="$4"; WN="$5"; PREFIX="$6"
cd "$REPO/tensat"
export LD_LIBRARY_PATH=$PWD/../taso/build_gpu:/opt/conda/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/../taso/python:$PWD/../NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages
export CUDA_VISIBLE_DEVICES=0
rm -f tmp/${PREFIX}_verif*.model*
echo "=== verif-cost extraction ==="
./target/debug/tensat -r "$RULES" -s none --model_file "$MODEL" \
  --n_iter 12 --n_sec 120 --n_nodes 500000 --no_cycle --no_runtime_report \
  --verif_cost --interval_file "$INTERVALS" --export_model tmp/${PREFIX} 2>&1 \
  | grep -iaE "verif-cost|leaf intervals matched|gap-cost|panic|error|thread" | tail -6
f=tmp/${PREFIX}_verif.model
if [ -f "$f" ]; then
  /opt/conda/bin/python3 ../NNs/reconstruct_generic.py "$f" "$WN" "$f.weight_names.json" \
    ../NNs/reassoc_results/${PREFIX}_verif.onnx >/dev/null 2>&1 \
    && echo "reconstructed -> ${PREFIX}_verif.onnx (depth $(/opt/conda/bin/python3 -c "import structural_signature as ss; print(ss.analyze('$f')['max_depth'])" 2>/dev/null))" \
    || echo "RECON FAILED"
else
  echo "NO _verif.model exported"
fi
echo VERIF_RUN_DONE
