#!/bin/bash
# Reconstruct all tensat forms for a model to ONNX (lowering ewmax/ewmin->relu),
# recording each form's tree depth. Runs in the container.
# Usage: recon_forms.sh <repo> <export_prefix> <wN_npz> <out_subdir>
#   export_prefix: e.g. lattice_out  (matches tensat/tmp/<prefix>_diverse*.model)
set -u
REPO="$1"; PREFIX="$2"; WN="$3"; SUB="$4"; cd "$REPO"
export LD_LIBRARY_PATH=$PWD/taso/build_gpu:/opt/conda/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages
export PYTHONPATH=$PWD/taso/python:$PYTHONPATH
PY=/opt/conda/bin/python3
OUT="NNs/reassoc_results/$SUB"; mkdir -p "$OUT"; rm -f "$OUT"/recon_*.onnx; > "$OUT/manifest.txt"
for f in tensat/tmp/${PREFIX}_diverse*.model; do
  case "$f" in *.json) continue;; esac
  i=$(basename "$f" .model | sed 's/.*diverse//')
  depth=$($PY -c "import structural_signature as ss; print(ss.analyze('$f')['max_depth'])" 2>/dev/null)
  [ -z "$depth" ] && depth=-1
  $PY NNs/reconstruct_generic.py "$f" "$WN" "$f.weight_names.json" "$OUT/recon_$i.onnx" >/dev/null 2>&1 \
    && echo "$i $depth $OUT/recon_$i.onnx" >> "$OUT/manifest.txt" \
    || echo "$i $depth RECON_FAILED" >> "$OUT/manifest.txt"
done
echo "reconstructed: $(grep -c onnx "$OUT/manifest.txt") / $(ls tensat/tmp/${PREFIX}_diverse*.model 2>/dev/null | grep -vc json)"
echo "depths: $(awk '{print $2}' "$OUT/manifest.txt" | sort -n | uniq -c | tr '\n' ' ')"
echo RECON_DONE
