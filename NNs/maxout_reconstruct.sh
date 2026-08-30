#!/bin/bash
# Reconstruct all 40 tensat maxout forms to ONNX (lowering ewmax->relu), recording
# each form's tree depth. Runs in the container (taso python).
set -u
REPO="$1"; cd "$REPO"
export LD_LIBRARY_PATH=$PWD/taso/build_gpu:/opt/conda/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/taso/python:$PWD/NNs/reassoc_results:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages
PY=/opt/conda/bin/python3
WN=NNs/reassoc_results/maxout_wN.npz
OUT=NNs/reassoc_results/maxout_forms; mkdir -p "$OUT"; rm -f "$OUT"/recon_*.onnx
> "$OUT/manifest.txt"
for f in tensat/tmp/maxout_out_diverse*.model; do
  case "$f" in *.json) continue;; esac
  i=$(basename "$f" .model | sed 's/.*diverse//')
  depth=$($PY -c "import structural_signature as ss; print(ss.analyze('$f')['max_depth'])" 2>/dev/null)
  $PY NNs/reconstruct_generic.py "$f" "$WN" "$f.weight_names.json" "$OUT/recon_$i.onnx" >/dev/null 2>&1 \
    && echo "$i $depth $OUT/recon_$i.onnx" >> "$OUT/manifest.txt" \
    || echo "$i $depth RECON_FAILED" >> "$OUT/manifest.txt"
done
echo "reconstructed: $(grep -c onnx "$OUT/manifest.txt") / $(ls tensat/tmp/maxout_out_diverse*.model | grep -vc json)"
echo "depth range: $(awk '{print $2}' "$OUT/manifest.txt" | sort -n | head -1) .. $(awk '{print $2}' "$OUT/manifest.txt" | sort -n | tail -1)"
echo "distinct depths: $(awk '{print $2}' "$OUT/manifest.txt" | sort -n | uniq -c | tr '\n' ' ')"
echo RECON_DONE
