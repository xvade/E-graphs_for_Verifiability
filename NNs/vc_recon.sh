#!/bin/bash
# Reconstruct the verif_cost-steered forms (CPU-only, taso python RPATH now -> taso/build).
# Run inside: apptainer exec tensat.sif bash NNs/vc_recon.sh
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
cd "$REPO"
export LD_LIBRARY_PATH="$PWD/taso/build:/opt/conda/lib"
export PYTHONPATH="$PWD/taso/python:$PWD/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages"
PY=/opt/conda/bin/python3
R=NNs/reassoc_results
for NAME in maxout lattice; do
  case $NAME in
    maxout)  WN=$R/maxout_wN.npz;;
    lattice) WN=$R/lattice_wN.npz;;
  esac
  f=tensat/tmp/${NAME}_vconly_verif.model
  SUB=$R/${NAME}_vc_forms; mkdir -p "$SUB"
  d=$("$PY" -c "import structural_signature as ss; print(ss.analyze('$f')['max_depth'])" 2>/dev/null)
  echo "### $NAME verif form depth ${d:-?} ###"
  "$PY" NNs/reconstruct_generic.py "$f" "$WN" "$f.weight_names.json" "$SUB/recon_vc.onnx" 2>&1 | tail -2
  echo "vc ${d:-0} $SUB/recon_vc.onnx" > "$SUB/manifest.txt"
  sz=$(stat -c %s "$SUB/recon_vc.onnx" 2>/dev/null || echo 0)
  echo "  onnx bytes: $sz"
done
