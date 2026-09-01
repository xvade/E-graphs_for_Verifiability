#!/bin/bash
# Batch reverify (maxout, lattice, mnist_tiny_mlp) with the 1097 PWL+matmul core, on a GPU node.
# Each model: tensat forms -> reconstruct -> alpha-CROWN cert_ub. Results tee'd to a durable log.
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
R="$REPO/NNs/reassoc_results"
CORE="$R/relaxed_d3_core.txt"
RES="$R/reverify_core1097_results.txt"
echo "==== reverify with 1097 core  $(date) ====" | tee "$RES"

# preflight the abcrown venv on this GPU
"$REPO/alpha-beta-CROWN/.venv/bin/python" -c "import torch;print('venv torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | tee -a "$RES"

run() {  # name taso wN ref wbx [intervals]
  bash "$REPO/NNs/reverify_model.sh" "$1" "$2" "$3" "$4" "$5" "$CORE" "${6:-}" 2>&1 | tee -a "$RES"
}

run maxout         "$REPO/NNs/maxout.taso"         "$R/maxout_wN.npz"          NNs/maxout.onnx          NNs/reassoc_results/maxout_wbx.npz          "$R/maxout_intervals.json"
run lattice        "$REPO/NNs/lattice.taso"        "$R/lattice_wN.npz"         NNs/lattice.onnx         NNs/reassoc_results/lattice_wbx.npz         "$R/lattice_intervals.json"
run mnist_tiny_mlp "$REPO/NNs/mnist_tiny_mlp.taso" "$R/mnist_tiny_mlp_wN.npz"  NNs/mnist_tiny_mlp.onnx  NNs/reassoc_results/mnist_tiny_mlp_wbx.npz
echo "==== BATCH DONE $(date) ====" | tee -a "$RES"
