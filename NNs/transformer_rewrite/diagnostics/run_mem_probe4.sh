#!/bin/bash
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$REPO/NNs/transformer_rewrite"
"$REPO/alpha-beta-CROWN/.venv/bin/python" -u diagnostics/_mem_probe4.py
