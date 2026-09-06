#!/bin/bash
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
export OMP_NUM_THREADS=4; cd "$REPO/NNs/transformer_rewrite"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "$PY" -u deept_gauge.py probe --name "${1:-sst_bert_small_3}" --eps "${2:-0.03}"
