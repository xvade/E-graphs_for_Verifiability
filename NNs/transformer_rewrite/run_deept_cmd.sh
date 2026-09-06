#!/bin/bash
# generic: run_deept_cmd.sh <cmd> [args...]  (cwd = NNs/transformer_rewrite; relative --out/--save_json paths are made absolute by the caller)
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$REPO/NNs/transformer_rewrite"
"$PY" -u deept_gauge.py "$@"
