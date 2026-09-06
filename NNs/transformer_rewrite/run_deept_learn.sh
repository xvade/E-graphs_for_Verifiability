#!/bin/bash
# usage: run_deept_learn.sh <name> <out.pt> [extra args...]
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$REPO/NNs/transformer_rewrite"
name="$1"; out="$2"; shift 2; case "$out" in /*) ;; *) out="$REPO/$out";; esac   # the learner runs with cwd NNs/transformer_rewrite
"$PY" -u deept_gauge.py learn --name "$name" --out "$out" "$@"
