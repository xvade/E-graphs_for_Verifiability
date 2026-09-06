#!/bin/bash
# usage: run_official_genbab.sh <config.yaml> <tag>  -> UNMODIFIED abcrown with the GenBaB benchmark config (300 s timeout), alone on this node's GPU
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
cd "$REPO/alpha-beta-CROWN/complete_verifier"
echo "START_$2 $(date) on $(hostname)" >> "$S/official_sequence.log"
"$PY" -u abcrown.py --config "$1" > "$S/official_$2.log" 2>&1
echo "DONE_$2 $(date)" >> "$S/official_sequence.log"
