#!/bin/bash
# usage: run_official_one.sh <name>   -> runs abcrown with cfg_vit_<name>.yaml alone on this node's GPU
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
cd "$REPO/alpha-beta-CROWN/complete_verifier"
echo "START_$1 $(date) on $(hostname)" >> "$S/official_sequence.log"
"$PY" -u abcrown.py --config "$REPO/NNs/vit_rewrite/cfg_vit_$1.yaml" > "$S/official_$1.log" 2>&1
echo "DONE_$1 $(date)" >> "$S/official_sequence.log"
