#!/bin/bash
# usage: run_official_genbab.sh <config.yaml> <tag>  -> UNMODIFIED abcrown with the GenBaB benchmark config (300 s timeout), alone on this node's GPU
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
cd "$REPO/alpha-beta-CROWN/complete_verifier"
echo "START_$2 $(date) on $(hostname)" >> "$S/official_sequence.log"
"$PY" -u abcrown.py --config "$1" > "$S/official_$2.log" 2>&1
echo "DONE_$2 $(date)" >> "$S/official_sequence.log"
