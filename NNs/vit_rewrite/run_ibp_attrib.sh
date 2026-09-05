#!/bin/bash
# Slack attribution for ibp_3_3_8 (INEXACT diagnostics): linearize one attention nonlinearity at a time; lse vanilla CROWN, 8 instances.
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
cd "$REPO"; export OMP_NUM_THREADS=4
INST="9119,2351,2675,4879,6025,6263,5233,6948"
for d in "" linQK linSM linAV linQK,linSM,linAV; do
  tag=${d:-none}
  "$PY" NNs/vit_rewrite/vit_bounds.py --model ibp_3_3_8 --variant base --softmax lse --methods CROWN --instances "$INST" --width 1 --diag "$d" --tag "_attrib_${tag//,/+}" 2>/dev/null | grep -E "SUMMARY|^\s*\[" | tail -1 | sed "s/^/diag=$tag  /"
done > "$S/ibp_attrib.log"
echo "DONE_attrib $(date)" >> "$S/official_sequence.log"
