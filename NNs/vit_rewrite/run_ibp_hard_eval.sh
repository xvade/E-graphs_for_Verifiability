#!/bin/bash
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
cd "$REPO"; export OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=
until grep -q "^DONE_ibp_hard " "$S/official_sequence.log"; do sleep 60; done
[ -f NNs/vit_rewrite/gauges/ibp_mix_hard.pt ] || { echo "NO_GAUGE_FILE $(date)" >> "$S/official_sequence.log"; exit 1; }
"$PY" NNs/vit_rewrite/vit_bounds.py --model ibp_3_3_8 --variant base --gauge_file NNs/vit_rewrite/gauges/ibp_mix_hard.pt --softmax lse --methods CROWN --instances all --width 1 --mc 100 > "$S/full100_ibp_hardG.log" 2>&1
echo "DONE_ibp_hard_eval $(date)" >> "$S/official_sequence.log"
