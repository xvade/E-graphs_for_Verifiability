#!/bin/bash
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
cd "$REPO"; export OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=
"$PY" NNs/vit_rewrite/vit_bounds.py --model pgd_2_3_16 --variant base --gauge_file NNs/vit_rewrite/gauges/pgd_mix_idinit.pt --softmax lse --methods CROWN --instances all --width 1 --mc 100 > "$S/full100_pgd_idinit.log" 2>&1
echo "DONE_idinit_eval $(date)" >> "$S/official_sequence.log"
