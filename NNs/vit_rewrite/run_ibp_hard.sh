#!/bin/bash
# CPU-only re-learn of the ibp_3_3_8 gauge on HARD train boxes (stock CROWN min-lb nearest 0), concurrent with the official GPU runs.
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
cd "$REPO"; export OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=
echo "START_ibp_hard $(date)" >> "$S/official_sequence.log"
"$PY" NNs/vit_rewrite/vit_gauge_opt.py --model ibp_3_3_8 --steps 300 --batch 2 --n_train 192 --pool 600 --hard 1 --lr 0.01 --init svd --obj mix --softmax lse --seed 0 --log_every 25 --out NNs/vit_rewrite/gauges/ibp_mix_hard.pt > "$S/gauge_opt_ibp_hard.log" 2>&1
echo "DONE_ibp_hard $(date)" >> "$S/official_sequence.log"
