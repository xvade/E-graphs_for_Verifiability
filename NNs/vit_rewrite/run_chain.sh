#!/bin/bash
# Sequential GPU chain (run alone on the L40S so official BaB runs are not contaminated by memory contention):
#   1. learn gauge for ibp_3_3_8   2-4. UNMODIFIED official abcrown pipeline (vit.yaml settings) on learnedG / stock / R45.
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
S="$REPO/NNs/vit_rewrite/_scratch"
PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
echo "START_ibp_gauge $(date)" > "$S/official_sequence.log"
"$PY" NNs/vit_rewrite/vit_gauge_opt.py --model ibp_3_3_8 --steps 300 --batch 2 --n_train 128 --lr 0.01 --init svd --obj mix --softmax lse --seed 0 --out NNs/vit_rewrite/gauges/ibp_mix_svdinit.pt > "$S/gauge_opt_ibp_mix.log" 2>&1
cd "$REPO/alpha-beta-CROWN/complete_verifier"
for name in learnedG_pgd stock_pgd R45_both_svd; do
  echo "START_$name $(date)" >> "$S/official_sequence.log"
  "$PY" abcrown.py --config "$REPO/NNs/vit_rewrite/cfg_vit_$name.yaml" > "$S/official_$name.log" 2>&1
done
echo "DONE $(date)" >> "$S/official_sequence.log"
