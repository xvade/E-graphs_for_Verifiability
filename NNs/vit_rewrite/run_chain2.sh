#!/bin/bash
# Resumed chain after the user's interactive allocation ended (23:28). GPU stays exclusive to the official abcrown
# runs (learnedG -> stock -> R45, untouched vit.yaml settings); CPU-only side jobs run concurrently.
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
S="$REPO/NNs/vit_rewrite/_scratch"
PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
echo "START_chain2 $(date) on $(hostname)" >> "$S/official_sequence.log"
# CPU side jobs (no GPU): out-of-sample eval of the learned ibp gauge; init=id robustness learner for pgd
( export OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=; "$PY" NNs/vit_rewrite/vit_bounds.py --model ibp_3_3_8 --variant base --gauge_file NNs/vit_rewrite/gauges/ibp_mix_svdinit.pt --softmax lse --methods CROWN --instances all --width 1 --mc 100 > "$S/full100_ibp_learnedG.log" 2>&1; echo "DONE_ibp_eval $(date)" >> "$S/official_sequence.log" ) &
( export OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=; "$PY" NNs/vit_rewrite/vit_gauge_opt.py --model pgd_2_3_16 --steps 200 --batch 32 --n_train 512 --lr 0.01 --init id --obj mix --softmax lse --seed 1 --log_every 10 --out NNs/vit_rewrite/gauges/pgd_mix_idinit.pt > "$S/gauge_opt_pgd_idinit.log" 2>&1; echo "DONE_idinit $(date)" >> "$S/official_sequence.log" ) &
export OMP_NUM_THREADS=4  # 10 CPUs total on this allocation: 3+3 side jobs + 4 abcrown
cd "$REPO/alpha-beta-CROWN/complete_verifier"
for name in learnedG_pgd stock_pgd R45_both_svd; do
  echo "START_$name $(date)" >> "$S/official_sequence.log"
  "$PY" -u abcrown.py --config "$REPO/NNs/vit_rewrite/cfg_vit_$name.yaml" > "$S/official_$name.log" 2>&1
done
wait
echo "DONE $(date)" >> "$S/official_sequence.log"
