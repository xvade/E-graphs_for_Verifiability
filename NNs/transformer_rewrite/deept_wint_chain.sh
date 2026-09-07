#!/bin/bash
# Rigorous certificate transfer: paired eval of sst_bert_small_6 with the folded attention weights declared as 2-ulp fp32 intervals
# (stock_wint / gauged_wint next to stock / gauged), so the gauged certificate provably covers the exact rewrite = the original network.
#   deept_wint_chain.sh smoke   -> 2 test sentences <= 6 tokens, 4 bisection steps (minutes)
#   deept_wint_chain.sh full    -> the standard 40 test sentences <= 12 tokens / 294 positions (~4-5 h on an L40S)
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
W=$1; G="$T/gauges/deept_small6_seed0.pt"
mark() { echo "$1 wint_$W $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
case $W in
  smoke) ARGS="--max_len 6 --n_sent 2 --iters 4"; OUT="$T/results/deept_small6_wint_smoke.json" ;;
  full)  ARGS="--max_len 12 --n_sent 40 --iters 10"; OUT="$T/results/deept_small6_wint_eval_short_seed0.json" ;;
esac
mark START; "$PY" -u deept_gauge.py eval --name sst_bert_small_6 --weight_intervals 1 --gauge "$G" $ARGS --seed 0 --eps_list 0.01,0.02,0.03 --hi 0.1 --save_json "$OUT" > "$S/deept_wint_$W.log" 2>&1; mark "DONE rc=$?"
