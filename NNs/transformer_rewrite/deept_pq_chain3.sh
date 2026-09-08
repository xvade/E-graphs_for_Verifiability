#!/bin/bash
# Route A: per-query gauge optimisation on sst_bert_small_6 test instances (Adam on the gauge from the learned S6 init, plain CROWN
# in the loop; sound for every iterate).  Progressive save + resume, so it can be re-submitted after a pre-emption.
#   deept_pq_chain.sh smoke   -> 1 sentence <= 6 tokens, 3 steps, fixed eps + radius
#   deept_pq_chain.sh eps     -> 20 test sentences <= 12 tokens: verified counts at eps 0.02 / 0.03 (stock / fixed gauge / per-query)
#   deept_pq_chain.sh radius  -> same instances: certified radius stock / fixed gauge / per-query (optimised at the fixed gauge's radius)
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
W=$1; G="$T/gauges/deept_small6_seed0.pt"
mark() { echo "$1 pq_$W $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
case $W in
  smoke)  ARGS="--max_len 6 --n_sent 1 --pq_steps 3 --eps_list 0.02,0.03 --pq_radius 1 --iters 4"; OUT="$T/results/deept_small6_pq_smoke.json" ;;
  eps)    ARGS="--max_len 12 --n_sent 20 --pq_steps 20 --eps_list 0.02,0.03 --pq_radius 0 --iters 10 --pq_max_tokens 10"; OUT="$T/results/deept_small6_pq_eps_seed0.json" ;;
  radius) ARGS="--max_len 12 --n_sent 20 --pq_steps 20 --eps_list 0.02 --pq_stop_verified 1 --pq_radius 1 --iters 10"; OUT="$T/results/deept_small6_pq_radius_seed0.json" ;;
esac
for try in $(seq 1 80); do mark "START try=$try"; "$PY" -u deept_gauge.py eval_pq --name sst_bert_small_6 --gauge "$G" $ARGS --seed 0 --hi 0.1 --save_json "$OUT" >> "$S/deept_pq_$W.log" 2>&1; RC=$?; mark "DONE rc=$RC"; [ $RC -eq 3 ] || break; done; exit $RC
