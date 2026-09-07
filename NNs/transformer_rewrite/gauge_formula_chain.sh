#!/bin/bash
# Formula search for the attention gauge (gauge_formula.py validate): scores gauges of known CROWN quality + weight-only candidates
# on sst_bert_small_6.   gauge_formula_chain.sh smoke | full
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
W=$1; G="gauges"   # relative: the repo path contains spaces and $ARGS is word-split
ALL="$G/deept_small6_seed0.pt,$G/deept_small6_oodyelp_seed0.pt,$G/deept_small6_oodrandom_seed0.pt,$G/deept_small6_2w_seed0.pt,$G/pbvtrained_small6_inner_seed0.pt,$G/pbvtrained_small6_origin_seed0.pt"
mark() { echo "$1 formula_$W $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
case $W in
  smoke) ARGS="--n_sent 2 --pos 1 --n_sst 4 --l1_steps 20 --hi 0.05 --gauges $G/deept_small6_seed0.pt,$G/pbvtrained_small6_origin_seed0.pt"; OUT="$T/results/formula_small6_smoke.json" ;;
  full)  ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 400 --hi 0.1 --gauges $ALL"; OUT="$T/results/formula_small6_validate.json" ;;
  big3)  NAME=sst_bert_big_3;   ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 400 --hi 0.1 --gauges $G/deept_big3_seed0.pt";  OUT="$T/results/formula_big3_validate.json" ;;
  yelp3) NAME=yelp_bert_small_3; ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 400 --hi 0.1 --gauges $G/deept_yelp3_seed0.pt"; OUT="$T/results/formula_yelp3_validate.json" ;;
esac
mark START; "$PY" -u gauge_formula.py validate --name ${NAME:-sst_bert_small_6} $ARGS --seed 0 --out "$OUT" >> "$S/formula_$W.log" 2>&1; RC=$?; mark "DONE rc=$RC"; exit $RC
