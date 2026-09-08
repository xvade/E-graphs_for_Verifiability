#!/bin/bash
# Diagnostic ceiling for the formula: warm-start the CROWN learner from the closed-form gauge (few steps, low lr, the learner's own
# tuning boxes) and paired-eval the result against the gauge learned from identity.  This is a HYBRID (uses the verifier), not the
# manual procedure; it answers whether headroom above the learned gauge exists in the formula's basin.
#   deept_init_chain.sh <tag> <init_gauge.pt> <learned_gauge.pt> <model> <eps_list> [steps=40] [lr=0.005] [n_sent=40]
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
TAG=$1; INIT=$2; LEARNED=$3; NAME=$4; EPS=$5; STEPS=${6:-40}; LR=${7:-0.005}; NSENT=${8:-40}; OUT="gauges/deept_${TAG}_seed0.pt"
mark() { echo "$1 init_$TAG $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mark START
"$PY" -u deept_gauge.py learn --name $NAME --data auto --split dev --max_len 8 --n_sent $NSENT --pos_per_sent 3 --seed 0 --steps $STEPS --accum 4 --lr $LR --cond_pen 1e-4 --clip 1 \
  --hi 0.1 --radius_iters 8 --n_eval 48 --log_every 5 --gauge "$INIT" --out "$OUT" > "$S/deept_learn_$TAG.log" 2>&1 || { RC=$?; mark "DONE(learn failed) rc=$RC"; exit $RC; }
"$PY" -u deept_gauge.py eval --name $NAME --gauge "$OUT,$LEARNED" --max_len 12 --n_sent 40 --seed 0 --eps_list "$EPS" --hi 0.1 --iters 10 \
  --save_json "$T/results/deept_formula_${TAG}_eval_short_seed0.json" > "$S/deept_eval_short_formula_$TAG.log" 2>&1; RC=$?; mark "DONE rc=$RC"; exit $RC
