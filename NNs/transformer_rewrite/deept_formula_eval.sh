#!/bin/bash
# Paired 294-position eval (same protocol/instances as gauges/deept_small6_seed0.pt) of FORMULA gauges vs the learned S6 gauge.
#   deept_formula_eval.sh <tag> <gauge1.pt[,gauge2.pt,...]>     (relative gauge paths; tags gauged, gauged2, ... in file order)
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
TAG=$1; GAUGES=$2; NAME=${3:-sst_bert_small_6}; EPS=${4:-0.01,0.02,0.03}   # eps grid of the model's earlier paired eval
mark() { echo "$1 formula_eval_$TAG $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mark START; "$PY" -u deept_gauge.py eval --name $NAME --gauge "$GAUGES" --max_len 12 --n_sent 40 --seed 0 --eps_list "$EPS" --hi 0.1 --iters 10 --save_json "$T/results/deept_formula_${TAG}_eval_short_seed0.json" > "$S/deept_eval_short_formula_$TAG.log" 2>&1; RC=$?; mark "DONE rc=$RC"; exit $RC
