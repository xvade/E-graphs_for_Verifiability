#!/bin/bash
# The single comparison: CROWN-Optimized (alpha-CROWN, 20 it) on the stock weights vs on the gauged weights, gauge trained ONCE by
# the standard learner (gauges/deept_<m>_seed0.pt).  Per-instance resumable (.part_* files), skip-if-done, --requeue for ckpt.
#   sbatch alpha_single.sbatch <small6|big3> <max_len> <eps_list> <tag>
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"; M="$1"; ML="$2"; EPS="$3"; TAG="$4"
case $M in small6) NAME=sst_bert_small_6 ;; big3) NAME=sst_bert_big_3 ;; small3) NAME=sst_bert_small_3 ;; esac
EJ="$T/results/deept_${M}_eval_alpha_${TAG}.json"; LOG="$S/deept_eval_alpha_${M}_${TAG}.log"
mark() { echo "$1 alpha_${M}_${TAG} $(date) job=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0}" >> "$S/official_sequence.log"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; mark START
if [ -f "$EJ" ]; then mark "already complete, skipped"; exit 0; fi
"$PY" -u deept_gauge.py eval_alpha --name $NAME --n_sent 500 --max_len $ML --eps_list $EPS --seed 0 --hi 0.1 --gauge gauges/deept_${M}_seed0.pt --save_json "$EJ" >> "$LOG" 2>&1; RC=$?
mark "DONE rc=$RC"; exit $RC
