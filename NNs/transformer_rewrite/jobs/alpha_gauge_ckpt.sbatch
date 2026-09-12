#!/bin/bash
#SBATCH -p ckpt-all -A ckpt-amath --qos=ckpt-gpu --gres=gpu:a100:1 -c 4 --mem=40G --time=10:00:00 --requeue
#SBATCH -o logs/alpha_gauge_%j.out
# alpha-CROWN (CROWN-Optimized, 20 it) with ONE given gauge on all instances, stock skipped (taken from the earlier single-comparison
# JSON offline).  Per-instance resumable (.part_gauged_<eps>), skip-if-done.
#   sbatch alpha_gauge.sbatch <small6|big3> <max_len> <eps_list> <tag> <gauge.pt (relative to transformer_rewrite)>
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"; M="$1"; ML="$2"; EPS="$3"; TAG="$4"; G="$5"
case $M in small6) NAME=sst_bert_small_6 ;; big3) NAME=sst_bert_big_3 ;; small3) NAME=sst_bert_small_3 ;; esac
EJ="$T/results/deept_${M}_eval_alpha_${TAG}.json"; LOG="$S/deept_eval_alpha_${M}_${TAG}.log"
mark() { echo "$1 alpha_${M}_${TAG} $(date) job=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0}" >> "$S/official_sequence.log"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; mark START
if [ -f "$EJ" ]; then mark "already complete, skipped"; exit 0; fi
"$PY" -u deept_gauge.py eval_alpha --name $NAME --n_sent 500 --max_len $ML --eps_list $EPS --seed 0 --hi 0.1 --skip_stock 1 --gauge "$G" --save_json "$EJ" >> "$LOG" 2>&1; RC=$?
mark "DONE rc=$RC"; exit $RC
