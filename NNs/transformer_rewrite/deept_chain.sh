#!/bin/bash
# usage: deept_chain.sh <what>...            (what in alpha|alpha6|alpha5|alldev|small6|small12|small12b|long0|long1, run in order, one GPU)
#   as a batch job:   sbatch -p gpu-l40s -A gpu-l40s-amath --gres=gpu:l40s:1 -c 5 --mem=25G -t 16:00:00 deept_chain.sh alpha small12 long0
#   inside a running allocation: JOBID=<id> NODE=<node> deept_chain.sh ...   (steps via srun --jobid --overlap; waits for a free node)
# One GPU job per GPU (concurrent jobs OOM'd each other). Each step logs to NNs/vit_rewrite/_scratch/deept_*.log and marks
# START/DONE in official_sequence.log.
#   alpha  : paired alpha-CROWN eval (CROWN-Optimized, 20 it) of the small_3 seed-0 gauge, test sentences <= 8 tokens, eps 0.03/0.04
#   alpha6 : same for the small_6 seed-0 gauge on test sentences <= 6 tokens, eps 0.02/0.03 — CUDA OOM at 44 GB (2026-09-06; alpha-CROWN
#            retains far more than plain grad-mode CROWN, which fit <= 8 tokens for the learner); alpha5 retries at <= 5 tokens
#   alldev : small_3 learner on ALL dev-set positions of sentences <= 10 tokens (overfitting test) + short paired eval
#   small6 : learner (dev sentences <= 8 tokens: the 6-layer autograd graph is ~2x small_3's) + short paired eval, sst_bert_small_6
#   small12: same for sst_bert_small_12 (dev sentences <= 7 tokens) — CUDA OOM at 44 GB (2026-09-06); small12b retries at <= 6 tokens, 5 positions/sentence -- OOM'd at the first grad-mode CROWN on the 12-layer graph (44 GB)
#   small12b: retry with dev sentences <= 6 tokens (5 positions each), falling back to <= 5 tokens if that OOMs too; then the short eval
#   long0/1: long-sentence paired eval (12 test sentences <= 32 tokens, 251 positions) of the small_3 seed-0 / seed-1 gauge
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
if [ -n "$JOBID" ]; then
  while squeue -s -u sgvtc -h -o "%j %N" | grep -v "interact\|extern" | grep -q " $NODE\$"; do sleep 60; done
  RUN="srun --jobid=$JOBID --overlap -N1 -n1 -w $NODE --cpus-per-task=4 --gres=gpu:l40s:1"
else RUN=""; fi
learn_eval() {  # name tag max_len n_sent pos_per_sent steps accum
  local name=$1 tag=$2 G="$T/gauges/deept_${2}_seed0.pt"
  $RUN "$PY" -u deept_gauge.py learn --name $name --split dev --max_len $3 --n_sent $4 --pos_per_sent $5 --steps $6 --accum $7 --lr 0.01 --seed 0 --log_every 10 --out "$G" > "$S/deept_learn_$tag.log" 2>&1
  [ -f "$G" ] && $RUN "$PY" -u deept_gauge.py eval --name $name --gauge "$G" --max_len 12 --n_sent 40 --seed 0 --eps_list 0.01,0.02,0.03 --hi 0.1 --iters 10 --save_json "$T/results/deept_${tag}_eval_short_seed0.json" > "$S/deept_eval_short_$tag.log" 2>&1
}
for what in "$@"; do
  echo "START_deept_$what $(date) on $(hostname) job ${SLURM_JOB_ID:-$JOBID}" >> "$S/official_sequence.log"
  case $what in
    alpha)   $RUN "$PY" -u deept_gauge.py eval_alpha --name sst_bert_small_3 --gauge "$T/gauges/deept_small3_seed0.pt" --max_len 8 --n_sent 44 --seed 0 --eps_list 0.03,0.04 --save_json "$T/results/deept_small3_eval_alpha_seed0.json" > "$S/deept_eval_alpha_seed0.log" 2>&1 ;;
    alpha6)  $RUN "$PY" -u deept_gauge.py eval_alpha --name sst_bert_small_6 --gauge "$T/gauges/deept_small6_seed0.pt" --max_len 6 --n_sent 99 --seed 0 --eps_list 0.02,0.03 --save_json "$T/results/deept_small6_eval_alpha_seed0.json" > "$S/deept_eval_alpha_small6_seed0.log" 2>&1 ;;
    alpha5)  $RUN "$PY" -u deept_gauge.py eval_alpha --name sst_bert_small_6 --gauge "$T/gauges/deept_small6_seed0.pt" --max_len 5 --n_sent 99 --seed 0 --eps_list 0.02,0.03 --save_json "$T/results/deept_small6_eval_alpha5_seed0.json" > "$S/deept_eval_alpha5_small6_seed0.log" 2>&1 ;;
    alldev)  learn_eval sst_bert_small_3 small3_alldev 10 97 99 300 8 ;;
    small6)  learn_eval sst_bert_small_6 small6 8 60 3 120 4 ;;
    small12) learn_eval sst_bert_small_12 small12 7 80 3 120 4 ;;
    small12b) G="$T/gauges/deept_small12_seed0.pt"; rm -f "$G"
      $RUN "$PY" -u deept_gauge.py learn --name sst_bert_small_12 --split dev --max_len 6 --n_sent 99 --pos_per_sent 5 --steps 120 --accum 4 --lr 0.01 --seed 0 --log_every 10 --out "$G" > "$S/deept_learn_small12_len6.log" 2>&1
      [ -f "$G" ] || $RUN "$PY" -u deept_gauge.py learn --name sst_bert_small_12 --split dev --max_len 5 --n_sent 99 --pos_per_sent 5 --steps 120 --accum 4 --lr 0.01 --seed 0 --log_every 10 --out "$G" > "$S/deept_learn_small12_len5.log" 2>&1
      [ -f "$G" ] && $RUN "$PY" -u deept_gauge.py eval --name sst_bert_small_12 --gauge "$G" --max_len 12 --n_sent 40 --seed 0 --eps_list 0.005,0.01,0.02 --hi 0.1 --iters 10 --save_json "$T/results/deept_small12_eval_short_seed0.json" > "$S/deept_eval_short_small12.log" 2>&1 ;;
    small12b) learn_eval sst_bert_small_12 small12 6 99 5 120 4 ;;
    long0|long1) s=${what#long}; $RUN "$PY" -u deept_gauge.py eval --name sst_bert_small_3 --gauge "$T/gauges/deept_small3_seed$s.pt" --max_len 32 --n_sent 12 --seed 7 --eps_list 0.005,0.01,0.02 --hi 0.1 --iters 8 --save_json "$T/results/deept_small3_eval_long_seed$s.json" > "$S/deept_eval_long_seed$s.log" 2>&1 ;;
  esac
  echo "DONE_deept_$what $(date)" >> "$S/official_sequence.log"
done
