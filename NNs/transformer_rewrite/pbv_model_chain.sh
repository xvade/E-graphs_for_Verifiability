#!/bin/bash
# Gauge pipeline on one of Huang et al.'s own (retrained) Shi-style models, e.g. model_sst_3 (symlinked into the DeepT model root):
#   sbatch ... pbv_model_chain.sh <name> <tag> [max_len n_sent pos_per_sent steps accum]
# Steps (each START/DONE-marked in official_sequence.log, logs in NNs/vit_rewrite/_scratch/pbvg_*.log):
#   attrib -> learn (dev boxes) -> paired auto_LiRPA eval (test) -> export stock+gauged Shi-layout ckpts -> their verifier
#   (origin, originPlus, bilinear; l_inf, 20 sentences, 10 bisection steps, <= 32 tokens = their protocol) on stock and gauged.
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; PBV="$REPO/deept_benchmarks/PBVerification"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
NAME=$1; TAG=$2; ML=${3:-10}; NS=${4:-40}; PPS=${5:-3}; STEPS=${6:-120}; ACC=${7:-4}; G="$T/gauges/${TAG}_seed0.pt"
mark() { echo "$1 pbvg_${TAG}_$2 $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mark START attrib; "$PY" -u deept_gauge.py attrib --name $NAME --max_len 12 --n_sent 12 --pos_per_sent 2 --seed 3 --save_json "$T/results/${TAG}_attrib.json" > "$S/pbvg_${TAG}_attrib.log" 2>&1; mark DONE attrib
mark START learn; "$PY" -u deept_gauge.py learn --name $NAME --split dev --max_len $ML --n_sent $NS --pos_per_sent $PPS --steps $STEPS --accum $ACC --lr 0.01 --seed 0 --log_every 10 --out "$G" > "$S/pbvg_${TAG}_learn.log" 2>&1; mark DONE learn
[ -f "$G" ] || { echo "no gauge written; abort" >> "$S/official_sequence.log"; exit 1; }
mark START eval; "$PY" -u deept_gauge.py eval --name $NAME --gauge "$G" --max_len 12 --n_sent 40 --seed 0 --eps_list 0.01,0.02,0.03 --hi 0.1 --iters 10 --save_json "$T/results/${TAG}_eval_short_seed0.json" > "$S/pbvg_${TAG}_eval.log" 2>&1; mark DONE eval
mark START export
"$PY" export_gauged_ckpt.py --name $NAME --gauge none --out "$REPO/deept_benchmarks/gauged_ckpts/${NAME}_stock" > "$S/pbvg_${TAG}_export.log" 2>&1
"$PY" export_gauged_ckpt.py --name $NAME --gauge "$G" --out "$REPO/deept_benchmarks/gauged_ckpts/${NAME}_gauged" >> "$S/pbvg_${TAG}_export.log" 2>&1; mark DONE export
for VER in origin originPlus bilinear; do for W in stock gauged; do
  mark START "${VER}_${W}"; "$PBV/run_pbv.sh" "$REPO/deept_benchmarks/gauged_ckpts/${NAME}_${W}" $VER 10 20 10 32 "${TAG}_${VER}_${W}" > "$S/pbv_${TAG}_${VER}_${W}.out" 2>&1; mark DONE "${VER}_${W} rc=$?"
done; done
