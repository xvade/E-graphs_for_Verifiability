#!/bin/bash
# Out-of-distribution gauge tuning for sst_bert_small_6 (question: is the gauge a property of the weights, i.e. derivable?):
#   deept_ood_chain.sh yelp    -> gauge tuned on Yelp-review boxes (Yelp true labels, SST model's tokenizer; 'dev' = Yelp train.csv)
#   deept_ood_chain.sh random  -> gauge tuned on random vocabulary sequences (model's own prediction as label; no dataset at all)
# then the standard paired auto_LiRPA eval on the SAME 40 SST test sentences <= 12 tokens (294 positions) as gauges/deept_small6_seed0.pt.
#   sbatch -p gpu-l40s -A gpu-l40s-amath --gres=gpu:l40s:1 -c 5 --mem=25G -t 08:00:00 -J ood_yelp -o "<log>" deept_ood_chain.sh yelp
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
X=$1; NAME=sst_bert_small_6; TAG="small6_ood$X"; G="$T/gauges/deept_${TAG}_seed0.pt"
mark() { echo "$1 ood_${X}_$2 $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mark START learn; "$PY" -u deept_gauge.py learn --name $NAME --data $X --split dev --max_len 8 --n_sent 60 --pos_per_sent 3 --steps 120 --accum 4 --lr 0.01 --seed 0 --log_every 10 --out "$G" > "$S/deept_learn_$TAG.log" 2>&1; mark "DONE rc=$?" learn
[ -f "$G" ] || { echo "no gauge written; abort ood_$X" >> "$S/official_sequence.log"; exit 1; }
mark START eval; "$PY" -u deept_gauge.py eval --name $NAME --data sst --gauge "$G" --max_len 12 --n_sent 40 --seed 0 --eps_list 0.01,0.02,0.03 --hi 0.1 --iters 10 --save_json "$T/results/deept_${TAG}_eval_short_seed0.json" > "$S/deept_eval_short_$TAG.log" 2>&1; mark "DONE rc=$?" eval
