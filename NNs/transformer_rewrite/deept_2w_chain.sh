#!/bin/bash
# Two-word perturbation on sst_bert_small_6: learn a gauge on TWO-word boxes (dev sentences <= 8 tokens, 3 position pairs each),
# then paired eval on two-word test boxes (40 test sentences <= 12 tokens, up to 7 pairs each) of THREE weight sets:
# stock, the two-word-trained gauge (tag gauged) and the one-word-trained S6 gauge gauges/deept_small6_seed0.pt (tag gauged2).
# The eval eps grid is 0.5 / 1 / 1.5 x the median stock two-word radius of the tuning boxes (parsed from the learner log).
#   sbatch ... deept_2w_chain.sh
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
NAME=sst_bert_small_6; TAG=small6_2w; G="$T/gauges/deept_${TAG}_seed0.pt"; G1="$T/gauges/deept_small6_seed0.pt"
mark() { echo "$1 twoword_$2 $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mark START learn; "$PY" -u deept_gauge.py learn --name $NAME --k_words 2 --split dev --max_len 8 --n_sent 60 --pos_per_sent 3 --steps 120 --accum 4 --lr 0.01 --seed 0 --log_every 10 --out "$G" > "$S/deept_learn_$TAG.log" 2>&1; mark "DONE rc=$?" learn
[ -f "$G" ] || { echo "no gauge written; abort twoword" >> "$S/official_sequence.log"; exit 1; }
MED=$(grep -o "median [0-9.]*" "$S/deept_learn_$TAG.log" | head -1 | awk '{print $2}')
EPS=$(awk -v m="$MED" 'BEGIN{printf "%.4g,%.4g,%.4g", 0.5*m, m, 1.5*m}'); echo "eps grid from median stock two-word tuning radius $MED: $EPS" >> "$S/official_sequence.log"
mark START eval; "$PY" -u deept_gauge.py eval --name $NAME --k_words 2 --pairs_per_sent 7 --gauge "$G,$G1" --max_len 12 --n_sent 40 --seed 0 --eps_list "$EPS" --hi 0.1 --iters 10 --save_json "$T/results/deept_${TAG}_eval_short_seed0.json" > "$S/deept_eval_short_$TAG.log" 2>&1; mark "DONE rc=$?" eval
