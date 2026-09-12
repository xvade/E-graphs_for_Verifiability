#!/bin/bash
# Gauge pipeline on further DeepT-release checkpoints (Yelp polarity models; SST width / LayerNorm variants), one GPU, sequential.
#   sbatch -p gpu-l40s -A gpu-l40s-amath --gres=gpu:l40s:1 -c 5 --mem=25G -t 12:00:00 deept_yelp_chain.sh <what>...
#   what in: attrib:<name>  (attention-share attribution, 12 test sentences x 2 positions, eps = 1x / 1.5x stock radius)
#            full:<name>:<tag>[:max_len:n_sent:pos:steps:accum]  (attrib -> learn on 'dev' (Yelp: train.csv) -> paired test eval)
# The eval eps grid is derived from the attribution's stock radii (median r -> 0.5r, r, 1.5r, 3 significant digits) so that the
# fixed-eps verified counts are informative for models whose radius scale differs from SST's.  Logs: _scratch/deept_<tag>_*.log,
# START/DONE markers in official_sequence.log.  Separate from deept_chain.sh so the two session instances never edit one running script.
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
mark() { echo "$1 deept_$2 $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
attrib() {  # name tag
  mark START "${2}_attrib"; "$PY" -u deept_gauge.py attrib --name $1 --max_len 12 --n_sent 12 --pos_per_sent 2 --seed 3 --save_json "$T/results/deept_${2}_attrib.json" > "$S/deept_${2}_attrib.log" 2>&1; mark "DONE rc=$?" "${2}_attrib"
}
eps_from_attrib() {  # tag -> "e1,e2,e3"
  "$PY" - "$T/results/deept_${1}_attrib.json" <<'PYEOF'
import json, sys, numpy as np
r = np.median([row[4] for row in json.load(open(sys.argv[1]))["rows"] if row[3] == 1.0])   # row: sent, pos, len, scale, eps(=scale x stock radius), ...
print(",".join(f"{float(f'{x:.3g}'):g}" for x in (0.5 * r, r, 1.5 * r)))
PYEOF
}
for what in "$@"; do
  IFS=: read -r mode name tag ml ns pps steps acc <<< "$what"
  case $mode in
    attrib) attrib $name ${tag:-$name} ;;
    full)
      ml=${ml:-8}; ns=${ns:-40}; pps=${pps:-3}; steps=${steps:-120}; acc=${acc:-4}; G="$T/gauges/deept_${tag}_seed0.pt"
      [ -f "$T/results/deept_${tag}_attrib.json" ] || attrib $name $tag
      EPS=$(eps_from_attrib $tag); echo "eps grid for $tag: $EPS" >> "$S/official_sequence.log"
      mark START "${tag}_learn"; "$PY" -u deept_gauge.py learn --name $name --split dev --max_len $ml --n_sent $ns --pos_per_sent $pps --steps $steps --accum $acc --lr 0.01 --seed 0 --log_every 10 --out "$G" > "$S/deept_learn_${tag}.log" 2>&1; mark "DONE rc=$?" "${tag}_learn"
      [ -f "$G" ] || { echo "no gauge written for $tag; skipping eval" >> "$S/official_sequence.log"; continue; }
      mark START "${tag}_eval"; "$PY" -u deept_gauge.py eval --name $name --gauge "$G" --max_len 12 --n_sent 40 --seed 0 --eps_list $EPS --hi 0.1 --iters 10 --save_json "$T/results/deept_${tag}_eval_short_seed0.json" > "$S/deept_eval_short_${tag}.log" 2>&1; mark "DONE rc=$?" "${tag}_eval" ;;
  esac
done
mark DONE_ALL "yelp_chain"
