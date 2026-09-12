#!/bin/bash
# Gauge learned AGAINST Huang et al.'s verifier, then evaluated everywhere.  One step per invocation (so each fits the 9 h
# checkpoint-partition cap; chain with sbatch --dependency=afterok):
#   pbv_learn_chain.sh learn  <deept_name> <tag> <version origin|inner> <max_len> [n_sent pos steps accum]
#       -> gauges/pbvtrained_<tag>_<version>_seed0.pt + exported ckpt deept_benchmarks/gauged_ckpts/<deept_name>_pbv<version>
#   pbv_learn_chain.sh verify <deept_name> <tag> <version-trained> <their-version> <samples> <iters> <maxlen>
#       -> their verifier (<their-version>) on the exported ckpt; results/pbv_<tag>_<their-version>_pbv<version-trained>.json
#   pbv_learn_chain.sh lirpa  <deept_name> <tag> <version-trained>   -> paired auto_LiRPA eval of that gauge (40 test sentences <= 12)
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; PBV="$REPO/deept_benchmarks/PBVerification"; CK="$REPO/deept_benchmarks/gauged_ckpts"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NLTK_DATA="$REPO/deept_benchmarks/nltk_data" TF_CPP_MIN_LOG_LEVEL=2; cd "$T"
MODE=$1; NAME=$2; TAG=$3; VER=$4; shift 4
mark() { echo "$1 pbvl_${TAG}_$2 $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
case $MODE in
  learn)
    ML=$1; NS=${2:-40}; PPS=${3:-3}; STEPS=${4:-120}; ACC=${5:-4}; G="$T/gauges/pbvtrained_${TAG}_${VER}_seed0.pt"
    mark START "learn_$VER"; "$PY" -u pbv_learn.py --ckpt "$CK/${NAME}_stock" --version $VER --split dev --max_len $ML --n_sent $NS --pos_per_sent $PPS --steps $STEPS --accum $ACC --lr 0.01 --seed 0 --out "$G" > "$S/pbvl_${TAG}_learn_$VER.log" 2>&1; mark "DONE rc=$?" "learn_$VER"
    [ -f "$G" ] || { echo "no gauge; abort" >> "$S/official_sequence.log"; exit 1; }
    mark START "export_$VER"; "$PY" export_gauged_ckpt.py --name $NAME --gauge "$G" --out "$CK/${NAME}_pbv$VER" > "$S/pbvl_${TAG}_export_$VER.log" 2>&1; mark "DONE rc=$?" "export_$VER" ;;
  verify)
    TV=$1; NS=$2; IT=$3; ML=$4
    mark START "${TV}_pbv$VER"; "$PBV/run_pbv.sh" "$CK/${NAME}_pbv$VER" $TV 10 $NS $IT $ML "${TAG}_${TV}_pbv$VER" > "$S/pbv_${TAG}_${TV}_pbv$VER.out" 2>&1; mark "DONE rc=$?" "${TV}_pbv$VER" ;;
  lirpa)
    G="$T/gauges/pbvtrained_${TAG}_${VER}_seed0.pt"
    mark START "lirpa_$VER"; "$PY" -u deept_gauge.py eval --name $NAME --gauge "$G" --max_len 12 --n_sent 40 --seed 0 --eps_list 0.01,0.02,0.03 --hi 0.1 --iters 10 --save_json "$T/results/pbvtrained_${TAG}_${VER}_eval_short_seed0.json" > "$S/pbvl_${TAG}_lirpa_$VER.log" 2>&1; mark "DONE rc=$?" "lirpa_$VER" ;;
esac
