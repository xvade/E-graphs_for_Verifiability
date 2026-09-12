#!/bin/bash
# held-out radius screen with the identity-initialised and the unified-rule-initialised learners, single and per-label
#   sbatch screen_init.sbatch <small6|big3>
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"; M="$1"
case $M in small6) NAME=sst_bert_small_6 ;; big3) NAME=sst_bert_big_3 ;; esac
mark() { echo "$1 formula_${M}_init $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
G="gauges/deept_${M}_seed0.pt"; for t in lab0 lab1 init initlab0 initlab1; do f="gauges/deept_${M}_${t}_seed0.pt"; [ -f "$f" ] || { mark "ABORT missing $f"; exit 3; }; G="$G,$f"; done
mark START
"$PY" -u gauge_formula.py validate --name $NAME --seed 0 --n_sent 12 --pos 2 --n_sst 48 --l1_steps 400 --l1N 1 --hi 0.1 --radius_names identity,${M},${M}_lab0,${M}_lab1,${M}_init,${M}_initlab0,${M}_initlab1,cand:svd_jacN_all,cand:l1N_qk+av0 --n_dev 24 --dev_split test --tag _init --gauges "$G" --out "$T/results/formula_${M}_init.json" > "$S/formula_${M}_init.log" 2>&1; RC=$?
mark "DONE rc=$RC"; exit $RC
