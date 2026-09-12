#!/bin/bash
# held-out radius screen on sst_bert_big_3 with the per-class learned gauges (snapshot gauge_formula_lab.py)
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"; G="$T/gauges"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
mark() { echo "$1 formula_big3_lab $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
for L in 0 1; do [ -f "$G/deept_big3_lab${L}_seed0.pt" ] || { mark "ABORT missing lab$L gauge"; exit 3; }; done
mark START
"$PY" -u gauge_formula_lab.py validate --name sst_bert_big_3 --seed 0 --n_sent 12 --pos 2 --n_sst 48 --l1_steps 400 --l1N 1 --hi 0.1 --radius_names identity,big3,big3_lab0,big3_lab1,cand:svd_jacN_all,cand:l1N_qk+av0 --n_dev 24 --dev_split test --tag _lab --gauges $G/deept_big3_seed0.pt,$G/deept_big3_lab0_seed0.pt,$G/deept_big3_lab1_seed0.pt --out "$T/results/formula_big3_lab.json" > "$S/formula_big3_lab.log" 2>&1; RC=$?
mark "DONE rc=$RC"; exit $RC
