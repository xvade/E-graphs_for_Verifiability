#!/bin/bash
# held-out radius screen on sst_bert_small_6 with the per-class learned gauges (r5_small6 args + lab gauges; relative gauge paths)
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
mark() { echo "$1 formula_small6_lab $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
for L in 0 1; do [ -f "gauges/deept_small6_lab${L}_seed0.pt" ] || { mark "ABORT missing lab$L gauge"; exit 3; }; done
mark START
"$PY" -u gauge_formula.py validate --name sst_bert_small_6 --seed 0 --n_sent 12 --pos 2 --n_sst 48 --l1_steps 400 --l1N 1 --hi 0.1 --radius_names identity,small6,small6_lab0,small6_lab1,cand:svd_jacN_all,cand:l1N_qk+av0 --n_dev 24 --dev_split test --tag _lab --gauges gauges/deept_small6_seed0.pt,gauges/deept_small6_lab0_seed0.pt,gauges/deept_small6_lab1_seed0.pt --out "$T/results/formula_small6_lab.json" > "$S/formula_small6_lab.log" 2>&1; RC=$?
mark "DONE rc=$RC"; exit $RC
