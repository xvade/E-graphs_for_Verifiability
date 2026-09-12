#!/bin/bash
# per-class learner on sst_bert_big_3: usage sbatch learn_lab.sbatch <label>
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"; L="$1"
mark() { echo "$1 deept_big3_lab${L}_learn $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
mark START
"$PY" -u deept_gauge.py learn --name sst_bert_big_3 --split dev --max_len 8 --n_sent 40 --pos_per_sent 3 --steps 120 --accum 4 --lr 0.01 --cond_pen 0.0001 --clip 1.0 --hi 0.1 --radius_iters 8 --n_eval 48 --seed 0 --log_every 10 --label $L --out "$T/gauges/deept_big3_lab${L}_seed0.pt" > "$S/deept_learn_big3_lab${L}.log" 2>&1; RC=$?
mark "DONE rc=$RC"; exit $RC
