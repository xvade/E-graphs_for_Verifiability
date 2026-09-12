#!/bin/bash
#SBATCH -p ckpt-all -A ckpt-amath --qos=ckpt-gpu --gres=gpu:a100:1 -c 4 --mem=40G -t 2:00:00 -J probe_big3 --requeue
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "START probe_big3 $(date) job=$SLURM_JOB_ID restart=$SLURM_RESTART_COUNT" >> "$S/official_sequence.log"
"$PY" -u diagnostics/_alpha_mem_probe_big3.py sst_bert_big_3 6 7 8 >> "$S/deept_alpha_mem_probe_big3.log" 2>&1; RC=$?
echo "DONE rc=$RC probe_big3 $(date) job=$SLURM_JOB_ID" >> "$S/official_sequence.log"; exit $RC
