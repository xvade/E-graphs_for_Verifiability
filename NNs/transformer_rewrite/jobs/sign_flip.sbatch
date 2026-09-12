#!/bin/bash
#SBATCH -p gpu-l40s -A gpu-l40s-amath --gres=gpu:1 -c 4 --mem=32G -t 3:00:00 -J sign_flip
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "START sign_flip $(date) job=$SLURM_JOB_ID" >> "$S/official_sequence.log"
"$PY" -u diagnostics/_sign_flip_test.py sst_bert_small_6 5 10 0.02 >> "$S/sign_flip_small6.log" 2>&1; RC=$?
echo "DONE rc=$RC sign_flip $(date) job=$SLURM_JOB_ID" >> "$S/official_sequence.log"; exit $RC
