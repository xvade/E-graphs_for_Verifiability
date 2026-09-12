#!/bin/bash
#SBATCH -p ckpt-all -A ckpt-amath --qos=ckpt-gpu --gres=gpu:a100:1 -c 4 --mem=40G -t 2:30:00 -J perclass_alpha --requeue
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "START perclass_alpha $(date) job=$SLURM_JOB_ID restart=$SLURM_RESTART_COUNT" >> "$S/official_sequence.log"
[ -f results/sign_flip_small6_perclass.json ] && { echo "already done" >> "$S/official_sequence.log"; exit 0; }
"$PY" -u diagnostics/_sign_flip_test.py sst_bert_small_6 5 5 0.02 identity,learned,learned_lab0,learned_lab1 _perclass >> "$S/sign_flip_small6_perclass.log" 2>&1; RC=$?
echo "DONE rc=$RC perclass_alpha $(date) job=$SLURM_JOB_ID" >> "$S/official_sequence.log"; exit $RC
