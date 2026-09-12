#!/bin/bash
#SBATCH -p ckpt-all -A ckpt-amath --qos=ckpt-gpu --gres=gpu:a100:1 -c 4 --mem=40G --time=8:00:00 --requeue
#SBATCH -o logs/unf_%j.out
# usage: sbatch unf.sbatch <tag> <extra args...>   (per-instance resume via --save_json; safe under ckpt pre-emption)
set -u
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; T="$REPO/NNs/transformer_rewrite"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"; TAG=$1; shift
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "START unf_$TAG $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"
"$PY" -u deept_unfolded.py --name sst_bert_small_6 --gauge gauges/deept_small6_seed0.pt --save_json "results/deept_small6_unf_$TAG.json" "$@" >> "$S/deept_unf_$TAG.log" 2>&1; RC=$?
echo "DONE rc=$RC unf_$TAG $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; exit $RC
