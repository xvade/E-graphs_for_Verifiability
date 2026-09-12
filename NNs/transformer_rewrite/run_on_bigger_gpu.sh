#!/bin/bash
# usage (batch): sbatch -p ckpt-all -A ckpt-amath --qos=ckpt-gpu --gres=gpu:a100:1 -c 5 --mem=40G -t 5:00:00 run_on_bigger_gpu.sh <deept_chain args>
# Guard for chain steps that need > 44 GB (alpha-CROWN on the 6-layer DeepT model): refuse to run on a card with < 60 GB, so a
# 40 GB A100 does not silently repeat the L40S OOM. Then runs deept_chain.sh with the given steps.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; S="$REPO/NNs/vit_rewrite/_scratch"
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ${mem} MiB" | tee -a "$S/official_sequence.log"
if [ -z "$mem" ] || [ "$mem" -lt 60000 ]; then echo "ABORT_bigger_gpu: only ${mem} MiB on this card (job $SLURM_JOB_ID) $(date)" >> "$S/official_sequence.log"; exit 1; fi
exec "$REPO/NNs/transformer_rewrite/deept_chain.sh" "$@"
