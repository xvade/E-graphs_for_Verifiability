#!/bin/bash
# sbatch -p ckpt-all -A ckpt-amath --qos=ckpt-gpu --gres=gpu:l40s:1 -c 4 --mem=25G -t 0:40:00 diagnostics/run_alpha_mem_probe.sh
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$REPO/NNs/transformer_rewrite"
"$REPO/alpha-beta-CROWN/.venv/bin/python" -u diagnostics/_alpha_mem_probe.py
