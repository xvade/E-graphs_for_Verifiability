#!/bin/bash
# sbatch -p gpu-l40s -A gpu-l40s-amath --gres=gpu:l40s:1 -c 4 --mem=20G -t 1:00:00 diagnostics/run_nan_cliff_probe.sh
# NaN-cliff probe for small_6 (stock + gauged, 40 sampled instances); ~20 min per instance on the login-node CPU, minutes on a GPU.
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; export OMP_NUM_THREADS=4; cd "$REPO/NNs/transformer_rewrite"
"$REPO/alpha-beta-CROWN/.venv/bin/python" -u diagnostics/_nan_cliff_probe.py sst_bert_small_6 results/deept_small6_eval_short_seed0.json gauges/deept_small6_seed0.pt 40 > "$REPO/NNs/vit_rewrite/_scratch/deept_nancliff_small6_gpu.log" 2>&1
