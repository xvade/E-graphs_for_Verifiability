#!/bin/bash
# User-requested parallelism: run the two secondary controls on CPU now (device: cpu), concurrently with the GPU chain.
# Comparable level: initial CROWN (deterministic, no time cap). alpha-CROWN/BaB on CPU are time-capped -> not comparable to GPU runs.
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
cd "$REPO/alpha-beta-CROWN/complete_verifier"; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for name in base_export idinitG_patched; do
  ( echo "START_${name}_cpu $(date)" >> "$S/official_sequence.log"
    "$PY" -u abcrown.py --config "$REPO/NNs/vit_rewrite/cfg_vit_${name}_cpu.yaml" --device cpu > "$S/official_${name}_cpu.log" 2>&1
    echo "DONE_${name}_cpu $(date)" >> "$S/official_sequence.log" ) &
done
wait
