#!/bin/bash
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; cd "$REPO"; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for m in pgd_2_3_16 ibp_3_3_8; do "$PY" NNs/vit_rewrite/vit_sample_diag.py $m 20 256 2>/dev/null; done
