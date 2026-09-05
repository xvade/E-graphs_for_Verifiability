#!/bin/bash
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; cd "$REPO"; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
NEW="60,388,4671,5927,7064,9106,9145,1029,1546,1683,3527,3775,4021,5090,5755,6219,7308,9689"
"$PY" NNs/vit_rewrite/vit_box_discrepancy.py pgd_2_3_16 vit_learnedG_patched 1000 "$NEW" 2>/dev/null
"$PY" NNs/vit_rewrite/vit_box_discrepancy.py pgd_2_3_16 vit_learnedG_patched 200 2>/dev/null
"$PY" NNs/vit_rewrite/vit_box_discrepancy.py pgd_2_3_16 vit_idinitG_patched 200 2>/dev/null
