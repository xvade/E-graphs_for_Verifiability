#!/bin/bash
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; cd "$REPO"; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for m in pgd_2_3_16 ibp_3_3_8; do "$PY" NNs/vit_rewrite/vit_sample_diag.py $m 20 256 2>/dev/null; done
