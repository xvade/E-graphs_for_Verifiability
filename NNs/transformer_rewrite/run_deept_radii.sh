#!/bin/bash
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$REPO/NNs/transformer_rewrite"
"$PY" -u deept_gauge.py radii --name "${1:-sst_bert_small_3}" --max_len "${2:-12}" --n_sent "${3:-20}" --alpha "${4:-1}" --save_json "$REPO/NNs/transformer_rewrite/results/${1:-sst_bert_small_3}_stock_radii_len${2:-12}.json"
