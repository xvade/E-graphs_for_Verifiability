#!/bin/bash
# usage: run_deept_learn.sh <name> <out.pt> [extra args...]
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$REPO/NNs/transformer_rewrite"
name="$1"; out="$2"; shift 2; case "$out" in /*) ;; *) out="$REPO/$out";; esac   # the learner runs with cwd NNs/transformer_rewrite
"$PY" -u deept_gauge.py learn --name "$name" --out "$out" "$@"
