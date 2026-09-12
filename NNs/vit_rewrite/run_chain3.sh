#!/bin/bash
# Follow-on official runs (after run_chain2's DONE marker, so the GPU stays exclusive): the export-path controls.
#  learnedG_patched = learned gauge written INTO the stock ONNX graph (identical structure)  -> clean headline
#  base_export      = identity weights through vit_export.py's re-export path                -> measures the export confound
#  idinitG_patched  = independently learned gauge (id init, seed 1) in the stock graph      -> replication at the top tier
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
until grep -q "^DONE " "$S/official_sequence.log"; do sleep 60; done
cd "$REPO/alpha-beta-CROWN/complete_verifier"
for name in learnedG_patched base_export idinitG_patched; do
  echo "START_$name $(date)" >> "$S/official_sequence.log"
  "$PY" -u abcrown.py --config "$REPO/NNs/vit_rewrite/cfg_vit_$name.yaml" > "$S/official_$name.log" 2>&1
done
echo "DONE_ALL $(date)" >> "$S/official_sequence.log"
