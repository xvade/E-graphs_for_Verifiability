#!/bin/bash
# Prune one rule file (arg1 in, arg2 out) inside tensat.sif. Diagnostic driver.
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
IN="$1"; OUTF="$2"
apptainer exec "$REPO/tensat.sif" bash -lc "
  cd '$REPO/tensat'
  export LD_LIBRARY_PATH=\$PWD/../taso/build:/opt/conda/lib
  ./target/debug/tensat -m redundancy -r '$IN' -o '$OUTF' \
    --redundancy_iters 4 --n_nodes 8000 --n_sec 4 2>&1 | tail -6
"
echo "core lines: $(wc -l < "$OUTF" 2>/dev/null || echo MISSING)"
