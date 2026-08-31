#!/bin/bash
# Full rule-generation pipeline (depth-3, all-relaxed PWL family), one command.
#
#   [1] generator (PWL focus, depth 3, all 4 quotient relaxations)
#   [2] pb2egg           -> egg rules  +  multi-output rules SAVED to a .multi.pb sidecar
#   [3] pre-dedup        -> alpha-equivalence collapse
#   [4] Z3-verify        -> relaxed_d3_verified.txt   <-- PRE-PRUNE artifact, retained for
#                                                          future learned pruning
#   [5] redundancy-prune -> relaxed_d3_core.txt       <-- minimal-complete core
#
# EVERY stage output is durable (nothing is an unsaved temp). The Z3-verified set is the
# full sound rule set BEFORE any pruning -- that is the "all rules pre-prune" snapshot.
#
# Run inside the container on a compute node, e.g.:
#   sbatch -A cpu-g2-amath -p cpu-g2 -c 8 --mem=200G -t 8:00:00 \
#     --wrap 'apptainer exec "<repo>/tensat.sif" bash "<repo>/NNs/run_rule_gen.sh"'
set -euo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
OUT="$REPO/NNs/reassoc_results"
GEN="$REPO/taso/src/generator"
TENSAT="$REPO/tensat/target/debug/tensat"
export PATH=/opt/conda/bin:$PATH
export LD_LIBRARY_PATH="$REPO/taso/build:/opt/conda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$OUT:$REPO/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages"
ts(){ date +%H:%M:%S; }

echo "[$(ts)] [0/5] regenerate protobuf python binding"
protoc -I "$REPO/taso/src/core" --python_out="$OUT" "$REPO/taso/src/core/rules.proto"

echo "[$(ts)] [1/5] generator (PWL focus, depth 3, RELAX_{SUBGRAPH,SUPERGRAPH,VARORDER,SUBST})"
cd "$GEN"
protoc -I ../core --cpp_out=. ../core/rules.proto
g++ generator.cc rules.pb.cc -o generator_pwl -I ../../include -I/opt/conda/include \
    -L/opt/conda/lib -lprotobuf -std=c++11 -pthread -O2 -DPWL_FOCUS
RELAX_SUBGRAPH=1 RELAX_SUPERGRAPH=1 RELAX_VARORDER=1 RELAX_SUBST=1 ./generator_pwl | tail -2
cp graph_subst.pb "$OUT/relaxed_d3_graph_subst.pb"

echo "[$(ts)] [2/5] pb2egg (+ multi-output save)"
python3 "$REPO/NNs/pb2egg.py" "$OUT/relaxed_d3_graph_subst.pb" "$OUT/relaxed_d3_egg.txt" \
    --multi-out "$OUT/relaxed_d3_egg.multi.pb" | sed 's/^/    /'

echo "[$(ts)] [3/5] pre-dedup (alpha-equivalence)"
python3 "$REPO/NNs/prededup.py" "$OUT/relaxed_d3_egg.txt" "$OUT/relaxed_d3_dedup.txt" | sed 's/^/    /'

echo "[$(ts)] [4/5] Z3 verify -> relaxed_d3_verified.txt (PRE-PRUNE, retained)"
python3 "$REPO/NNs/z3_verify_egg.py" "$OUT/relaxed_d3_dedup.txt" "$OUT/relaxed_d3_verified.txt" \
    --timeout_ms 10000 | sed 's/^/    /'

echo "[$(ts)] [5/5] redundancy prune -> relaxed_d3_core.txt (minimal core)"
"$TENSAT" -m redundancy -r "$OUT/relaxed_d3_verified.txt" -o "$OUT/relaxed_d3_core.txt" \
    --redundancy_iters 4 --n_nodes 8000 --n_sec 4 | tail -5

echo "[$(ts)] DONE. Artifact line counts:"
wc -l "$OUT/relaxed_d3_egg.txt" "$OUT/relaxed_d3_dedup.txt" \
      "$OUT/relaxed_d3_verified.txt" "$OUT/relaxed_d3_core.txt"
ls -la "$OUT/relaxed_d3_egg.multi.pb" "$OUT/relaxed_d3_graph_subst.pb"
