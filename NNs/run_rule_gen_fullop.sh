#!/bin/bash
# FULL-OP rule generation (NO -DPWL_FOCUS) so conv/pool/concat rules are produced -- the
# conv-inclusive core the PWL_FOCUS run (relaxed_d3_core) lacks. Stops after dedup: the
# conv-rule VERIFICATION routing (conv-touching -> `-m verify` axioms; pure-PWL -> z3) is a
# separate follow-on, since z3 treats conv as uninterpreted and would reject conv rules.
#
# Gate: full-op all-relaxed depth-3 will be >> the 849,839 PWL transfers. If the generator
# or dedup blows up, fall back to RELAX_SUBST-only (the main lever). Outputs: fullop_d3_*.
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
OUT="$REPO/NNs/reassoc_results"
GEN="$REPO/taso/src/generator"
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH="$OUT:$REPO/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/z3pkg:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages"
ts(){ date +%H:%M:%S; }

echo "[$(ts)] [0/3] protobuf bindings"
protoc -I "$REPO/taso/src/core" --python_out="$OUT" "$REPO/taso/src/core/rules.proto"

echo "[$(ts)] [1/3] generator (FULL op set, depth 3, all 4 relaxations)"
cd "$GEN"
protoc -I ../core --cpp_out=. ../core/rules.proto
g++ generator.cc rules.pb.cc -o generator_fullop -I ../../include -I/opt/conda/include \
    -L/opt/conda/lib -lprotobuf -std=c++11 -pthread -O2      # NOTE: no -DPWL_FOCUS
RELAX_SUBGRAPH=1 RELAX_SUPERGRAPH=1 RELAX_VARORDER=1 RELAX_SUBST=1 ./generator_fullop | tail -2
cp graph_subst.pb "$OUT/fullop_d3_graph_subst.pb"
echo "[$(ts)] pb size: $(du -h "$OUT/fullop_d3_graph_subst.pb" | cut -f1)"

echo "[$(ts)] [2/3] pb2egg (+ multi-output save)"
python3 "$REPO/NNs/pb2egg.py" "$OUT/fullop_d3_graph_subst.pb" "$OUT/fullop_d3_egg.txt" \
    --multi-out "$OUT/fullop_d3_egg.multi.pb" | sed 's/^/    /'

echo "[$(ts)] [3/3] pre-dedup (alpha-equivalence)"
python3 "$REPO/NNs/prededup.py" "$OUT/fullop_d3_egg.txt" "$OUT/fullop_d3_dedup.txt" | sed 's/^/    /'

echo "[$(ts)] DONE (through dedup). Op histogram of dedup set:"
grep -oE '\((conv2d|matmul|poolmax|poolavg|concat[0-9]?|transpose|ewmax|ewmin|ewadd|ewsub|ewmul|relu|smul|enlarge) ' \
    "$OUT/fullop_d3_dedup.txt" | tr -d '( ' | sort | uniq -c | sort -rn
wc -l "$OUT/fullop_d3_egg.txt" "$OUT/fullop_d3_dedup.txt"
echo "[$(ts)] NEXT: verify (conv->-m verify, pwl->z3) + prune -> fullop core"
