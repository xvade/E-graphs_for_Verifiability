#!/bin/bash
# FULL-OP depth-3 generation with substitution-dedup ACTIVE (the canonical run we
# have never actually completed): GEN_COMMUTE off, RELAX_SUBST OFF (so the O(N^2)
# same_via_subst renaming-dedup runs), keeping subgraph/supergraph/varorder relaxed.
# vs run_rule_gen_fullop.sh this drops ONLY `RELAX_SUBST=1` from the generator env.
# Purpose: measure whether the quadratic subst-dedup is affordable at full-op depth-3
# (output is strictly smaller than the 976 MB RELAX_SUBST=1 baseline; wall is the unknown).
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
OUT="$REPO/NNs/reassoc_results"
GEN="$REPO/taso/src/generator"
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH="$OUT:$REPO/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/z3pkg:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages"
ts(){ date +%H:%M:%S; }

echo "[$(ts)] [0/3] protobuf bindings"
protoc -I "$REPO/taso/src/core" --python_out="$OUT" "$REPO/taso/src/core/rules.proto"

echo "[$(ts)] [1/3] generator (FULL op set, depth 3, subgraph/supergraph/varorder relaxed; SUBST dedup ON, no commute)"
cd "$GEN"
protoc -I ../core --cpp_out=. ../core/rules.proto
g++ generator.cc rules.pb.cc -o generator_fullop_subst -I ../../include -I/opt/conda/include \
    -L/opt/conda/lib -lprotobuf -std=c++11 -pthread -O2      # NOTE: no -DPWL_FOCUS
# RELAX_SUBST intentionally NOT exported (dedup active). GEN_COMMUTE not set.
RELAX_SUBGRAPH=1 RELAX_SUPERGRAPH=1 RELAX_VARORDER=1 ./generator_fullop_subst | tail -3
# shard-aware copy: prefer a single graph_subst.pb, else the shard set
if [ -f graph_subst.pb ]; then cp graph_subst.pb "$OUT/fullop_subst_d3_graph_subst.pb"
else cp graph_subst_*.pb "$OUT/"; fi
echo "[$(ts)] pb size: $(du -ch "$OUT"/fullop_subst_d3_graph_subst.pb 2>/dev/null | tail -1 | cut -f1)"

IN="$OUT/fullop_subst_d3_graph_subst.pb"; [ -f "$IN" ] || IN="$GEN"   # dir = shard glob for pb2egg
echo "[$(ts)] [2/3] pb2egg (+ multi-output save)"
python3 "$REPO/NNs/pb2egg.py" "$IN" "$OUT/fullop_subst_d3_egg.txt" \
    --multi-out "$OUT/fullop_subst_d3_egg.multi.pb" | sed 's/^/    /'

echo "[$(ts)] [3/3] pre-dedup (alpha-equivalence)"
python3 "$REPO/NNs/prededup.py" "$OUT/fullop_subst_d3_egg.txt" "$OUT/fullop_subst_d3_dedup.txt" | sed 's/^/    /'

echo "[$(ts)] DONE (through dedup). Op histogram of dedup set:"
grep -oE '\((conv2d|matmul|poolmax|poolavg|concat[0-9]?|transpose|ewmax|ewmin|ewadd|ewsub|ewmul|relu|smul|enlarge) ' \
    "$OUT/fullop_subst_d3_dedup.txt" | tr -d '( ' | sort | uniq -c | sort -rn
wc -l "$OUT/fullop_subst_d3_egg.txt" "$OUT/fullop_subst_d3_dedup.txt"
echo "[$(ts)] NEXT: verify (conv->-m verify, pwl->z3) + prune -> core"
