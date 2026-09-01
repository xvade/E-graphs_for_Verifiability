#!/bin/bash
# CORRECTED commute regeneration (2026-09-01). Fixes the two faults that broke commute_d3:
#   1. RELAX_SUBST is now UNSET (substitution dedup ACTIVE). GEN_COMMUTE + subst-off keeps the
#      ewmax/ewmin commutativity representatives while the dedup collapses the copy-flood ->
#      no 2 GB protobuf explosion. (Proven at depth 2: GEN_COMMUTE_SUBST_PROBE.md.)
#   2. Toolchain paths fixed for this environment: miniconda3 protoc/g++/libs, taso_py python,
#      z3pkg on PYTHONPATH, LD_LIBRARY_PATH for libprotobuf + libtaso.
# PWL focus, depth 3, GEN_COMMUTE + RELAX_SUBGRAPH/SUPERGRAPH/VARORDER (NO RELAX_SUBST).
# Overwrites the 0-byte failed commute_d3_* outputs. Log: commute_d3_rerun.log.
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
OUT="$REPO/NNs/reassoc_results"
GEN="$REPO/taso/src/generator"
CONDA=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/miniconda3
TASOPY="$CONDA/envs/taso_py/bin/python3"
Z3PKG=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/z3pkg
TENSAT="$REPO/tensat/target/debug/tensat"
P=commute_d3
export PATH="$CONDA/bin:$PATH"
export LD_LIBRARY_PATH="$REPO/taso/build:$CONDA/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$OUT:$REPO/NNs:$Z3PKG"
ts(){ date '+%H:%M:%S'; }

echo "[$(ts)] [0/5] protobuf python binding"
protoc -I "$REPO/taso/src/core" --python_out="$OUT" "$REPO/taso/src/core/rules.proto"

echo "[$(ts)] [1/5] generator (PWL, depth 3, GEN_COMMUTE + subgraph/supergraph/varorder; RELAX_SUBST OFF)"
cd "$GEN"
protoc -I ../core --cpp_out=. ../core/rules.proto
g++ generator.cc rules.pb.cc -o generator_commute_fixed -I ../../include -I"$CONDA/include" \
    -L"$CONDA/lib" -lprotobuf -std=c++11 -pthread -O2 -DPWL_FOCUS
# NOTE: RELAX_SUBST intentionally NOT exported.
RELAX_SUBGRAPH=1 RELAX_SUPERGRAPH=1 RELAX_VARORDER=1 GEN_COMMUTE=1 ./generator_commute_fixed | tail -2
cp graph_subst.pb "$OUT/${P}_graph_subst.pb"
echo "[$(ts)] pb size: $(du -h "$OUT/${P}_graph_subst.pb" | cut -f1)"

echo "[$(ts)] [2/5] pb2egg (+ multi-output save)"
$TASOPY "$REPO/NNs/pb2egg.py" "$OUT/${P}_graph_subst.pb" "$OUT/${P}_egg.txt" \
    --multi-out "$OUT/${P}_egg.multi.pb" | sed 's/^/    /'

echo "[$(ts)] [3/5] pre-dedup (alpha-equivalence)"
$TASOPY "$REPO/NNs/prededup.py" "$OUT/${P}_egg.txt" "$OUT/${P}_dedup.txt" | sed 's/^/    /'

echo "[$(ts)] [4/5] Z3 verify -> ${P}_verified.txt (PRE-PRUNE, retained)"
$TASOPY "$REPO/NNs/z3_verify_egg.py" "$OUT/${P}_dedup.txt" "$OUT/${P}_verified.txt" \
    --timeout_ms 10000 | sed 's/^/    /'

echo "[$(ts)] [5/5] redundancy prune -> ${P}_core.txt"
"$TENSAT" -m redundancy -r "$OUT/${P}_verified.txt" -o "$OUT/${P}_core.txt" \
    --redundancy_iters 4 --n_nodes 8000 --n_sec 4 | tail -5

echo "[$(ts)] DONE. Counts + commutativity check:"
wc -l "$OUT/${P}_egg.txt" "$OUT/${P}_dedup.txt" "$OUT/${P}_verified.txt" "$OUT/${P}_core.txt"
echo "bare ewmax/ewmin commutativity rules in core (2-leaf a<->b swaps):"
grep -cE '\(ew(max|min) \?[a-z_0-9]+ \?[a-z_0-9]+\)=>\(ew(max|min) \?[a-z_0-9]+ \?[a-z_0-9]+\)' "$OUT/${P}_core.txt" || true
