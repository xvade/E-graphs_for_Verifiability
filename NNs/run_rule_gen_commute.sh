#!/bin/bash
# Rule generation WITH binary commutativity (GEN_COMMUTE=1): the generator enumerates
# commutative-op operands in both orders (k from 0, k!=j) so max(a,b)==max(b,a) collides and
# the commutativity rule is emitted -- the closure gap the canonical unordered-pair loop left.
# Same pipeline as run_rule_gen.sh (PWL focus, depth 3, all relaxations) but GEN_COMMUTE and a
# commute_d3_ prefix, so it does NOT clobber relaxed_d3_* (enables a with/without comparison).
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
OUT="$REPO/NNs/reassoc_results"
GEN="$REPO/taso/src/generator"
TENSAT="$REPO/tensat/target/debug/tensat"
P=commute_d3
export PATH=/opt/conda/bin:$PATH
export LD_LIBRARY_PATH="$REPO/taso/build:/opt/conda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$OUT:$REPO/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/z3pkg:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages"
ts(){ date +%H:%M:%S; }

echo "[$(ts)] [0/5] protobuf binding"
protoc -I "$REPO/taso/src/core" --python_out="$OUT" "$REPO/taso/src/core/rules.proto"

echo "[$(ts)] [1/5] generator (PWL, depth 3, all relaxations + GEN_COMMUTE)"
cd "$GEN"
protoc -I ../core --cpp_out=. ../core/rules.proto
g++ generator.cc rules.pb.cc -o generator_pwl -I ../../include -I/opt/conda/include \
    -L/opt/conda/lib -lprotobuf -std=c++11 -pthread -O2 -DPWL_FOCUS
RELAX_SUBGRAPH=1 RELAX_SUPERGRAPH=1 RELAX_VARORDER=1 RELAX_SUBST=1 GEN_COMMUTE=1 ./generator_pwl | tail -2
cp graph_subst.pb "$OUT/${P}_graph_subst.pb"

echo "[$(ts)] [2/5] pb2egg (+ multi-output save)"
python3 "$REPO/NNs/pb2egg.py" "$OUT/${P}_graph_subst.pb" "$OUT/${P}_egg.txt" \
    --multi-out "$OUT/${P}_egg.multi.pb" | sed 's/^/    /'

echo "[$(ts)] [3/5] pre-dedup (alpha-equivalence)"
python3 "$REPO/NNs/prededup.py" "$OUT/${P}_egg.txt" "$OUT/${P}_dedup.txt" | sed 's/^/    /'

echo "[$(ts)] [4/5] Z3 verify -> ${P}_verified.txt (PRE-PRUNE, retained)"
python3 "$REPO/NNs/z3_verify_egg.py" "$OUT/${P}_dedup.txt" "$OUT/${P}_verified.txt" \
    --timeout_ms 10000 | sed 's/^/    /'

echo "[$(ts)] [5/5] redundancy prune -> ${P}_core.txt"
"$TENSAT" -m redundancy -r "$OUT/${P}_verified.txt" -o "$OUT/${P}_core.txt" \
    --redundancy_iters 4 --n_nodes 8000 --n_sec 4 | tail -5

echo "[$(ts)] DONE. Counts + commutativity check:"
wc -l "$OUT/${P}_egg.txt" "$OUT/${P}_dedup.txt" "$OUT/${P}_verified.txt" "$OUT/${P}_core.txt"
echo "bare ewmax/ewmin commutativity rules in core (2-leaf a<->b swaps):"
grep -cE '\(ew(max|min) \?[a-z_0-9]+ \?[a-z_0-9]+\)=>\(ew(max|min) \?[a-z_0-9]+ \?[a-z_0-9]+\)' "$OUT/${P}_core.txt" || true
