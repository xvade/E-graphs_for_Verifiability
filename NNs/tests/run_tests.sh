#!/bin/bash
# Regression/validation tests for the rule-generation pipeline.
# Run inside the container:  apptainer exec tensat.sif bash NNs/tests/run_tests.sh
# Exits nonzero if any test fails. No pytest dependency (plain asserts).
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
R="$REPO/NNs/reassoc_results"
PB="$REPO/taso/graph_subst.pb"            # git-tracked ORIGINAL TASO corpus (full ops)
PY=/opt/conda/bin/python3
TENSAT="$REPO/tensat/target/debug/tensat"
export PYTHONPATH="$R:$REPO/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/pycontainer/lib/python3.14/site-packages"
export PATH=/opt/conda/bin:$PATH
export LD_LIBRARY_PATH="$REPO/taso/build:/opt/conda/lib:${LD_LIBRARY_PATH:-}"
TMP=$(mktemp -d)
PASS=0; FAIL=0
ok(){   echo "  PASS: $1"; PASS=$((PASS+1)); }
bad(){  echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
assert_ge(){ [ "$1" -ge "$2" ] && ok "$3 ($1 >= $2)" || bad "$3 (got $1, want >= $2)"; }
assert_eq(){ [ "$1" = "$2" ] && ok "$3 (=$1)" || bad "$3 (got $1, want $2)"; }

# Regenerate the python protobuf binding to match taso's proto (byte-compat).
protoc -I "$REPO/taso/src/core" --python_out="$R" "$REPO/taso/src/core/rules.proto" 2>/dev/null

EGG="$TMP/egg.txt"
STATS=$($PY "$REPO/NNs/pb2egg.py" "$PB" "$EGG" 2>&1)
nonclean=$(echo "$STATS" | grep -oE "non-clean ops\):[ ]*[0-9]+" | grep -oE "[0-9]+$")
nconv=$(grep -coE "\(conv2d " "$EGG"); nconcat=$(grep -coE "\(concat[0-9]* " "$EGG")
nmatmul=$(grep -coE "\(matmul " "$EGG"); ntot=$(grep -c "=>" "$EGG")

echo "== Test 1: regression -- non-clean ops (conv/pool/concat) are NOT dropped =="
# This is the test that would have caught the pb2egg clean-only bug. On the original
# clean-only converter, conv/concat were 100% dropped (nonclean drop == all of them).
assert_eq "$nonclean" "0" "pb2egg drops zero non-clean ops"
assert_ge "$nconv"    "1" "conv2d rules are emitted (were dropped before)"
assert_ge "$nconcat"  "1" "concat rules are emitted (were dropped before)"

echo "== Test 2: parse-validity -- every emitted rule parses in current tensat =="
# Catches format drift (op arity/child-order) forever.
pc=$("$TENSAT" -m parse_check -r "$EGG" 2>&1 | grep -oE "parse_check: [0-9]+ OK, [0-9]+ FAIL")
nfail=$(echo "$pc" | grep -oE "[0-9]+ FAIL" | grep -oE "^[0-9]+")
assert_eq "${nfail:-X}" "0" "all $ntot emitted rules parse ($pc)"

echo "== Test 3: reproduction (coverage) -- full-op corpus from the original pb =="
# NOTE: exact reproduction of the hand-committed taso_rules.txt is NOT feasible -- the
# git-tracked graph_subst.pb is a different, smaller corpus, and taso_rules.txt is in a
# stale egg format (e.g. 5-param poolmax, 3-arg enlarge). So this pins to the available
# original pb and asserts the full-op FAMILIES are recovered with the expected counts.
assert_eq "$ntot"    "116" "total emitted rules from original pb"
assert_ge "$nconv"   "31"  "conv2d family coverage"
assert_ge "$nconcat" "54"  "concat family coverage"
assert_ge "$nmatmul" "26"  "matmul family coverage"

rm -rf "$TMP"
echo "======================================"
echo "TESTS: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && echo "ALL_TESTS_PASSED" || { echo "SOME_TESTS_FAILED"; exit 1; }
