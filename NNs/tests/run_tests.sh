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

echo "== Test 4: axiom verifier (rules()) -- SOUND (rejects false rules) and LIVE (proves) =="
# Guards the -m verify axiom set. This is the permanent guard for the exact class of bug
# that dead-code'd rules() for ~6 years: (a) construction must not panic -> every axiom
# parses under the CURRENT Mdl arities (stale arities panic here); (b) NO axiom may prove a
# known-FALSE rewrite -> catches any future unsound axiom edit; (c) the migrated set still
# proves the min/max family. See NNs/reassoc_results/verify_canaries_false.txt.
CAN=$("$TENSAT" -m verify -r "$R/verify_canaries_false.txt" 2>&1)
canpanic=$(echo "$CAN" | grep -c "panic")
canrej=$(echo "$CAN"   | grep -c "Couldn't prove 5 rule(s)")
assert_eq "$canpanic" "0" "rules() constructs without panic (all axioms parse at current arity)"
assert_ge "$canrej"   "1" "all 5 negative-canary (false) rules are REJECTED (soundness)"
MM=$("$TENSAT" -m verify -r "$R/minmax_verify_test.txt" 2>&1 | grep -c "Proved 8 on this trip")
assert_ge "$MM" "1" "min/max axioms still prove all 8 representative rules"

echo "== Test 5: prededup -- alpha-equivalence collapses copies, keeps comm vs identity =="
# Pins the load-bearing invariant of prededup.canon: two input-renamed copies of the
# SAME rewrite collapse to one, but a commutativity swap and its identity do NOT
# (they canonicalize differently). Regression guard: if canon ever became AC-aware it
# would wrongly drop the comm rules the lattice needs.
cat > "$TMP/dd_in.txt" <<'EOF'
(ewmax ?a ?b)=>(ewmax ?b ?a)
(ewmax ?x ?y)=>(ewmax ?y ?x)
(ewmax ?a ?b)=>(ewmax ?a ?b)
(ewadd ?p ?q)=>(ewadd ?q ?p)
EOF
$PY "$REPO/NNs/prededup.py" "$TMP/dd_in.txt" "$TMP/dd_out.txt" >/dev/null 2>&1
ddn=$(grep -c "=>" "$TMP/dd_out.txt")
ddcomm=$(grep -c '(ewmax ?input_0 ?input_1)=>(ewmax ?input_1 ?input_0)' "$TMP/dd_out.txt")
ddident=$(grep -c '(ewmax ?input_0 ?input_1)=>(ewmax ?input_0 ?input_1)' "$TMP/dd_out.txt")
assert_eq "$ddn"     "3" "4 rules (2 alpha-equiv comm copies) -> 3 unique"
assert_eq "$ddcomm"  "1" "the ewmax commutativity rule survives dedup"
assert_eq "$ddident" "1" "the ewmax identity rule survives (NOT collapsed into comm)"

echo "== Test 6: sexpr_to_functional -- functional round-trip + bare-atom drop =="
# Pins the s-expr -> tensat `-m verify` functional form and the bare-atom-side guard
# (a bare-atom RHS would merge with the next rule under the whitespace-free grammar).
cat > "$TMP/sx_in.txt" <<'EOF'
(ewmax ?input_0 ?input_1)=>(ewmax ?input_1 ?input_0)
(ewadd ?input_0 ?input_1)=>?input_0
EOF
sxout=$($PY "$REPO/NNs/sexpr_to_functional.py" "$TMP/sx_in.txt" "$TMP/sx_out.txt" 2>&1)
sxfun=$(grep -c 'ewmax(input_0,input_1)==ewmax(input_1,input_0)' "$TMP/sx_out.txt")
sxdrop=$(echo "$sxout" | grep -oE "dropped [0-9]+" | grep -oE "[0-9]+")
assert_eq "$sxfun"  "1" "ewmax comm rule converts to functional form"
assert_eq "${sxdrop:-X}" "1" "bare-atom-RHS rule is dropped (1)"

echo "== Test 7: structural_signature -- parse + depth + concat axis =="
# Pins the .model parser and the concat-axis feature (axis 0 vs >0 is the project's
# single most verifiability-relevant structural signal, BUGS.md #11/#12).
# Fixture: Input(1) Weight(2) -> Conv(3) -> Concat(4, axis=1). No blank separators
# (4 lines/node); empty params line for leaves.
printf '1\n0\n0:0\n\n2\n1\n0:0\n\n3\n3\n1:0,2:0\n\n4\n21\n3:0\n1\n' > "$TMP/ss.model"
ssjson=$($PY "$REPO/NNs/structural_signature.py" "$TMP/ss.model" 2>/dev/null)
ssnodes=$($PY -c "import json,sys; print(json.loads(sys.argv[1])['node_count'])" "$ssjson" 2>/dev/null)
ssdepth=$($PY -c "import json,sys; print(json.loads(sys.argv[1])['max_depth'])" "$ssjson" 2>/dev/null)
ssaxis=$($PY -c "import json,sys; a=json.loads(sys.argv[1])['concat_split_axes']; print(a[0]['axis'] if a else 'NONE')" "$ssjson" 2>/dev/null)
assert_eq "${ssnodes:-X}" "4" "structural_signature parses 4 nodes"
assert_eq "${ssdepth:-X}" "2" "longest dependency chain (Concat<-Conv<-leaf) = depth 2"
assert_eq "${ssaxis:-X}"  "1" "Concat axis read from params (=1)"

echo "== Test 8: redundancy mode -- grounds PWL rules and prunes a derivable one =="
# Exercises tensat -m redundancy on the prebuilt binary. Two rules where the
# second is a renamed duplicate of the first (add-commutativity): the pruner must
# recognize both as groundable (elementwise/PWL) and drop the derivable copy.
printf '(ewadd ?a ?b)=>(ewadd ?b ?a)\n(ewadd ?x ?y)=>(ewadd ?y ?x)\n' > "$TMP/red_in.txt"
RED=$("$TENSAT" -m redundancy -r "$TMP/red_in.txt" -o "$TMP/red_out.txt" \
      --redundancy_iters 3 --n_nodes 3000 --n_sec 3 2>&1)
redground=$(echo "$RED" | grep -oE "[0-9]+ groundable" | grep -oE "^[0-9]+")
redpruned=$(echo "$RED" | grep -oE "pruned [0-9]+" | grep -oE "[0-9]+")
redkept=$(grep -c "=>" "$TMP/red_out.txt")
assert_eq "${redground:-X}" "2" "both PWL rules recognized as groundable"
assert_ge "${redpruned:-0}" "1" "the renamed-duplicate rule is pruned as redundant"
assert_eq "$redkept" "1" "exactly one representative kept"

echo "== Test 9: transpose emission -- GATED by default (apply-unsafe), PROBLEMATIC.md #8 =="
# transpose parses + Z3-verifies but tensat can't apply it (rewrites.rs todo!()), so it is
# apply-UNsafe and pb2egg drops it by default. --emit-unapplicable keeps it (Z3 studies).
FIX="$REPO/NNs/tests/transpose_fixture.pb"
DSTATS=$($PY "$REPO/NNs/pb2egg.py" "$FIX" "$TMP/tp_def.txt" 2>&1)
tpdef=$(grep -c "=>" "$TMP/tp_def.txt" 2>/dev/null)
tpskip=$(echo "$DSTATS" | grep -oE "unapplicable ops\): [0-9]+" | grep -oE "[0-9]+$")
assert_eq "${tpdef:-X}"  "0"  "transpose rules gated by default (0 emitted)"
assert_eq "${tpskip:-X}" "20" "20 transpose rules counted tensat-unapplicable"
TEGG="$TMP/tp_egg.txt"
TPSTATS=$($PY "$REPO/NNs/pb2egg.py" "$FIX" "$TEGG" --emit-unapplicable 2>&1)
tpnonclean=$(echo "$TPSTATS" | grep -oE "non-clean ops\):[ ]*[0-9]+" | grep -oE "[0-9]+$")
tpemit=$(grep -c "=>" "$TEGG"); tptrans=$(grep -c "(transpose " "$TEGG")
tppc=$("$TENSAT" -m parse_check -r "$TEGG" 2>&1 | grep -oE "[0-9]+ FAIL" | grep -oE "^[0-9]+")
assert_eq "$tpemit"        "20" "--emit-unapplicable emits all 20 transpose rules"
assert_eq "${tpnonclean:-X}" "0" "0 non-clean drops (transpose is a clean op)"
assert_eq "${tppc:-X}"     "0"  "all emitted transpose rules parse in tensat"

echo "== Test 10: transpose perm decode round-trip (PROBLEMATIC.md #6) =="
# fb0b3db fixed a Release-build uninitialized-read that corrupted Transpose perm to
# garbage ([0,0]). Pins pb2egg._decode_perm against a hardcoded idx oracle + a taso
# core round-trip. See NNs/tests/test_transpose_perm.py.
tpp=$($PY "$REPO/NNs/tests/test_transpose_perm.py" 2>&1 | grep -oE "TRANSPOSE PERM TEST: [A-Z]+" | grep -oE "[A-Z]+$")
assert_eq "${tpp:-X}" "PASS" "transpose perm decode oracle + taso round-trip"

echo "== Test 11: const_* emission -- GATED by default (apply-unsafe), PROBLEMATIC.md #8 =="
# Cpool/Iconv/Imatmul/Iewmul parse + Z3-verify but tensat can't apply them -> gated by default.
CFIX="$REPO/NNs/tests/const_fixture.pb"
CDSTATS=$($PY "$REPO/NNs/pb2egg.py" "$CFIX" "$TMP/const_def.txt" 2>&1)
cdef=$(grep -c "=>" "$TMP/const_def.txt" 2>/dev/null)
cskip=$(echo "$CDSTATS" | grep -oE "unapplicable ops\): [0-9]+" | grep -oE "[0-9]+$")
assert_eq "${cdef:-X}"  "18" "Iewmul/Imatmul/Iconv rules now emit by default (applicable)"
assert_eq "${cskip:-X}" "6"  "6 Cpool-only rules still tensat-unapplicable (gated)"
CEGG="$TMP/const_egg.txt"
CSTATS=$($PY "$REPO/NNs/pb2egg.py" "$CFIX" "$CEGG" --emit-unapplicable 2>&1)
cnonclean=$(echo "$CSTATS" | grep -oE "non-clean ops\):[ ]*[0-9]+" | grep -oE "[0-9]+$")
cemit=$(grep -c "=>" "$CEGG"); ctypes=$(grep -oE "\((Cpool|Iconv|Imatmul|Iewmul)" "$CEGG" | sort -u | wc -l)
cpc=$("$TENSAT" -m parse_check -r "$CEGG" 2>&1 | grep -oE "[0-9]+ FAIL" | grep -oE "^[0-9]+")
assert_eq "$cemit"        "24" "--emit-unapplicable emits all 24 const rules"
assert_eq "${cnonclean:-X}" "0" "0 non-clean drops (const_* are clean ops)"
assert_ge "$ctypes"       "4"  "all 4 const ops (Cpool/Iconv/Imatmul/Iewmul) emitted"
assert_eq "${cpc:-X}"     "0"  "all emitted const rules parse in tensat"

echo "== Test 12: apply-smoke -- emitted (default) ops don't panic tensat; gated ops do =="
# The gate the emission/parse_check/Z3 tests can't give: run a guaranteed-fire rule through a
# 2-iteration saturation on mnist_tiny_mlp and check for the rewrites.rs todo!() panic. This is
# what would have caught transpose/const shipping as apply-panicking rules.
M="$REPO/NNs/mnist_tiny_mlp.taso"
apply_probe() {  # $1 rule ; $2 label ; $3 expect: "ok" (no panic) | "panic"
  printf '%s\n' "$1" > "$TMP/asmoke.txt"
  local out; out=$("$TENSAT" -r "$TMP/asmoke.txt" -s none --model_file "$M" \
      --n_iter 2 --n_sec 8 --no_cycle --no_runtime_report 2>&1)
  local panicked=no; echo "$out" | grep -qaE "todo|not yet implemented|panicked" && panicked=yes
  if [ "$3" = "ok" ]; then
    [ "$panicked" = "no" ] && ok "$2 applies without panic (apply-safe)" || bad "$2 must not panic"
  else
    [ "$panicked" = "yes" ] && ok "$2 panics as expected (gated -- gating is load-bearing)" || bad "$2 expected to panic"
  fi
}
apply_probe '(relu ?input_1)=>(ewadd (relu ?input_1) (relu ?input_1))'          "ewadd"  ok
apply_probe '(relu ?input_1)=>(matmul 0 (relu ?input_1) (relu ?input_1))'       "matmul" ok
apply_probe '(relu ?input_1)=>(ewmax (relu ?input_1) (relu ?input_1))'          "ewmax"  ok
apply_probe '(relu ?input_1)=>(ewmul (relu ?input_1) (relu ?input_1))'          "ewmul"  ok
apply_probe '(relu ?input_1)=>(transpose (transpose (relu ?input_1) 1_0 0) 1_0 0)' "transpose (gated)" panic
apply_probe '(relu ?input_1)=>(ewmul (relu ?input_1) (Iewmul))'                 "const Iewmul (applicable)" ok
apply_probe '(relu ?input_1)=>(matmul 0 (relu ?input_1) (Imatmul))'             "const Imatmul (applicable)" ok
apply_probe '(relu ?input_1)=>(conv2d 1 1 0 0 (relu ?input_1) (Iconv 3 3))'     "const Iconv (applicable)" ok
apply_probe '(relu ?input_1)=>(ewmul (relu ?input_1) (Cpool 3 3))'              "const Cpool (still gated)" panic

rm -rf "$TMP"
echo "======================================"
echo "TESTS: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && echo "ALL_TESTS_PASSED" || { echo "SOME_TESTS_FAILED"; exit 1; }
