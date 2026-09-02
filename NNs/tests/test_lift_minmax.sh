#!/bin/bash
# Regression test for the automated min/max-gadget lifter (NNs/lift_minmax_gadgets.py).
# Runs on the real VNN-COMP tll net (and its normalized variant) and asserts:
#   - the gadget recognizer finds the 8 banks + 1 routing layer from weights alone,
#   - the emitted ONNX has explicit Min/Max nodes,
#   - the numeric gate PASSES (lifted function == original, < 1e-5).
# Needs onnx/torch/onnxruntime (taso_py); no taso/container required (ONNX->ONNX).
#
#   bash NNs/tests/test_lift_minmax.sh
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
PY=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/miniconda3/envs/taso_py/bin/python3
TLLDIR="$REPO/NNs/candidate_models/exotic2023/tll"
TMP=$(mktemp -d)
PASS=0; FAIL=0
ok(){ echo "  PASS: $1"; PASS=$((PASS+1)); }
bad(){ echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

run_one() {  # $1 = onnx path, $2 = label
  local out="$TMP/$(basename "$1" .onnx)_lifted.onnx"
  local log; log=$("$PY" "$REPO/NNs/lift_minmax_gadgets.py" "$1" "$out" 2>/dev/null)
  local rc=$?
  echo "== $2 =="
  echo "$log" | grep -qE "recognized: 8 gadget banks, 1 routing" \
    && ok "$2: recognized 8 gadget banks + 1 routing layer (from weights)" \
    || bad "$2: gadget recognition wrong"
  echo "$log" | grep -qE "emitted: [1-9][0-9]* Min \+ 15 Max" \
    && ok "$2: emitted explicit Min + 15 Max nodes" \
    || bad "$2: Min/Max emission wrong"
  [ "$rc" -eq 0 ] && echo "$log" | grep -q "PASS (function preserved)" \
    && ok "$2: numeric gate PASSED (function preserved)" \
    || bad "$2: numeric gate FAILED"
}

run_one "$TLLDIR/tllBench_N16_instance_1_0.onnx" "tll instance 1 (raw MatMul+Add)"
[ -f "$TLLDIR/tll_N16_norm.onnx" ] && run_one "$TLLDIR/tll_N16_norm.onnx" "tll normalized (Gemm-fused)"

rm -rf "$TMP"
echo "======================================"
echo "LIFTER TESTS: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && echo "ALL_LIFTER_TESTS_PASSED" || { echo "SOME_FAILED"; exit 1; }
