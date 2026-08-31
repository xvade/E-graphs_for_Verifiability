#!/bin/bash
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
R="$REPO/NNs/reassoc_results"
cd "$REPO/tensat"
export CARGO_HOME=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/cargo_container
export LD_LIBRARY_PATH="$PWD/../taso/build:/opt/conda/lib:${LD_LIBRARY_PATH:-}"
export CARGO_NET_OFFLINE=true
export PATH=/opt/conda/bin:/opt/cargo/bin:$PATH
export TASO_LIB_DIR=../taso/build TASO_INCLUDE_DIR=../taso/include PROTOBUF_LIB_DIR=/opt/conda/lib

echo "########## BUILD (CARGO_HOME=$CARGO_HOME) ##########"
BIN="$PWD/target/debug/tensat"
rm -f "$BIN"                       # force: only a fresh compile can recreate it
cargo build --offline 2>&1 | tail -20
[ -x "$BIN" ] || { echo "BUILD_FAILED (binary not produced)"; exit 1; }
echo "BUILD_OK ($(stat -c %y "$BIN"))"

echo; echo "########## TEST 1: regression -- min/max axioms still prove (expect Proved 8, no panic) ##########"
"$BIN" -m verify -r "$R/minmax_verify_test.txt" 2>&1 | grep -E "Proved|Couldn't|panic|thread" | head -20

echo; echo "########## TEST 2: positive -- orig 116 current-arity rules (report proved count) ##########"
"$BIN" -m verify -r "$R/orig_full_functional.txt" 2>&1 | grep -E "^Proved|Couldn't prove [0-9]|panic|thread" | head -40

echo; echo "########## TEST 3: NEGATIVE CANARIES -- 5 known-FALSE rules (expect: Couldn't prove 5) ##########"
"$BIN" -m verify -r "$R/verify_canaries_false.txt" 2>&1 | grep -E "Proved|Couldn't|panic|thread|=>" | head -20
echo "########## DONE ##########"
