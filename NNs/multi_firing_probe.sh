#!/bin/bash
# Firing probe for the CONVERTED learned multi-pattern corpus (pb2multi.py output).
# Runs INSIDE tensat.sif via `apptainer exec --nv` on a GPU node.
#
# Reads the per-rule funnel that run_one() prints (rewrites.rs:1263):
#   DEBUG multi-pattern rule[i] src1=.. .. (pairs,compatible,valid,cycle_ok) this_rule=(a,b,c,d) ..
# and the totals line (rewrites.rs:1270). Semantics of the columns (verified in src):
#   pairs      operand e-class pairs found for (src1,src2)
#   compatible mergeable shared-variable substs
#   valid      both dst patterns pass check_pat (apply-safe / buildable)
#   cycle_ok   ACTUALLY applied (dst built + unioned). == "fired".
# CYCLE FLAG (rewrites.rs:1375, `if self.no_cycle {check} else {true}`):
#   --no_cycle PRESENT  -> cycle filter ACTIVE   -> cycle_ok = valid MINUS cycle-blocked. <-- the answer run.
#   --no_cycle ABSENT   -> filter off            -> cycle_ok == valid (upper bound; tests e-graph growth).
# Learned rules occupy the LEADING pair-indices (main.rs:495 chains PRE_DEFINED_MULTI AFTER),
# so rule[i] with i < (lines/2) is OURS; higher i are the stock predefined multi rules.
#
# Usage: multi_firing_probe.sh <MODEL> <MULTI_FILE> <TAG> [--cycle-off]
#   MODEL      e.g. resnet2b  (reads ../NNs/<MODEL>.taso)
#   MULTI_FILE path to the two-line-per-rule multi corpus (relative to tensat/ or absolute)
#   TAG        label for the log/stats file names
#   --cycle-off  omit --no_cycle (run the upper-bound variant)
set -e
cd "$(dirname "$0")/../tensat"

MODEL="${1:?model}"; MULTI_FILE="${2:?multi_file}"; TAG="${3:?tag}"; MODE="${4:-}"
CYCLE_ARG="--no_cycle"; CYCLE_LABEL="cycle-filter-ON"
if [ "$MODE" = "--cycle-off" ]; then CYCLE_ARG=""; CYCLE_LABEL="cycle-filter-OFF(upper-bound)"; fi

export CARGO_HOME=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/cargo_container
export TASO_LIB_DIR=$PWD/../taso/build_gpu
export TASO_INCLUDE_DIR=$PWD/../taso/include
export PROTOBUF_LIB_DIR=/opt/conda/lib
export LD_LIBRARY_PATH=$TASO_LIB_DIR:$PROTOBUF_LIB_DIR:$LD_LIBRARY_PATH

echo "===== BUILD (debug, ensures current instrumentation) ====="
/opt/cargo/bin/cargo build 2>&1 | tail -4

OUTDIR=../NNs/matchprobe_logs
mkdir -p "$OUTDIR" tmp
NP=$(( $(grep -c '' "$MULTI_FILE" 2>/dev/null || echo 0) ))
LOG="$OUTDIR/firing_${TAG}.log"

echo ""
echo "===== RUN: model=$MODEL  multi=$MULTI_FILE  $CYCLE_LABEL ====="
echo "  learned-rule lines in file: $NP (pairs: $((NP/2)))  [predefined multi appended after these]"
./target/debug/tensat \
    -r converted.txt -t "$MULTI_FILE" -u -s none \
    --model_file ../NNs/${MODEL}.taso \
    --n_iter 15 --iter_multi 15 --n_sec 300 --n_nodes 500000 \
    $CYCLE_ARG --no_runtime_report \
    -e greedy -o tmp/firing_${TAG}_stats.json \
    > "$LOG" 2>&1 || echo "  (tensat exited nonzero — see log)"
echo "  wrote $LOG ($(wc -l < "$LOG") lines)"
echo ""
echo "----- fired rules (cycle_ok>0), OUR indices only (i < $((NP/2))) -----"
grep 'DEBUG multi-pattern rule\[' "$LOG" | \
  awk -v np=$((NP/2)) '
     match($0, /rule\[[0-9]+\]/) { s=substr($0,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); idx=s+0 }
     match($0, /this_rule=\([0-9]+, *[0-9]+, *[0-9]+, *[0-9]+\)/) {
        t=substr($0,RSTART,RLENGTH); gsub(/[^0-9,]/,"",t); split(t,arr,",");
        if (arr[4]+0 > 0 && idx < np)
           print "  rule["idx"] cycle_ok="arr[4]" | pairs="arr[1]" compat="arr[2]" valid="arr[3]
     }' || true
echo "----- totals line -----"
grep 'DEBUG multi-pattern totals' "$LOG" | tail -1 || echo "  (no totals line — multi lane may not have run)"
echo "===== DONE ($TAG) ====="
