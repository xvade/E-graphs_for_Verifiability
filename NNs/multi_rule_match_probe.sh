#!/bin/bash
# Instrumented rerun to read off which multi-pattern rules actually MATCH
# (and fire into the e-graph) on each model, vs. never binding at all.
# Runs INSIDE tensat.sif via apptainer exec --nv on a GPU node.
#
# The binary's run_one() hook prints, per saturation iteration:
#   DEBUG multi-pattern canonical[i] = <pat> -> N eclasses matched, M substs
#   DEBUG multi-pattern rule[i] ... (pairs,compatible,valid,cycle_ok) this_rule=..
# which tells us, per rule: candidate operand pairs found, how many were
# shape-compatible, how many passed validity, how many actually got added
# (cycle_ok). A rule with pairs=0 / 0 eclasses matched never bound; a rule
# with cycle_ok>0 fired into the e-graph (so any absence from the extracted
# model is an extraction-selection outcome, not a missing match).
set -e
cd "$(dirname "$0")/../tensat"

export CARGO_HOME=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/cargo_container
export TASO_LIB_DIR=$PWD/../taso/build_gpu
export TASO_INCLUDE_DIR=$PWD/../taso/include
export PROTOBUF_LIB_DIR=/opt/conda/lib
export LD_LIBRARY_PATH=$TASO_LIB_DIR:$PROTOBUF_LIB_DIR:$LD_LIBRARY_PATH

echo "===== BUILD (debug) ====="
/opt/cargo/bin/cargo build 2>&1 | tail -5

OUTDIR=../NNs/matchprobe_logs
mkdir -p "$OUTDIR" tmp

for MODEL in mnist_cnn_a resnet2b inception_mnist; do
  echo ""
  echo "===== RUN: $MODEL ====="
  ./target/debug/tensat \
      -r converted.txt -t converted_multi.txt -u -s none \
      --model_file ../NNs/${MODEL}.taso \
      --n_iter 15 --iter_multi 15 --n_sec 300 --n_nodes 500000 \
      --no_cycle --no_runtime_report \
      -e greedy -o tmp/matchprobe_${MODEL}_stats.json \
      > "$OUTDIR/${MODEL}.log" 2>&1 || echo "  (tensat exited nonzero for $MODEL — see log)"
  echo "  wrote $OUTDIR/${MODEL}.log ($(wc -l < "$OUTDIR/${MODEL}.log") lines)"
done
echo ""
echo "===== DONE ====="
