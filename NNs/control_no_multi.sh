#!/bin/bash
# Control: single-pattern lane ONLY (no -t). Measures the e-graph the stock
# converted.txt rules build on resnet2b WITHOUT the learned multi corpus.
# If final Nodes/Classes == the with-multi run (183/112), the multi lane's
# NET e-graph contribution is zero (its ewadd-AC content is already provided
# by stock single-pattern rules).
set -e
cd "$(dirname "$0")/../tensat"
export CARGO_HOME=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/cargo_container
export TASO_LIB_DIR=$PWD/../taso/build_gpu
export TASO_INCLUDE_DIR=$PWD/../taso/include
export PROTOBUF_LIB_DIR=/opt/conda/lib
export LD_LIBRARY_PATH=$TASO_LIB_DIR:$PROTOBUF_LIB_DIR:$LD_LIBRARY_PATH
./target/debug/tensat \
    -r converted.txt -u -s none \
    --model_file ../NNs/resnet2b.taso \
    --n_iter 15 --n_sec 300 --n_nodes 500000 \
    --no_cycle --no_runtime_report \
    -e greedy -o tmp/control_no_multi_stats.json \
    > ../NNs/matchprobe_logs/control_no_multi.log 2>&1 || echo "(nonzero exit)"
echo "final size (control, no multi lane):"
python3 -c "import json;d=json.load(open('tmp/control_no_multi_stats.json'));print('  nodes',d['nodes'],'classes',d['classes'],'iter',d['iter'],'saturation',round(d['saturation'],2))"
