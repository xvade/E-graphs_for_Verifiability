# Hand-constructs a variant of resnet2b where two of its Relu ops (guids 128
# and 137 in NNs/resnet2b.taso -- the relus after layer1.0.conv1 and
# layer1.1.conv1, both real shape (1,16,8,8)) are fused into one batched
# relu via concat(axis=0)+relu+split(axis=0), mirroring the multi-pattern
# rewrite tensat's rule[7]/[10] (converted_multi.txt) proved valid on this
# exact graph (thousands of successful cycle_ok applications in an
# instrumented run) but never actually extracts, because relu's own real
# cost scales with element count -- a relu over the concatenated
# (batch=2) tensor costs about as much as the two original relus
# combined, so the extra concat+split overhead can only ever make the
# fused form cost *more* under TASO's real cost model, regardless of how
# heavily Concat/Split are discounted (tried up to 20x in
# tensat/src/optimize.rs's --favor_fusion, still didn't flip it). This
# script demonstrates the equivalence directly instead of coaxing the
# extractor into finding it: relu is a pure elementwise op, so
# concat-then-relu-then-split along any axis is exactly identical to
# relu-ing each part separately, unconditionally (no rewrite-rule
# machinery needed to prove it here -- it's definitionally true).
#
# Otherwise mirrors reconstruct_optimized.py's parse-and-build approach,
# minus the Transpose/weight-shape-matching machinery this simpler,
# already-correctly-biased script doesn't need (this always parses the
# ORIGINAL resnet2b.taso, not a tensat-optimized export).
import sys
import numpy as np
import onnx
from onnx import numpy_helper
import taso as ts

FUSE_GUIDS = (128, 137)

def load_original_weights(onnx_path):
    model = onnx.load(onnx_path)
    by_shape = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init).astype(np.float32)
        by_shape[tuple(arr.shape)] = arr
    return by_shape

def parse_and_build(model_path, original_weights):
    graph = ts.new_graph()
    nodes = {}
    weight_arrays = {}  # guid -> real numpy array, only for Weight-typed guids
    pending_fuse = {}  # guid -> pre-relu input tensor, for the two guids in FUSE_GUIDS
    with open(model_path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        guid = int(lines[i]); i += 1
        op = int(lines[i]); i += 1
        deps = [tuple(int(x) for x in d.split(":")) for d in lines[i].split(",")]; i += 1
        params = [int(p) for p in lines[i].split(",")]; i += 1

        optype = ts.op_table[op]
        if optype == "Input":
            node = [graph.new_input(dims=tuple(params))]
        elif optype == "Weight":
            shape = tuple(params)
            if shape in original_weights:
                arr = original_weights[shape]
                weight_arrays[guid] = arr
                node = [graph.new_weight(dims=shape, data=arr)]
            else:
                # export_to_file() includes every constructed Weight node,
                # including orphaned ones with no real trained data --
                # e.g. the disconnected node _constant() creates for a
                # Reshape's shape-constant (its actual value reaches the
                # Reshape via the _constant_node_values side channel, not
                # a real graph edge, so this node itself is provably
                # unused). Confirmed via a dependency scan of this exact
                # file that guid 120 (shape (2,)) has zero consumers;
                # any real weight is expected to be found above instead.
                print(f"WARNING: guid {guid} has unmatched shape {shape}, "
                      "using a placeholder (expected only for orphaned "
                      "Constant-derived nodes with zero consumers)")
                node = [graph.new_weight(dims=shape, data=np.zeros(shape, dtype=np.float32))]
        elif optype == "Reshape":
            node = [graph.reshape(nodes[deps[0][0]][deps[0][1]], shape=tuple(params))]
        elif optype == "Transpose":
            # Same workaround as reconstruct_optimized.py: fold
            # weight-derived transposes directly in numpy rather than
            # relying on graph.transpose(), which this script has no
            # other need to exercise.
            ndim = params[0]
            perm = tuple(params[1:1 + ndim])
            src_guid = deps[0][0]
            if src_guid not in weight_arrays:
                raise NotImplementedError(
                    "Transpose of a non-weight tensor not handled by this script")
            arr = np.transpose(weight_arrays[src_guid], perm)
            weight_arrays[guid] = arr
            node = [graph.new_weight(dims=arr.shape, data=arr)]
        elif optype == "Conv":
            stride_h, stride_w, pad_enum, act_enum = params[8], params[9], params[10], params[11]
            padding = "SAME" if pad_enum == 0 else "VALID"
            conv_out = graph.conv2d(input=nodes[deps[0][0]][deps[0][1]],
                                     weight=nodes[deps[1][0]][deps[1][1]],
                                     strides=(stride_h, stride_w), padding=padding,
                                     activation="NONE")
            activation_fns = {0: None, 1: graph.sigmoid, 2: graph.relu, 3: graph.tanh}
            fn = activation_fns[act_enum]
            node = [fn(conv_out) if fn else conv_out]
        elif optype == "Matmul":
            node = [graph.matmul(nodes[deps[0][0]][deps[0][1]], nodes[deps[1][0]][deps[1][1]])]
        elif optype == "Add":
            node = [graph.add(nodes[deps[0][0]][deps[0][1]], nodes[deps[1][0]][deps[1][1]])]
        elif optype == "Relu":
            if guid in FUSE_GUIDS:
                pending_fuse[guid] = nodes[deps[0][0]][deps[0][1]]
                if len(pending_fuse) == len(FUSE_GUIDS):
                    a_guid, b_guid = FUSE_GUIDS
                    a, b = pending_fuse[a_guid], pending_fuse[b_guid]
                    assert a.dim(0) == 1 and b.dim(0) == 1, "fusion assumes batch=1 inputs"
                    fused_in = graph.concat(0, [a, b])
                    fused_out = graph.relu(fused_in)
                    out_a, out_b = graph.split(fused_out, 0, [1, 1])
                    nodes[a_guid] = [out_a]
                    nodes[b_guid] = [out_b]
                    print(f"fused Relu guids {a_guid} and {b_guid} via concat+relu+split")
                if guid not in nodes:
                    continue  # skip appending node/nodes[guid]=node below until pair completes
                node = nodes[guid]
            else:
                node = [graph.relu(nodes[deps[0][0]][deps[0][1]])]
        else:
            raise NotImplementedError(f"op type {optype} not handled by this script")
        nodes[guid] = node
    return graph

if __name__ == "__main__":
    model_path, onnx_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    original_weights = load_original_weights(onnx_path)
    graph = parse_and_build(model_path, original_weights)
    onnx_model = ts.export_onnx(graph)
    onnx_model.opset_import[0].version = 13
    onnx.save(onnx_model, out_path)
    print(f"exported {out_path} OK")
