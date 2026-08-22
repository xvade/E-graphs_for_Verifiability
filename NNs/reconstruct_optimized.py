# Rebuilds a live TASO graph from a tensat-exported .model file (the format
# tensat's src/parse.rs::parse_model() reads -- see tests/parse.rs), with
# REAL weight data attached (matched back to the original ONNX model's
# initializers by shape), and exports the result to ONNX with real, correct
# numeric values.
#
# Mirrors taso/examples/load_model.py's op dispatch, but that script leaves
# new_weight() without real data (defaults to random -- fine for TASO's own
# runtime-speedup benchmarking, useless for us). This only handles the op
# types that actually appear in mnist_tiny_mlp's export (Input, Weight,
# Reshape, Transpose, Matmul, Add, Relu) -- extend the dispatch dict below
# for other op types as needed.
#
# Shape-based weight matching is unambiguous here because mnist_tiny_mlp's
# four weight tensors (784x20, 20, 20x10, 10 -- i.e. their TASO/ONNX-order
# shapes) are all distinct. A model with repeated weight shapes would need
# a less ambiguous correlation (e.g. tensat carrying stable weight
# identities through extraction), not just this shape lookup.
#
# Weight-derived Transpose is folded here in Python (np.transpose), not via
# graph.transpose()+export_onnx()'s attribute round-trip. That path is
# independently broken: core.pyx's get_operator_attr('perm') decodes the
# permutation as a plain base-N digit sequence
# (dims[i]=perIdx%N; perIdx//=N), but transpose.cc's encoder
# (permutation_to_index) and the *other* get_or_create_transpose overload
# it calls don't appear to round-trip consistently through that path --
# every Transpose we exported came back with an invalid perm (repeated
# index, e.g. [0, 0]), which ONNX rejects outright. Not worth chasing
# further given the workaround: since we already hold the real weight
# arrays in Python, transposing them directly and emitting a literal
# Weight node sidesteps the bug entirely. graph.transpose() would still be
# needed (and would still hit this bug) for a transpose applied to a
# non-weight (activation) tensor -- none appear in this model.
import sys
import numpy as np
import onnx
from onnx import numpy_helper
import taso as ts

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
            if shape not in original_weights:
                raise KeyError(
                    f"no original weight with shape {shape} for guid {guid} -- "
                    "shape-based matching is ambiguous or incomplete for this model"
                )
            arr = original_weights[shape]
            weight_arrays[guid] = arr
            node = [graph.new_weight(dims=shape, data=arr)]
        elif optype == "Reshape":
            node = [graph.reshape(nodes[deps[0][0]][deps[0][1]], shape=tuple(params))]
        elif optype == "Transpose":
            ndim = params[0]
            perm = tuple(params[1:1 + ndim])
            src_guid = deps[0][0]
            if src_guid not in weight_arrays:
                raise NotImplementedError(
                    "Transpose of a non-weight tensor -- graph.transpose()'s "
                    "ONNX perm round-trip is broken (see module docstring), "
                    "and this script only folds weight-derived transposes"
                )
            arr = np.transpose(weight_arrays[src_guid], perm)
            weight_arrays[guid] = arr
            node = [graph.new_weight(dims=arr.shape, data=arr)]
        elif optype == "Matmul":
            # tensat's own parser (parse.rs) ignores matmul's params too;
            # matching that semantics here for consistency.
            node = [graph.matmul(nodes[deps[0][0]][deps[0][1]], nodes[deps[1][0]][deps[1][1]])]
        elif optype == "Add":
            node = [graph.add(nodes[deps[0][0]][deps[0][1]], nodes[deps[1][0]][deps[1][1]])]
        elif optype == "Relu":
            node = [graph.relu(nodes[deps[0][0]][deps[0][1]])]
        else:
            raise NotImplementedError(f"op type {optype} not handled by this script yet")
        nodes[guid] = node
    return graph

if __name__ == "__main__":
    model_path, onnx_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    original_weights = load_original_weights(onnx_path)
    graph = parse_and_build(model_path, original_weights)
    # NOTE: graph.preprocess_weights() looked like the right call here (it's
    # what TASO's own Graph::optimize() uses internally to fold weight-only
    # subgraphs before its own export/run), but it has a real bug -- ops.cc's
    # preprocess_weights() mutates newGraph->inEdges while iterating over it,
    # and in practice it collapsed our whole graph down to one stray node.
    # It's not needed for correctness anyway: export_onnx() already pulls
    # real data from any genuine Weight node, so an un-folded Transpose (or
    # other op) applied to a real weight array is already a semantically
    # correct ONNX node -- folding only matters as a perf optimization for
    # TASO's own benchmark-execution pipeline, which we don't need here.
    onnx_model = ts.export_onnx(graph)
    # export_onnx()'s helper.make_model() stamps whatever opset the
    # installed onnx package currently defaults to -- which can be an
    # unreleased version onnxruntime refuses to load. Pin to a stable one;
    # nothing here (MatMul/Add/Relu/Transpose/Reshape) needs anything new.
    onnx_model.opset_import[0].version = 13
    onnx.save(onnx_model, out_path)
    print(f"exported {out_path} OK")
