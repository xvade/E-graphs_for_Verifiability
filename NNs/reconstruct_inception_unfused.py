# Reconstructs the ORIGINAL (pre-tensat) InceptionMNIST graph from
# NNs/inception_mnist.taso with real weights, as the unfused baseline
# paired with reconstruct_inception_fused.py's output -- same real
# weights, same computation, for a clean side-by-side comparison.
#
# Uses the same named-weight-role approach as reconstruct_inception_fused.py
# (three (8,) biases collide by shape; roles determined here by direct
# structural tracing of NNs/inception_mnist.taso, unambiguous in this
# unfused graph since each bias is added straight to its own branch's
# conv output rather than pre-summed as in the fused version).
import numpy as np
import onnx
import taso as ts

MODEL_PATH = "NNs/inception_mnist.taso"
OUT_PATH = "NNs/inception_mnist_unfused.onnx"
WEIGHTS_NPZ = "NNs/inception_mnist_weights.npz"

GUID_ROLES = {
    101: "stem.weight",
    102: "stem.bias",
    103: "branchA.weight",
    104: "branchA.bias",
    105: "branchB.weight",
    106: "branchB.bias",
    107: "fc1.weight",
    108: "fc1.bias",
    109: "fc2.weight",
    110: "fc2.bias",
}


def volume(t):
    return int(np.prod([t.dim(i) for i in range(t.nDim)]))


def add_larger_first(graph, a, b):
    """See reconstruct_inception_fused.py's copy of this helper for why:
    alpha-beta-CROWN's onnx2pytorch loads ONNX Add as an in-place
    `out += inp`, which needs the first input to already be the larger
    (broadcast-target) shape. Add is symmetric, so reordering is free."""
    if volume(a) >= volume(b):
        return graph.add(a, b)
    return graph.add(b, a)


def load_named_weights():
    with np.load(WEIGHTS_NPZ) as f:
        return {k: f[k] for k in f.files}


def parse_and_build(model_path, named_weights):
    graph = ts.new_graph()
    nodes = {}
    weight_arrays = {}  # guid -> real numpy array, for weight-derived nodes only
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
            if guid in GUID_ROLES:
                arr = named_weights[GUID_ROLES[guid]]
                assert tuple(arr.shape) == tuple(params)
                weight_arrays[guid] = arr
                node = [graph.new_weight(dims=arr.shape, data=arr)]
            else:
                # orphaned Constant-derived node (Reshape's shape constant,
                # zero consumers) -- see reconstruct_fused_relu.py's
                # equivalent note for the same artifact in resnet2b.
                node = [graph.new_weight(dims=tuple(params), data=np.zeros(tuple(params), dtype=np.float32))]
        elif optype == "Reshape":
            node = [graph.reshape(nodes[deps[0][0]][deps[0][1]], shape=tuple(params))]
        elif optype == "Transpose":
            ndim = params[0]
            perm = tuple(params[1:1 + ndim])
            src_guid = deps[0][0]
            assert src_guid in weight_arrays, "Transpose of a non-weight tensor not handled here"
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
            node = [add_larger_first(graph, nodes[deps[0][0]][deps[0][1]], nodes[deps[1][0]][deps[1][1]])]
        elif optype == "Relu":
            node = [graph.relu(nodes[deps[0][0]][deps[0][1]])]
        else:
            raise NotImplementedError(f"op type {optype} not handled by this script")
        nodes[guid] = node
    return graph


if __name__ == "__main__":
    named_weights = load_named_weights()
    graph = parse_and_build(MODEL_PATH, named_weights)
    onnx_model = ts.export_onnx(graph)
    onnx_model.opset_import[0].version = 13  # no Split here, 13 is fine
    onnx.save(onnx_model, OUT_PATH)
    print(f"exported {OUT_PATH} OK")
