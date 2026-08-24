# Reconstructs a SECOND, better fusion of InceptionMNIST from
# tensat/tmp/inception_mnist_v2_optimized.model -- extracted after adding
# an active cost penalty (not just neutral non-discounting) for axis-0
# Concat/Split in tensat's --favor_fusion (tensat/src/optimize.rs), since
# the first fusion attempt (reconstruct_inception_fused.py) turned out to
# rely on a batch-axis (axis=0) relu-merge trick that BUGS.md #11 found
# structurally unverifiable by alpha-beta-CROWN (auto_LiRPA explicitly
# refuses to bound-propagate Concat on axis 0).
#
# This extraction instead batches the stem's relu together with the
# branch-merge's relu via a CHANNEL-axis (axis=1) concat+relu+split --
# mathematically identical trick, but auto_LiRPA only forbids axis 0, so
# this should be fully verifiable. Both original convs (branchA 1x1,
# branchB 3x3) stay separate/unfused this time -- no Enlarge, no
# weight-level conv fusion; the only structural difference from the
# unfused baseline is this one relu-batching detour, which redundantly
# recomputes the stem's relu a second time (the unused half of the split,
# guid 499:0 here, discarded) alongside the branch-merge's relu it
# actually needed. Wasteful but valid: not circular, since the stem's
# relu is computed via its own direct path *first* (needed by both convs)
# and only optionally re-derived a second time by this trick.
#
# Weight matching: branchA.bias and branchB.bias are unambiguous here
# (unlike the first fusion attempt) since each is added directly to its
# own conv's output, not pre-summed -- traced structurally the same way
# as reconstruct_inception_unfused.py.
import numpy as np
import onnx
import taso as ts

MODEL_PATH = "tensat/tmp/inception_mnist_v2_optimized.model"
OUT_PATH = "NNs/inception_mnist_fused_v2.onnx"
WEIGHTS_NPZ = "NNs/inception_mnist_weights.npz"

GUID_ROLES = {
    480: "stem.weight",
    482: "stem.bias",
    486: "branchA.weight",
    488: "branchA.bias",
    491: "branchB.weight",
    493: "branchB.bias",
    501: "fc1.weight",
    504: "fc1.bias",
    507: "fc2.weight",
    510: "fc2.bias",
}


def load_named_weights():
    with np.load(WEIGHTS_NPZ) as f:
        return {k: f[k] for k in f.files}


def volume(t):
    return int(np.prod([t.dim(i) for i in range(t.nDim)]))


def enlarge_np(w1, w2):
    """Mirrors taso/src/cudnn/enlarge_kernel.cu: zero-pad w1's spatial
    dims to match w2's, centered. Not exercised by this specific
    extraction (no Enlarge node here), kept for completeness/safety."""
    assert w1.ndim == 4 and w2.ndim == 4
    out_h, out_w = w2.shape[2], w2.shape[3]
    src_h, src_w = w1.shape[2], w1.shape[3]
    off_h = (out_h - src_h) // 2
    off_w = (out_w - src_w) // 2
    out = np.zeros((w1.shape[0], w1.shape[1], out_h, out_w), dtype=w1.dtype)
    out[:, :, off_h:off_h + src_h, off_w:off_w + src_w] = w1
    return out


def add_larger_first(graph, a, b):
    """alpha-beta-CROWN's onnx2pytorch loads ONNX Add as an in-place
    `out += inp`, which needs the first input to already be the larger
    (broadcast-target) shape. Add is symmetric, so reordering is free."""
    if volume(a) >= volume(b):
        return graph.add(a, b)
    return graph.add(b, a)


def parse_and_build(model_path, named_weights):
    graph = ts.new_graph()
    nodes = {}
    weight_arrays = {}
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
            role = GUID_ROLES[guid]
            arr = named_weights[role]
            assert tuple(arr.shape) == tuple(params), f"guid {guid} ({role}): shape mismatch {arr.shape} vs {tuple(params)}"
            weight_arrays[guid] = arr
            node = [graph.new_weight(dims=arr.shape, data=arr)]
        elif optype == "Reshape":
            src_guid = deps[0][0]
            node = [graph.reshape(nodes[deps[0][0]][deps[0][1]], shape=tuple(params))]
            if src_guid in weight_arrays:
                weight_arrays[guid] = weight_arrays[src_guid].reshape(tuple(params))
        elif optype == "Transpose":
            ndim = params[0]
            perm = tuple(params[1:1 + ndim])
            src_guid = deps[0][0]
            assert src_guid in weight_arrays, "Transpose of a non-weight tensor not handled here"
            arr = np.transpose(weight_arrays[src_guid], perm)
            weight_arrays[guid] = arr
            node = [graph.new_weight(dims=arr.shape, data=arr)]
        elif optype == "Enlarge":
            w1_guid, w2_guid = deps[0][0], deps[1][0]
            assert w1_guid in weight_arrays and w2_guid in weight_arrays
            arr = enlarge_np(weight_arrays[w1_guid], weight_arrays[w2_guid])
            weight_arrays[guid] = arr
            node = [graph.new_weight(dims=arr.shape, data=arr)]
        elif optype == "Concat":
            axis = params[0]
            src_guids = [d[0] for d in deps]
            if all(g in weight_arrays for g in src_guids):
                arrs = [weight_arrays[d[0]] for d in deps]
                arr = np.concatenate(arrs, axis=axis)
                weight_arrays[guid] = arr
                node = [graph.new_weight(dims=arr.shape, data=arr)]
            else:
                inputs = [nodes[d[0]][d[1]] for d in deps]
                node = [graph.concat(axis, inputs)]
        elif optype == "Split":
            axis = params[0]
            sizes = params[1:]
            src_guid, src_idx = deps[0]
            if src_guid in weight_arrays:
                arr = weight_arrays[src_guid]
                assert src_idx == 0
                parts = np.split(arr, np.cumsum(sizes)[:-1], axis=axis)
                node = [graph.new_weight(dims=part.shape, data=part) for part in parts]
            else:
                node = graph.split(nodes[src_guid][src_idx], axis, list(sizes))
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
    # This model has a real Split node -- opset 13 moved Split's sizes to
    # an input tensor and dropped the attribute form taso always emits
    # (BUGS.md #9). Pin to 11 like the first fusion attempt did.
    onnx_model.opset_import[0].version = 11

    # Drop any ONNX output with zero real consumers (BUGS.md's earlier
    # note on reconstruct_inception_fused.py): keep only the genuine
    # (1,10) logits output.
    real_outputs = [o for o in onnx_model.graph.output
                    if [d.dim_value for d in o.type.tensor_type.shape.dim] == [1, 10]]
    assert len(real_outputs) == 1, f"expected exactly one (1,10) output, found {len(onnx_model.graph.output)} total"
    del onnx_model.graph.output[:]
    onnx_model.graph.output.extend(real_outputs)

    onnx.save(onnx_model, OUT_PATH)
    print(f"exported {OUT_PATH} OK")
