# Reconstructs tensat's fused optimization of InceptionMNIST (real trained
# weights) from tensat/tmp/inception_mnist_fused_clean_optimized.model.
#
# This is the first model all session with a genuine (real-weight,
# TENSAT-selected) structural rewrite: it has an actual parallel branch
# (1x1 and 3x3 convs sharing the same input, both stride=1) matching
# tensat's PRE_DEFINED_MULTI fusion rule (tensat/src/rewrites.rs) exactly
# -- confirmed by manually decoding the exported .model file. The
# extracted graph turned out to be a valid but *hybrid* equivalent
# program, not a "pure" single-fused-conv replacement: it keeps one of
# the original two convs (branchA, guid 617) computed the ordinary way,
# and additionally builds a wider fused conv (concat(enlarge(branchA_w,
# 3x3), branchB_w) -> conv -> split) whose second half stands in for
# branchB's contribution. Both are mathematically equivalent to the
# original (conv_a(x)+bias_a) + (conv_b(x)+bias_b) -- confirmed by hand
# tracing the dependency graph -- just computed via redundant paths, which
# is presumably why the extractor picked it under real+jittered costs
# (cheaper node reuse locally, even if globally a bit wasteful).
#
# Weight matching: three of this model's real weights share shape (8,)
# (stem.bias, branchA.bias, branchB.bias), breaking pure shape-based
# lookup (see reconstruct_optimized.py's module docstring, which already
# flagged this exact risk). stem.bias is disambiguated structurally
# (guid 612's Reshape->Add partner is the conv that consumes the graph's
# Input node directly); branchA.bias/branchB.bias are only ever used
# summed together (guid 622 = Add(reshape(618), reshape(620))), so their
# assignment order between guids 618/620 doesn't affect correctness
# (addition is commutative) -- verified by hand-tracing the full
# computation back to the original merged = conv_a(x)+bias_a+conv_b(x)+
# bias_b formula.
#
# Guid numbers below are specific to this one export (tensat renumbers
# guids per run) -- this script is a one-off reconstruction of this
# specific optimized graph, not a general tool.
import sys
import numpy as np
import onnx
import taso as ts

MODEL_PATH = "tensat/tmp/inception_mnist_fused_clean_optimized.model"
OUT_PATH = "NNs/inception_mnist_fused.onnx"
# Extracted separately (host-side, taso_py env, torch available) since
# this container has taso but not torch -- see the one-liner that wrote
# this file for what's in it (InceptionMNIST.state_dict(), as float32).
WEIGHTS_NPZ = "NNs/inception_mnist_weights.npz"

GUID_ROLES = {
    610: "stem.weight",
    612: "stem.bias",
    616: "branchA.weight",
    618: "branchA.bias",   # order vs 620 doesn't matter -- see module docstring
    620: "branchB.bias",
    623: "branchB.weight",
    634: "fc1.weight",
    637: "fc1.bias",
    640: "fc2.weight",
    643: "fc2.bias",
}


def load_named_weights():
    with np.load(WEIGHTS_NPZ) as f:
        return {k: f[k] for k in f.files}


def volume(t):
    return int(np.prod([t.dim(i) for i in range(t.nDim)]))


def add_larger_first(graph, a, b):
    """graph.add() is mathematically commutative, but alpha-beta-CROWN's
    onnx2pytorch loads ONNX Add as an in-place `out += inp`, which
    requires the *first* input to already be the broadcast-target (larger)
    shape -- discovered when a bias-reshape (e.g. [1,8,1,1]) ended up
    first and a conv output (e.g. [1,8,28,28]) second, and onnx2pytorch's
    in-place add couldn't grow into the second operand's shape. TASO's own
    export order isn't guaranteed to put the larger tensor first, so
    reorder explicitly here (harmless: Add is symmetric)."""
    if volume(a) >= volume(b):
        return graph.add(a, b)
    return graph.add(b, a)


def enlarge_np(w1, w2):
    """Mirrors taso/src/cudnn/enlarge_kernel.cu exactly: zero-pad w1's last
    two (spatial) dims to match w2's, centered."""
    assert w1.ndim == 4 and w2.ndim == 4
    out_h, out_w = w2.shape[2], w2.shape[3]
    src_h, src_w = w1.shape[2], w1.shape[3]
    off_h = (out_h - src_h) // 2
    off_w = (out_w - src_w) // 2
    out = np.zeros((w1.shape[0], w1.shape[1], out_h, out_w), dtype=w1.dtype)
    out[:, :, off_h:off_h + src_h, off_w:off_w + src_w] = w1
    return out


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
            assert w1_guid in weight_arrays and w2_guid in weight_arrays, \
                "Enlarge on a non-weight tensor not handled here"
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
                node = []
                for k, part in enumerate(parts):
                    weight_arrays[(guid, k)] = part  # not referenced elsewhere, kept for clarity
                    node.append(graph.new_weight(dims=part.shape, data=part))
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
    # Unlike this project's earlier models, this graph has a real Split
    # node. TASO's exporter always emits Split's sizes as a node
    # *attribute* (taso/python/taso/__init__.py: operator_attrs['Split']
    # = ['axis', 'split']) -- correct for ONNX's own Split spec up through
    # opset 12, but opset 13 moved split sizes to a second *input* tensor
    # and dropped the attribute form entirely, so onnxruntime rejects
    # TASO's attribute-based Split under opset 13 ("Unrecognized
    # attribute: split"). Pin to 11 instead of the 13 used elsewhere in
    # this project -- still a long-stable, fully-supported opset for
    # every other op this graph uses (Conv/Relu/Add/Reshape/MatMul/
    # Concat/Transpose), just old enough for Split's attribute form to
    # still be valid.
    onnx_model.opset_import[0].version = 11

    # The hybrid extraction this optimized graph represents (see module
    # docstring) leaves split[0] of the fused conv's output genuinely
    # unused -- only split[1] feeds the rest of the network. export_onnx()
    # correctly (from its own perspective) treats any tensor with zero
    # consumers as a graph output, so this dangling half shows up as a
    # second, spurious ONNX output alongside the real one. Keep only the
    # true final output: shape (1, 10) (10-class logits), the only shape
    # a real output of this classifier can have.
    real_outputs = [o for o in onnx_model.graph.output
                    if [d.dim_value for d in o.type.tensor_type.shape.dim] == [1, 10]]
    assert len(real_outputs) == 1, f"expected exactly one (1,10) output, found {len(onnx_model.graph.output)} total"
    del onnx_model.graph.output[:]
    onnx_model.graph.output.extend(real_outputs)

    onnx.save(onnx_model, OUT_PATH)
    print(f"exported {OUT_PATH} OK")
