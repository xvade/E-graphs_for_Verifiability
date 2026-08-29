#!/usr/bin/env python3
# Hand-constructs the CONV-WEIGHT-FUSED InceptionMNIST -- the structure the
# multi-pattern conv-fusion rule (rules 11/13 + enlarge) actually FIRES into
# the e-graph (see NNs/matchprobe_logs/), as opposed to the relu-merge that
# was the only fusion previously verified. Two parallel convs sharing input
# Relu119_fwd0 -- branch B (Conv120, 1x1, pad 0) and branch A (Conv121, 3x3,
# pad 1) -- are merged into ONE wider conv by:
#   1. enlarging B's [8,8,1,1] kernel to [8,8,3,3], center-only (the 1x1
#      value at [:,:,1,1], zeros elsewhere). Run at pad 1 this is EXACTLY the
#      1x1-pad-0 result, borders included, because only the center tap is
#      nonzero and the center always maps to a real input pixel.
#   2. concatenating [B_enlarged; A] along output-channel axis 0 -> [16,8,3,3],
#      biases likewise -> [16]. This weight concat is a CONSTANT fold (done
#      here in numpy), so it never appears as an ONNX op -- exactly the
#      "weight-side concat folds away" property. The only NEW activation op is
#      the channel Split that recovers the two branches.
#
# Result is numerically IDENTICAL to the unfused model; the point is purely to
# measure whether the verifier's branching trips on the activation Split (as
# it did on the relu-merge's Concat, BUGS #11/#12) or handles it cleanly.
#
# Usage: build_inception_convfused.py  (reads/writes fixed paths under NNs/)
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

NNS = __file__.rsplit("/", 1)[0]
UNFUSED = f"{NNS}/inception_mnist_unfused_simplified.onnx"
OUT = f"{NNS}/inception_mnist_convfused.onnx"


def get_inits(model):
    return {i.name: numpy_helper.to_array(i).astype(np.float32) for i in model.graph.initializer}


def conv_attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            return list(a.ints) if a.ints else a.i
    return default


def main():
    m = onnx.load(UNFUSED)
    inits = get_inits(m)

    # locate the two branch convs by their shared input + weight shapes
    convs = {n.output[0]: n for n in m.graph.node if n.op_type == "Conv"}
    # Conv120: 1x1 branch (weight [8,8,1,1]); Conv121: 3x3 branch (weight [8,8,3,3])
    b_w = inits["Conv120_weight"]           # (8,8,1,1)
    b_b = inits["_v_34"]                     # (8,)
    a_w = inits["Conv121_weight"]           # (8,8,3,3)
    a_b = inits["_v_37"]                     # (8,)
    assert b_w.shape == (8, 8, 1, 1) and a_w.shape == (8, 8, 3, 3)

    # enlarge B's 1x1 -> 3x3, center-only
    b_w_enl = np.zeros((8, 8, 3, 3), dtype=np.float32)
    b_w_enl[:, :, 1, 1] = b_w[:, :, 0, 0]

    # concat [B; A] on output-channel axis -> wide conv (constant fold)
    wide_w = np.concatenate([b_w_enl, a_w], axis=0)   # (16,8,3,3)
    wide_b = np.concatenate([b_b, a_b], axis=0)       # (16,)

    # ---- rebuild graph: keep stem conv + relu, replace 2-conv+add with
    #      wide-conv -> split(axis1) -> add, keep the FC tail verbatim ----
    keep_nodes = []
    for n in m.graph.node:
        if n.output[0] in ("Conv120_fwd0", "Conv121_fwd0", "Add124_fwd0"):
            continue  # these get replaced
        keep_nodes.append(n)

    shared = "Relu119_fwd0"  # the input both branches shared
    wide_conv = helper.make_node(
        "Conv", [shared, "wide_w", "wide_b"], ["wide_conv_out"],
        name="WideConv", kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    split = helper.make_node(
        "Split", ["wide_conv_out"], ["split_B", "split_A"], name="ChannelSplit", axis=1)
    add = helper.make_node("Add", ["split_B", "split_A"], ["Add124_fwd0"], name="BranchAdd")

    # insert the 3 new nodes where the old branch nodes were (right after stem relu)
    new_nodes = []
    inserted = False
    for n in keep_nodes:
        new_nodes.append(n)
        if n.output[0] == shared and not inserted:
            new_nodes.extend([wide_conv, split, add])
            inserted = True

    # initializers: drop the two branch convs' weights/biases, add wide ones
    keep_inits = [i for i in m.graph.initializer
                  if i.name not in ("Conv120_weight", "Conv121_weight", "_v_34", "_v_37")]
    keep_inits.append(numpy_helper.from_array(wide_w, "wide_w"))
    keep_inits.append(numpy_helper.from_array(wide_b, "wide_b"))

    g = helper.make_graph(new_nodes, "inception_convfused",
                          m.graph.input, m.graph.output, keep_inits)
    fused = helper.make_model(g, opset_imports=[helper.make_opsetid("", 11)])
    fused.ir_version = m.ir_version
    onnx.checker.check_model(fused)
    onnx.save(fused, OUT)
    print(f"wrote {OUT}")

    # ---- numerical equivalence check vs unfused ----
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, 28, 28)).astype(np.float32)
    r_un = ort.InferenceSession(UNFUSED).run(None, {"data": x})[0]
    r_fu = ort.InferenceSession(OUT).run(None, {"data": x})[0]
    diff = float(np.max(np.abs(r_un - r_fu)))
    print(f"max abs diff unfused vs convfused: {diff:.2e}")
    assert diff < 1e-4, f"NOT numerically equivalent (diff={diff})"
    print("OK: convfused is numerically identical to unfused")


if __name__ == "__main__":
    main()
