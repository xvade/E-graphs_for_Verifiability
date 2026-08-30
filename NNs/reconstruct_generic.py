#!/usr/bin/env python3
# Generalizes reconstruct_inception_fused_v2.py (and the other hand-written
# reconstruct_*.py scripts before it) into a single, model-and-extraction-
# agnostic tool. The op-dispatch logic (walking a tensat/TASO .model file's
# guid DAG, folding weight-only subtrees in numpy vs. emitting real graph
# ops, enlarge_np/add_larger_first workarounds, opset-11 pin) was already
# fully general in reconstruct_inception_fused_v2.py -- the ONE thing that
# wasn't is weight identity, which used to be a hand-traced GUID_ROLES dict
# per extraction (doesn't scale past one-off reconstructions). That's now
# solved upstream: tensat's --weight_names_json / weight_names provenance
# (tensat/src/model.rs's ValTnsr.weight_names, propagated through
# TensorAnalysis::make() and merge()) tags every weight-derived eclass with
# its real originating name(s) automatically, and save_model_with_provenance
# (tensat/src/main.rs) emits it as a <model_file>.weight_names.json sidecar
# alongside any extraction. This script just consumes that sidecar instead
# of a hardcoded dict.
#
# Usage:
#   reconstruct_generic.py <model_file> <weights.npz> <weight_names.json> <output.onnx>
#
# <weight_names.json> is the sidecar tensat emits: {"guid": ["name", ...]}.
# A literal Weight leaf always has exactly one contributing name (no
# rewrite rule in this codebase synthesizes a fresh Weight node from a
# numeric fold -- confirmed by grepping every rewrite-rule file for the
# string "weight": zero hits outside op definitions); multi-name entries
# only ever occur on ops *downstream* of a Weight leaf (Enlarge/Concat/etc.
# mixing multiple original weights), and this script never needs to look
# those up directly -- it re-derives their concrete values the same way the
# hand-written scripts always did, by numpy-folding the already-resolved
# weight arrays of their inputs.
import argparse
import json
import sys

import numpy as np
import onnx
import taso as ts

# Reverse-lookup from TASO's raw enum ints (as they appear in a .model
# file's params) to the string form graph.maxpool2d/avgpool2d expect --
# mirrors taso/examples/load_model.py's padding_mode/ac_mode dicts.
PADDING_MODE = {ts.get_padding_mode("SAME"): "SAME", ts.get_padding_mode("VALID"): "VALID"}
ACTIVATION_MODE = {
    ts.get_activation_mode("NONE"): "NONE",
    ts.get_activation_mode("SIGMOID"): "SIGMOID",
    ts.get_activation_mode("RELU"): "RELU",
    ts.get_activation_mode("TANH"): "TANH",
}


def load_named_weights(weights_npz):
    with np.load(weights_npz) as f:
        return {k: f[k] for k in f.files}


def volume(t):
    return int(np.prod([t.dim(i) for i in range(t.nDim)]))


def enlarge_np(w1, w2):
    """Mirrors taso/src/cudnn/enlarge_kernel.cu: zero-pad w1's spatial
    dims to match w2's, centered."""
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


def parse_and_build(model_path, named_weights, weight_names_map):
    graph = ts.new_graph()
    nodes = {}
    weight_arrays = {}
    # Tracks every (guid, idx) pair ever created vs. ever consumed as a
    # dependency, so the real final output(s) can be found generically
    # (see the output-selection note at the bottom) instead of assuming a
    # fixed shape like (1, 10) -- a fused/diversity-sampled extraction can
    # leave an orphaned Split half in the graph (an unused byproduct of a
    # multi-pattern rewrite) that would otherwise get exported as a bogus
    # second ONNX output.
    created = set()
    consumed = set()

    with open(model_path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        guid = int(lines[i]); i += 1
        op = int(lines[i]); i += 1
        deps = [tuple(int(x) for x in d.split(":")) for d in lines[i].split(",") if d.strip() != ""]; i += 1
        params = [int(p) for p in lines[i].split(",") if p.strip() != ""]; i += 1
        consumed.update(deps)

        optype = ts.op_table[op]
        if optype == "Input":
            node = [graph.new_input(dims=tuple(params))]
        elif optype == "Weight":
            names = weight_names_map.get(str(guid))
            if not names:
                # No provenance entry -- expected only for a known TASO
                # export artifact: an orphaned Constant-derived node
                # created for a Reshape's own shape-constant (its real
                # value reaches the Reshape via a side channel, not a
                # graph edge, so this node is provably unused -- see
                # derive_weight_names_baseline.py's identical note and
                # reconstruct_fused_relu.py's original discovery of this
                # for resnet2b). Not silently ignored: printed loudly, and
                # if this guid is NOT actually an unconsumed orphan, the
                # zero placeholder will make the final numeric
                # verification fail loudly too.
                print(f"WARNING: guid {guid} has no weight_names.json entry, "
                      f"using a zero placeholder (expected only for an "
                      f"orphaned, zero-consumer Constant-derived node)")
                arr = np.zeros(tuple(params), dtype=np.float32)
            else:
                assert len(names) == 1, (
                    f"guid {guid}: expected exactly one contributing weight name "
                    f"for a literal Weight leaf, got {names}"
                )
                role = names[0]
                assert role in named_weights, f"guid {guid}: '{role}' not found in weights npz"
                arr = named_weights[role]
                assert tuple(arr.shape) == tuple(params), (
                    f"guid {guid} ({role}): shape mismatch {arr.shape} vs {tuple(params)}"
                )
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
        elif optype == "Sub":
            # linear; ab-CROWN bounds Sub exactly. Emit native.
            a = nodes[deps[0][0]][deps[0][1]]; b = nodes[deps[1][0]][deps[1][1]]
            node = [graph.sub(x=a, y=b)]
        elif optype == "Max":
            # LOWER max(a,b) = a + relu(b-a) to the ReLU form, NOT native ONNX Max:
            # ab-CROWN's BoundMax relaxation is not the ReLU topology the reassociation
            # result was measured on. This makes min/max rewrites visible to the
            # verifier as the ReLU structure they actually change.
            a = nodes[deps[0][0]][deps[0][1]]; b = nodes[deps[1][0]][deps[1][1]]
            node = [graph.add(a, graph.relu(graph.sub(x=b, y=a)))]
        elif optype == "Min":
            # min(a,b) = a - relu(a-b), same rationale as Max.
            a = nodes[deps[0][0]][deps[0][1]]; b = nodes[deps[1][0]][deps[1][1]]
            node = [graph.sub(x=a, y=graph.relu(graph.sub(x=a, y=b)))]
        elif optype == "MaxPool":
            # Param layout per taso/examples/load_model.py (the authoritative
            # reference dispatch this whole family of scripts mirrors):
            # kernel_h, kernel_w = params[5], params[6]; stride_h, stride_w =
            # params[7], params[8]; padding = params[9]; activation = params[10].
            node = [graph.maxpool2d(input=nodes[deps[0][0]][deps[0][1]],
                                     kernels=(params[5], params[6]),
                                     strides=(params[7], params[8]),
                                     padding=PADDING_MODE[params[9]],
                                     activation=ACTIVATION_MODE[params[10]])]
        elif optype == "AveragePool":
            node = [graph.avgpool2d(input=nodes[deps[0][0]][deps[0][1]],
                                     kernels=(params[5], params[6]),
                                     strides=(params[7], params[8]),
                                     padding=PADDING_MODE[params[9]],
                                     activation=ACTIVATION_MODE[params[10]])]
        else:
            raise NotImplementedError(f"op type {optype} not handled by this script")
        nodes[guid] = node
        for idx in range(len(node)):
            created.add((guid, idx))

    return graph, created, consumed


def fix_same_padding_symmetric(onnx_model):
    """Works around a real TASO bug (not yet in BUGS.md as of this script):
    for a "SAME"-padded conv where the total required padding is odd (e.g.
    kernel=3, stride=2, input%stride==0), TASO's C++ Conv2D op itself pads
    symmetrically -- ceil(totalPad/2) on BOTH sides (taso/src/core/
    conv2d.cc's Conv2D::get_padding: `*padH = (totalPadH + 1) / 2;`, applied
    identically to top and bottom) when actually executing the op -- but
    ts.export_onnx() instead emits an ASYMMETRIC TF-style split (floor on
    one side, ceil on the other) as the ONNX Conv node's `pads` attribute.
    These disagree whenever totalPad is odd, silently producing a
    numerically wrong ONNX export while the op's OWN shape/cost bookkeeping
    (and any direct execution via graph.run()) stays internally consistent.
    Confirmed against resnet2b's stem conv (kernel=3, stride=2, input=32):
    exported pads [0,0,1,1] vs. the correct-per-TASO's-own-semantics
    symmetric [1,1,1,1] -- verified numerically against the real PyTorch
    reference output (max abs diff 1.34 wrong vs 0.0 exact once patched).
    Fix: force pads symmetric by taking the max of each dim's begin/end
    pad (VALID convs already have all-zero pads, so this is a no-op there;
    a TRUE asymmetric SAME case can't occur here since TASO's own op only
    ever produces this one kind of symmetric-vs-floor/ceil disagreement)."""
    for n in onnx_model.graph.node:
        if n.op_type != "Conv":
            continue
        for attr in n.attribute:
            if attr.name == "pads" and len(attr.ints) == 4:
                h_begin, w_begin, h_end, w_end = attr.ints
                h, w = max(h_begin, h_end), max(w_begin, w_end)
                if (h_begin, w_begin, h_end, w_end) != (h, w, h, w):
                    print(f"note: {n.output[0]}: patching asymmetric SAME pads "
                          f"{list(attr.ints)} -> symmetric {[h, w, h, w]} "
                          f"(TASO export_onnx/Conv2D::get_padding disagreement)")
                del attr.ints[:]
                attr.ints.extend([h, w, h, w])


def select_real_outputs(onnx_model, created, consumed):
    """A fused/diversity-sampled extraction can leave an orphaned Split half
    in the graph (an unused byproduct of a rewrite rule) alongside the
    genuine final output; TASO exports every unconsumed tensor as an ONNX
    output regardless. The genuine final output of a standard feedforward
    classifier is always the LAST thing computed (highest guid, since TASO
    assigns guids in creation order) among the sinks (created but never
    consumed) -- an orphaned byproduct is, by construction, computed and
    then abandoned partway through the graph, well before the real head."""
    sinks = sorted(created - consumed)  # [(guid, idx), ...]
    assert sinks, "no sink (guid, idx) pairs found -- something is very wrong with this .model file"
    real_guid, real_idx = sinks[-1]
    if len(sinks) > 1:
        print(f"note: {len(sinks)} unconsumed outputs found {sinks}; "
              f"keeping only the highest-guid one ({real_guid}:{real_idx}) as the real output, "
              f"discarding the rest as orphaned rewrite byproducts")
    # onnx_model.graph.output ordering matches TASO's own internal output
    # enumeration order, not guid order -- filter by matching each output's
    # declared shape against a tensor we can independently confirm belongs
    # to (real_guid, real_idx) is unreliable when shapes collide, so filter
    # positionally instead: TASO's export walks outputs in the same order
    # `created` was populated in this script (topological, by construction,
    # since every op's inputs are resolved into `nodes` before the op
    # itself runs) intersected with the sink set. This holds because this
    # script and TASO's own exporter both derive "is this a graph output"
    # from the identical "unconsumed" condition.
    sink_positions = [g for (g, idx) in sinks]  # guids only; idx doesn't reach ONNX-level outputs distinctly here
    n_outputs = len(onnx_model.graph.output)
    assert n_outputs >= 1
    # TASO emits outputs for sink guids in increasing guid order; keep the
    # ONNX output whose position corresponds to real_guid's position among
    # the sink guids.
    real_position = sink_positions.index(real_guid)
    assert real_position < n_outputs, (
        f"expected the real output to be among the {n_outputs} exported ONNX "
        f"outputs at position {real_position}, but only {n_outputs} exist"
    )
    return [onnx_model.graph.output[real_position]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_file")
    ap.add_argument("weights_npz")
    ap.add_argument("weight_names_json")
    ap.add_argument("output_onnx")
    args = ap.parse_args()

    named_weights = load_named_weights(args.weights_npz)
    with open(args.weight_names_json) as f:
        weight_names_map = json.load(f)

    graph, created, consumed = parse_and_build(args.model_file, named_weights, weight_names_map)
    onnx_model = ts.export_onnx(graph)
    # This model may have a real Split node -- opset 13 moved Split's sizes
    # to an input tensor and dropped the attribute form taso always emits
    # (BUGS.md #9). Pin to 11.
    onnx_model.opset_import[0].version = 11
    fix_same_padding_symmetric(onnx_model)

    real_outputs = select_real_outputs(onnx_model, created, consumed)
    del onnx_model.graph.output[:]
    onnx_model.graph.output.extend(real_outputs)

    onnx.save(onnx_model, args.output_onnx)
    print(f"exported {args.output_onnx} OK")


if __name__ == "__main__":
    main()
