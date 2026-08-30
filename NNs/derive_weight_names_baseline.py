#!/usr/bin/env python3
# One-time derivation of a baseline guid->real_name sidecar for a model that
# was never hand-traced (unlike InceptionMNIST, whose GUID_ROLES dict this
# script's output format matches -- see NNs/inception_mnist_weight_names_baseline.json).
#
# Matches each Weight guid in the model's raw .taso export to the ONNX
# model's initializers, first by shape (unambiguous whenever a shape is
# unique), and within any shape that collides among multiple weights (e.g.
# resnet2b has three (16,16,3,3) conv kernels and five (16,) biases), by
# relative position: the Nth guid of that shape (in ascending guid order,
# i.e. TASO's own creation/usage order) is matched to the Nth ONNX
# initializer of that shape (in the ONNX file's own declaration order).
# This assumes TASO's ONNX loader creates Weight ops in the same relative
# order the ONNX file declares same-shaped initializers -- true whenever
# the ONNX graph's nodes reference each initializer in the same order the
# file lists them (the normal case for a standard torch.onnx.export, and
# spot-checked true for resnet2b's own file below). Not a proof, so this
# script's output MUST be validated by an end-to-end numeric reconstruction
# check afterward (see NNs/reconstruct_generic.py's regression check) --
# treat this as a good starting guess, not a guarantee, especially for a
# model without one already verified.
#
# Usage: derive_weight_names_baseline.py <model.taso> <model.onnx> \
#            <out_names.json> <out_weights.npz> [--override guid=name ...]
import argparse
import json
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper


def parse_weight_guids(taso_path):
    """Returns [(guid, shape_tuple), ...] in ascending-guid (file/creation)
    order, for every Weight-typed node in the .taso file. Op type 1 is
    OP_WEIGHT per taso/src/parse.rs's OpType_OP_WEIGHT arm and this file's
    own op_table (spot-checked directly against inception_mnist.taso, whose
    guid 101 has op type 1 and dims 8,1,3,3 == stem.weight, a known-correct
    entry from the existing hand-derived GUID_ROLES dict)."""
    with open(taso_path) as f:
        lines = f.read().splitlines()
    weights = []
    i = 0
    while i < len(lines):
        guid = int(lines[i]); i += 1
        op = int(lines[i]); i += 1
        i += 1  # deps, unused here
        params = [int(p) for p in lines[i].split(",") if p.strip()]; i += 1
        if op == 1:  # OP_WEIGHT
            weights.append((guid, tuple(params)))
    return weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("taso_path")
    ap.add_argument("onnx_path")
    ap.add_argument("out_names_json")
    ap.add_argument("out_weights_npz")
    ap.add_argument("--override", action="append", default=[],
                     help="guid=name, forces this guid's mapping regardless of shape/position matching")
    args = ap.parse_args()

    overrides = {}
    for o in args.override:
        guid_s, name = o.split("=", 1)
        overrides[int(guid_s)] = name

    onnx_model = onnx.load(args.onnx_path)
    init_by_shape = defaultdict(list)  # shape -> [name, ...] in file order
    weight_arrays = {}
    for init in onnx_model.graph.initializer:
        arr = numpy_helper.to_array(init).astype(np.float32)
        weight_arrays[init.name] = arr
        init_by_shape[tuple(arr.shape)].append(init.name)

    guids_by_shape = defaultdict(list)  # shape -> [guid, ...] in ascending-guid order
    for guid, shape in parse_weight_guids(args.taso_path):
        guids_by_shape[shape].append(guid)

    guid_to_name = {}
    for shape, guids in guids_by_shape.items():
        names = init_by_shape.get(shape, [])
        if not names:
            # A Weight guid with a shape matching NO real ONNX initializer
            # at all is a known TASO export artifact, not a bug: TASO's
            # export_to_file() includes every constructed Weight node,
            # including an orphaned one created for a Reshape's own
            # shape-constant (its real value reaches the Reshape via a
            # side channel, not a graph edge -- see
            # reconstruct_fused_relu.py's identical note for resnet2b's
            # own guid 120, shape (2,)). Leave it out of guid_to_name;
            # reconstruct_generic.py fills any such unmatched, unconsumed
            # Weight guid with a zero placeholder.
            print(f"note: shape {shape} (guids {guids}) matches no ONNX "
                  f"initializer -- treating as an orphaned Constant-derived "
                  f"node, not a real weight")
            continue
        assert len(names) == len(guids), (
            f"shape {shape}: {len(guids)} Weight guids in {args.taso_path} "
            f"({guids}) but {len(names)} ONNX initializers of that shape in "
            f"{args.onnx_path} ({names}) -- counts must match exactly"
        )
        if len(guids) > 1:
            print(f"note: shape {shape} collides across {len(guids)} weights "
                  f"-- matching by position: {list(zip(guids, names))}")
        for guid, name in zip(guids, names):
            guid_to_name[guid] = name

    for guid, name in overrides.items():
        assert name in weight_arrays, f"--override guid={guid}={name}: '{name}' not an ONNX initializer"
        print(f"override: guid {guid} -> {name} (was {guid_to_name.get(guid)!r})")
        guid_to_name[guid] = name

    with open(args.out_names_json, "w") as f:
        json.dump({str(g): n for g, n in sorted(guid_to_name.items())}, f, indent=2)
    np.savez(args.out_weights_npz, **weight_arrays)
    print(f"wrote {args.out_names_json} ({len(guid_to_name)} weights) and {args.out_weights_npz}")


if __name__ == "__main__":
    main()
