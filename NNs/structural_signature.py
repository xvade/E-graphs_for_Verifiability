#!/usr/bin/env python3
# Cheap, GPU-free structural analysis of a tensat/TASO .model file -- no
# reconstruction, no ab-CROWN, just parsing the same 4-line-per-node text
# format every reconstruct_*.py script already parses (factored out here
# instead of duplicated a 5th time). Used two ways:
#   1. Phase 4's pre-flight diversity gate: a cheap signature to dedupe
#      many samples of the same source model and count how many are
#      actually structurally distinct, before spending any ab-CROWN
#      compute on a model that turns out not to have real diversity.
#   2. Phase 7's structural-feature extraction for the verifiability
#      correlation: node count, op-type histogram, depth, branching
#      factor, and (as a first-class column, not buried in the histogram)
#      every Concat/Split's axis -- axis-0-vs-axis>0 is already the single
#      most verifiability-relevant feature this project has found
#      (BUGS.md #11/#12).
#
# Usage:
#   structural_signature.py <model_file> [<model_file> ...]   (prints JSON per file)
# Or import and call `analyze(path)` / `signature(path)` directly.
import argparse
import json
import sys
from collections import Counter, defaultdict

# Prefer the REAL op_table from taso itself -- this matters beyond display:
# Phase 7's concat_split_axes feature filters by name == "Concat"/"Split",
# so a wrong int->name guess here would silently corrupt the one feature
# this project has found matters most for verifiability (BUGS.md #11/#12).
# Only falls back to a hardcoded guess (spot-checked against a live
# ts.op_table query for indices 0,1,3,8,14,15,16,18; the rest are
# best-effort and exist only so this script still runs somewhere without
# taso importable -- treat op-name-dependent results as unverified if this
# fallback path is the one that ran) when taso genuinely isn't importable.
try:
    import taso as ts
    OP_NAMES = dict(ts.op_table)  # already a sparse {int: str} dict
except (ImportError, AttributeError):
    # AttributeError happens when `taso` isn't really importable but a
    # bare `import taso` still "succeeds" by resolving to this repo's own
    # taso/ submodule directory as an empty namespace package (whenever
    # cwd or its parent is on sys.path without taso/python on it too) --
    # confirmed hitting this outside the tensat.sif container's PYTHONPATH
    # setup. Falls back to the hardcoded guess table either way.
    OP_NAMES = {
        0: "Input", 1: "Weight", 3: "Conv", 8: "Relu", 9: "Sigmoid", 10: "Tanh",
        11: "Dropout", 14: "Reshape", 15: "Transpose", 16: "Add", 17: "Mul",
        18: "Matmul", 19: "Enlarge", 20: "MergeGConv", 21: "Concat",
        22: "Split", 23: "BatchNormalization", 24: "MaxPool", 25: "AveragePool",
    }


def parse_model_file(path):
    """Returns a dict guid -> {"op": int, "deps": [(guid,idx),...], "params": [int,...]}."""
    with open(path) as f:
        lines = f.read().splitlines()
    nodes = {}
    i = 0
    while i < len(lines):
        guid = int(lines[i]); i += 1
        op = int(lines[i]); i += 1
        deps = [tuple(int(x) for x in d.split(":")) for d in lines[i].split(",")]; i += 1
        params = [int(p) for p in lines[i].split(",") if p.strip() != ""]; i += 1
        nodes[guid] = {"op": op, "deps": deps, "params": params}
    return nodes


def analyze(path):
    nodes = parse_model_file(path)
    op_counts = Counter(OP_NAMES.get(n["op"], f"Op{n['op']}") for n in nodes.values())

    consumed = set()
    for n in nodes.values():
        consumed.update(n["deps"])
    created = set()
    for guid in nodes:
        created.add((guid, 0))
        created.add((guid, 1))  # harmless over-approximation; Split's 2nd output is real, others just never get consumed at idx 1
    # A guid's real "depth" here: longest dependency chain reaching it from
    # any Input/Weight leaf (both depth 0).
    depth_memo = {}

    def depth(guid):
        if guid in depth_memo:
            return depth_memo[guid]
        n = nodes[guid]
        if n["op"] in (0, 1):  # Input, Weight
            depth_memo[guid] = 0
            return 0
        d = 1 + max((depth(dep_guid) for dep_guid, _ in n["deps"]), default=0)
        depth_memo[guid] = d
        return d

    max_depth = max((depth(g) for g in nodes), default=0)
    max_branching = max((len(n["deps"]) for n in nodes.values()), default=0)

    concat_split_axes = []
    for guid, n in nodes.items():
        name = OP_NAMES.get(n["op"], f"Op{n['op']}")
        if name in ("Concat", "Split") and n["params"]:
            concat_split_axes.append({"guid": guid, "op": name, "axis": n["params"][0]})

    return {
        "path": path,
        "node_count": len(nodes),
        "op_counts": dict(sorted(op_counts.items())),
        "max_depth": max_depth,
        "max_branching_factor": max_branching,
        "concat_split_axes": sorted(concat_split_axes, key=lambda e: e["guid"]),
    }


def signature(path):
    """A cheap, hashable structural signature for deduping samples: the
    sorted op-type histogram plus every Concat/Split's axis (the one
    feature already known to matter most for verifiability -- two
    extractions with the same op-type counts but different Concat axes are
    NOT the same structure for this project's purposes)."""
    a = analyze(path)
    axes = tuple(sorted((e["op"], e["axis"]) for e in a["concat_split_axes"]))
    return (tuple(sorted(a["op_counts"].items())), axes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_files", nargs="+")
    args = ap.parse_args()
    for path in args.model_files:
        print(json.dumps(analyze(path)))


if __name__ == "__main__":
    main()
