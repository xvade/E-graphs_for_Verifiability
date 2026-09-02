#!/usr/bin/env python3
"""Env-independent dispatch tests for reconstruct_generic.py's op arms.

taso is unrunnable in this repo's available envs (host taso is a stub; the
container's python can't import onnx -- PROBLEMATIC.md #5), so a real
reconstruct round-trip can only run in the GPU env where tll ran. The DISPATCH
logic, though, is the only new code -- taso's ops themselves are not -- and it's
testable without taso by injecting a fake `taso`/`onnx` into sys.modules before
import and driving parse_and_build with a synthetic .model. This is the same
before-application guard Test 12 is for tensat (see PROBLEMATIC.md's process
lesson). Runtime numeric validation stays deferred to the reconstruct env.

    python3 NNs/tests/test_reconstruct_arms.py   # exits nonzero on any failure
"""
import os
import sys
import types
import tempfile

import numpy as np

# ---- fake taso / onnx (installed BEFORE importing the module under test) ----

# Mirrors taso's real OpType->name table (see taso/python/taso/_cython/core.pyx),
# including the load-bearing quirk this whole change is about: OP_MUL (int 19) has
# NO entry -- only OP_EW_MUL (17) is named "Mul". Matmul=18, Enlarge=20, so
# reconstruct_op_names must derive OP_MUL=19 from Matmul+1 and pass its Enlarge check.
FAKE_OP_TABLE = {
    0: "Input", 1: "Weight", 3: "Conv", 6: "MaxPool", 7: "AveragePool",
    8: "Relu", 9: "Sigmoid", 10: "Tanh", 12: "Concat", 13: "Split",
    14: "Reshape", 15: "Transpose", 16: "Add", 17: "Mul", 18: "Matmul",
    20: "Enlarge", 28: "Sub", 29: "Max", 30: "Min",
    # 19 == OP_MUL intentionally absent, exactly as in real taso.
}


class MockTensor:
    def __init__(self, dims):
        self.dims = tuple(int(d) for d in dims)
        self.nDim = len(self.dims)

    def dim(self, i):
        return self.dims[i]


class MockGraph:
    """Records every builder call; returns MockTensors so volume() works."""
    def __init__(self):
        self.calls = []

    def _rec(self, name, *args, **kw):
        self.calls.append((name, args, kw))

    def new_input(self, dims):
        self._rec("new_input", dims=dims)
        return MockTensor(dims)

    def new_weight(self, dims, data=None):
        self._rec("new_weight", dims=dims)
        return MockTensor(dims)

    def matmul(self, a, b, activation="NONE"):
        self._rec("matmul", a, b)
        return MockTensor(a.dims)

    def mul(self, a, b):
        self._rec("mul", a, b)
        return MockTensor(a.dims if a.nDim >= b.nDim else b.dims)

    def add(self, a, b):
        self._rec("add", a, b)
        return MockTensor(a.dims)

    def sub(self, x, y):
        self._rec("sub", x, y)
        return MockTensor(x.dims)

    def relu(self, x, inplace=False):
        self._rec("relu", x)
        return MockTensor(x.dims)

    def sigmoid(self, x):
        self._rec("sigmoid", x)
        return MockTensor(x.dims)

    def tanh(self, x):
        self._rec("tanh", x)
        return MockTensor(x.dims)

    def transpose(self, input, perm, shuffle=False):
        self._rec("transpose", input, perm=perm, shuffle=shuffle)
        permuted = tuple(input.dims[p] for p in perm) if input.nDim == len(perm) else input.dims
        return MockTensor(permuted)

    def reshape(self, x, shape):
        self._rec("reshape", x, shape=shape)
        return MockTensor(shape)


def _install_fakes():
    taso = types.ModuleType("taso")
    taso.op_table = dict(FAKE_OP_TABLE)
    taso.get_padding_mode = lambda s: {"SAME": 0, "VALID": 1}[s]
    taso.get_activation_mode = lambda s: {"NONE": 0, "SIGMOID": 1, "RELU": 2, "TANH": 3}[s]
    taso.new_graph = lambda: MockGraph()
    sys.modules["taso"] = taso
    sys.modules["onnx"] = types.ModuleType("onnx")  # imported at module top, unused here


_install_fakes()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import reconstruct_generic as R  # noqa: E402


# ------------------------------- harness -------------------------------------

_passed = _failed = 0


def ok(msg):
    global _passed
    _passed += 1
    print(f"  PASS: {msg}")


def bad(msg):
    global _failed
    _failed += 1
    print(f"  FAIL: {msg}")


def check(cond, msg):
    ok(msg) if cond else bad(msg)


def write_model(nodes):
    """nodes: list of (guid, op, deps_str, params_str). Returns a temp path."""
    lines = []
    for guid, op, deps, params in nodes:
        lines += [str(guid), str(op), deps, params]
    fd, path = tempfile.mkstemp(suffix=".model")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def calls_of(graph, name):
    return [c for c in graph.calls if c[0] == name]


# ------------------------------- tests ---------------------------------------

def test_op_names_derivation():
    names = R.reconstruct_op_names(FAKE_OP_TABLE)
    check(names.get(19) == "Mul", "OP_MUL (int 19, absent from op_table) resolves to 'Mul'")
    check(names.get(17) == "Mul", "OP_EW_MUL (int 17) still 'Mul'")
    # a shifted enum (Enlarge NOT at Matmul+2) must fail loudly
    shifted = {18: "Matmul", 19: "Enlarge"}  # Enlarge at Matmul+1, not +2
    try:
        R.reconstruct_op_names(shifted)
        bad("shifted OpType layout should raise AssertionError")
    except AssertionError:
        ok("shifted OpType layout fails loudly (enum-shift guard)")


def test_scalar_mul_activation():
    # input(1,4) * scalar_weight()  -> real graph.mul, tensor (larger) first
    nodes = [
        (0, 0, "", "1,4"),        # Input
        (1, 1, "", ""),           # Weight, 0-D scalar
        (2, 19, "0:0,1:0", ""),   # OP_MUL: activation * scalar
    ]
    path = write_model(nodes)
    try:
        graph, _, _ = R.parse_and_build(
            path, {"scale": np.float32(2.0)}, {"1": ["scale"]})
    finally:
        os.unlink(path)
    muls = calls_of(graph, "mul")
    check(len(muls) == 1, "scalar mul emits exactly one graph.mul (native ONNX Mul)")
    if muls:
        a, b = muls[0][1]
        check(a.nDim == 2 and b.nDim == 0, "mul_larger_first puts the tensor (larger) operand first")
    check(len(calls_of(graph, "new_weight")) == 1, "only the scalar weight is materialized, mul is a live op")


def test_weight_times_weight_folds():
    # weight(1,4) * scalar_weight()  -> numpy fold, one extra new_weight, no graph.mul
    nodes = [
        (1, 1, "", ""),           # scalar weight
        (3, 1, "", "1,4"),        # vector weight
        (4, 17, "3:0,1:0", ""),   # OP_EW_MUL: weight * weight
    ]
    path = write_model(nodes)
    try:
        graph, _, _ = R.parse_and_build(
            path, {"scale": np.float32(3.0), "w": np.ones((1, 4), np.float32)},
            {"1": ["scale"], "3": ["w"]})
    finally:
        os.unlink(path)
    check(len(calls_of(graph, "mul")) == 0, "weight*weight Mul folds -- no live graph.mul")
    check(len(calls_of(graph, "new_weight")) == 3, "fold materializes the product as a 3rd weight")


def test_activation_transpose():
    nodes = [
        (0, 0, "", "1,4"),          # Input (activation)
        (5, 15, "0:0", "2,1,0"),    # Transpose ndim=2 perm=(1,0) of an ACTIVATION
    ]
    path = write_model(nodes)
    try:
        graph, _, _ = R.parse_and_build(path, {}, {})
    finally:
        os.unlink(path)
    tps = calls_of(graph, "transpose")
    check(len(tps) == 1, "activation transpose emits a live graph.transpose")
    if tps:
        check(tps[0][2].get("shuffle") is True, "transpose passes shuffle=True (TASO ctor asserts it)")
        check(tps[0][2].get("perm") == (1, 0), "transpose perm decoded from params")


def test_weight_transpose_still_folds():
    nodes = [
        (1, 1, "", "1,4"),          # weight
        (2, 15, "1:0", "2,1,0"),    # Transpose of a WEIGHT
    ]
    path = write_model(nodes)
    try:
        graph, _, _ = R.parse_and_build(path, {"w": np.ones((1, 4), np.float32)}, {"1": ["w"]})
    finally:
        os.unlink(path)
    check(len(calls_of(graph, "transpose")) == 0, "weight transpose still folds in numpy (no live op)")
    check(len(calls_of(graph, "new_weight")) == 2, "folded transpose materialized as a weight")


def test_sigmoid_tanh():
    nodes = [
        (0, 0, "", "1,4"),      # Input
        (1, 9, "0:0", ""),      # Sigmoid
        (2, 10, "1:0", ""),     # Tanh
    ]
    path = write_model(nodes)
    try:
        graph, _, _ = R.parse_and_build(path, {}, {})
    finally:
        os.unlink(path)
    check(len(calls_of(graph, "sigmoid")) == 1, "standalone Sigmoid emits graph.sigmoid")
    check(len(calls_of(graph, "tanh")) == 1, "standalone Tanh emits graph.tanh")


def test_unknown_op_reports_raw_int():
    nodes = [(0, 99, "", "")]  # 99 not in op_table
    path = write_model(nodes)
    try:
        R.parse_and_build(path, {}, {})
        bad("unknown op should raise NotImplementedError")
    except NotImplementedError as e:
        check("99" in str(e), "unknown op error names the raw op int (99)")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    for fn in [
        test_op_names_derivation, test_scalar_mul_activation, test_weight_times_weight_folds,
        test_activation_transpose, test_weight_transpose_still_folds, test_sigmoid_tanh,
        test_unknown_op_reports_raw_int,
    ]:
        print(f"== {fn.__name__} ==")
        fn()
    print("=" * 38)
    print(f"RECONSTRUCT-ARM TESTS: {_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
