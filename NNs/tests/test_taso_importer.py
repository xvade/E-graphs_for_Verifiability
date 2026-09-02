#!/usr/bin/env python3
"""Env-independent tests for the TASO fork's ONNX-importer deltas.

The taso/ fork is part of our codebase (see MEMORY: forks-are-our-code), so its
importer changes get tests. taso can't be imported for real in-repo (the compiled
core.*.so is python-3.14/container-only, and that env's onnx is broken -- see
PROBLEMATIC.md #5), so this mocks `taso.core` and `onnx` in sys.modules before
importing the pure-python taso/__init__.py, then checks the two things our fork
changed and the "unregistered op is silently skipped" bug class they guard:

  * xf_operators REGISTRATIONS -- the dispatch table load_onnx keys on op.op_type.
    An op missing here is skipped, degenerating the graph (this is exactly how the
    MatMul-casing and Sigmoid bugs manifested: tll -> inputs+weights only;
    ffnnSIGMOID -> compute drops after the first Gemm).
  * The Sigmoid/Tanh/Gemm builder BEHAVIORS (bias handling, single-input activation).

    python3 NNs/tests/test_taso_importer.py   # exits nonzero on any failure
"""
import os
import sys
import types

# ---- mock taso.core + onnx BEFORE importing taso/__init__.py --------------------
# taso/__init__.py does `from .core import *` (a compiled extension we can't load
# here) and `import onnx` / `from onnx import ...`. Pre-registering both in
# sys.modules makes __init__ load with pure python only; it references no core
# symbol at module scope (core.PyGraph is only touched inside load_onnx, uncalled
# here), so an otherwise-empty mock core suffices.
_core = types.ModuleType("taso.core")
sys.modules["taso.core"] = _core

_onnx = types.ModuleType("onnx")
_onnx.helper = types.SimpleNamespace()
_onnx.TensorProto = types.SimpleNamespace()
_onnx.numpy_helper = types.SimpleNamespace(to_array=lambda x: x)
# _parse_attribute compares att.type against these; we never feed real attributes,
# so any distinct sentinels work.
_onnx.AttributeProto = types.SimpleNamespace(INT=1, INTS=2, FLOAT=3, STRING=4, TENSOR=5)
sys.modules["onnx"] = _onnx
sys.modules["onnx.helper"] = _onnx.helper

_TASO_PY = os.path.join(os.path.dirname(__file__), "..", "..", "taso", "python")
sys.path.insert(0, _TASO_PY)
import taso  # noqa: E402


# ------------------------------- harness -----------------------------------------
_passed = _failed = 0


def ok(m):
    global _passed
    _passed += 1
    print(f"  PASS: {m}")


def bad(m):
    global _failed
    _failed += 1
    print(f"  FAIL: {m}")


def check(c, m):
    ok(m) if c else bad(m)


class MockTensor:
    pass


class MockGraph:
    """Records builder calls so we can assert what each importer handler emits."""
    def __init__(self):
        self.calls = []

    def _rec(self, name, *a, **k):
        self.calls.append((name, a, k))
        return MockTensor()

    def __getattr__(self, name):
        return lambda *a, **k: self._rec(name, *a, **k)


def fake_op(inputs, attribute=None):
    return types.SimpleNamespace(input=list(inputs), attribute=list(attribute or []))


def names(graph):
    return [c[0] for c in graph.calls]


# ------------------------------- tests -------------------------------------------

def test_registrations_present():
    xf = taso.xf_operators
    # our fork's additions + the pre-existing ops whose absence caused degenerate imports
    for op in ["MatMul", "Matmul", "Sigmoid", "Tanh", "Gemm", "Add", "Relu",
               "Sub", "Max", "Min", "Conv", "Flatten", "Reshape"]:
        check(op in xf, f"xf_operators registers {op!r} (unregistered => load_onnx skips it)")
    check(xf.get("MatMul") is xf.get("Matmul"),
          "MatMul (capital) and Matmul alias the same handler (the casing fix)")


def test_sigmoid_tanh_single_input():
    for opname, call in [("Sigmoid", "sigmoid"), ("Tanh", "tanh")]:
        g = MockGraph()
        taso.xf_operators[opname](fake_op(["x"]), g, {"x": MockTensor()}, [])
        check(names(g) == [call], f"{opname} handler emits exactly graph.{call}")


def test_gemm_bias_and_plain():
    # 2-input Gemm: matmul only, no bias add
    g = MockGraph()
    taso.xf_operators["Gemm"](fake_op(["a", "b"]), g, {"a": MockTensor(), "b": MockTensor()}, [])
    check(names(g) == ["matmul"], "Gemm(A,B) -> matmul only (no spurious bias add)")
    # 3-input Gemm: matmul then bias add (the dropped-bias fix)
    g = MockGraph()
    taso.xf_operators["Gemm"](fake_op(["a", "b", "c"]),
                              g, {"a": MockTensor(), "b": MockTensor(), "c": MockTensor()}, [])
    check(names(g) == ["matmul", "add"], "Gemm(A,B,C) -> matmul + add (bias no longer dropped)")


def test_gemm_transpose_attrs():
    # transB=1 should transpose the second operand before matmul
    att = [types.SimpleNamespace(name="transB", type=_onnx.AttributeProto.INT, i=1)]
    g = MockGraph()
    taso.xf_operators["Gemm"](fake_op(["a", "b"], att),
                              g, {"a": MockTensor(), "b": MockTensor()}, [])
    check(names(g) == ["transpose", "matmul"],
          "Gemm transB=1 -> transpose(B) then matmul")
    check(g.calls[0][2].get("shuffle") is True or (1, 0) in g.calls[0][1] or
          g.calls[0][2].get("perm") == (1, 0) or (1, 0) in [a for a in g.calls[0][1]],
          "Gemm transpose uses a 2-D perm")


if __name__ == "__main__":
    for fn in [test_registrations_present, test_sigmoid_tanh_single_input,
               test_gemm_bias_and_plain, test_gemm_transpose_attrs]:
        print(f"== {fn.__name__} ==")
        fn()
    print("=" * 38)
    print(f"TASO-IMPORTER TESTS: {_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
