#!/usr/bin/env python3
"""Regression test for the Transpose perm decode (PROBLEMATIC.md #6).

Background: commit fb0b3db fixed a Release-build bug where
`assert(op.ptr->get_int_parameter(attr, &ret))` never called `get_int_parameter`
under NDEBUG (assert's argument is unevaluated), leaving `ret` uninitialized --
so Transpose's perm (and Conv/Pool attrs) came back as garbage like [0,0] that
ONNX rejects, but ONLY in Release builds (the GPU config). Debug/assert-enabled
builds always worked. This test pins the decode two ways:

(a) pb2egg._decode_perm against a HARDCODED idx oracle. The idx values are computed
    independently from idx = sum_i perm[i]*N**(N-1-i) (transpose.cc's
    permutation_to_index), NOT read back from taso -- so it catches the encoder and
    decoder drifting together. This also guards #8's pb2egg decoder.
(b) taso `core` round-trip transpose(perm) -> get_operator_attr('perm'), if the core
    ext is importable. The current CPU build is assert-enabled (where the bug never
    manifested), so this corroborates the decode path but does not re-prove the
    Release fix -- fb0b3db's diff does that.

Run standalone or via NNs/tests/run_tests.sh (in-container). Exit 0 on success.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pb2egg import _decode_perm

# perm -> permutation_to_index(perm), hardcoded (NOT derived from taso's encoder):
#   idx = sum_i perm[i] * N**(N-1-i),  N = len(perm)
ORACLE = {
    (1, 0): 2,          # 1*2 + 0
    (0, 1): 1,          # 0*2 + 1
    (0, 2, 1): 7,       # 0*9 + 2*3 + 1
    (2, 1, 0): 21,      # 2*9 + 1*3 + 0
    (1, 2, 0): 15,      # 1*9 + 2*3 + 0
    (0, 1, 3, 2): 30,   # 0*64 + 1*16 + 3*4 + 2
}
fails = 0

for perm, idx in ORACLE.items():
    got = _decode_perm(idx, len(perm))
    if got != list(perm):
        print("  FAIL decode idx={} N={} -> {} want {}".format(idx, len(perm), got, list(perm))); fails += 1
# non-permutation indices must be rejected (None), not emitted as garbage
if _decode_perm(0, 2) is not None:   # 0 -> [0,0], not a permutation
    print("  FAIL idx 0 (N=2) should decode to None"); fails += 1
if _decode_perm(4, 2) is not None:   # 4 -> overflow for N=2
    print("  FAIL idx 4 (N=2) should decode to None"); fails += 1
print("  decode-oracle: {}".format("PASS" if fails == 0 else "FAIL"))

# (b) taso core round-trip -- optional (needs the taso core ext; no onnx required).
try:
    import importlib.util
    so = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                      "taso", "python", "taso", "core.cpython-314-x86_64-linux-gnu.so")
    spec = importlib.util.spec_from_file_location("core", so)
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    rt_fail = 0
    for dims, perm in [((3, 4), (1, 0)), ((2, 3, 4), (0, 2, 1)), ((2, 3, 4), (2, 1, 0))]:
        g = core.PyGraph(); inp = g.new_input(dims=dims); g.transpose(inp, perm, True)
        got = None
        for op in g.get_operator_list():
            try: got = tuple(g.get_operator_attr(op, "perm"))
            except Exception: pass
        if got != perm:
            print("  FAIL taso round-trip {} -> {}".format(perm, got)); rt_fail += 1
    fails += rt_fail
    print("  taso-roundtrip: {}".format("PASS" if rt_fail == 0 else "FAIL"))
except Exception as e:
    print("  taso-roundtrip: SKIP ({}: core ext not importable here)".format(type(e).__name__))

print("TRANSPOSE PERM TEST: {}".format("PASS" if fails == 0 else "FAIL"))
sys.exit(1 if fails else 0)
