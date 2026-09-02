#!/usr/bin/env python3
"""Real (non-mock) TASO ingestion tests — run in the working taso+onnx env.

Unlike test_taso_importer.py (which mocks taso to test dispatch logic
env-independently), this drives the REAL compiled taso through onnx→taso ingestion
and asserts the built graph is non-degenerate. It only runs where taso+onnx co-exist:

    bash NNs/tests/in_taso_env.sh NNs/tests/test_ingest_taso.py

Exits nonzero on any failure.
"""
import sys
from collections import Counter

import taso  # noqa: E402  (resolved by in_taso_env.sh's PYTHONPATH)
import onnx  # noqa: E402

REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
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


def op_counts(onnx_path):
    """Ingest an ONNX model and return the taso graph's op-type histogram
    (via get_operator_list/type — avoids export_op, which lacks some cases)."""
    g = taso.load_onnx(onnx_path)
    ops = g.get_operator_list()
    return Counter(g.get_operator_type(op) for op in ops), len(ops)


def test_ffnnsigmoid_non_degenerate():
    """The Sigmoid-registration fix: ffnnSIGMOID (6x200 MNIST) must ingest with all
    6 Sigmoids + 7 Gemm-derived Matmuls, not degenerate to inputs+weights (which is
    what an unregistered Sigmoid produced — every activation silently skipped)."""
    m = f"{REPO}/NNs/candidate_models/eran2021_sigmoid/ffnnSIGMOID_Point_6x200.onnx"
    cnt, total = op_counts(m)
    print("    ffnnSIGMOID taso op_counts:", dict(cnt))
    check(cnt.get("Sigmoid", 0) == 6, "ffnnSIGMOID ingests all 6 Sigmoids (was 0 pre-fix)")
    check(cnt.get("Matmul", 0) == 7, "ffnnSIGMOID ingests all 7 Gemm→Matmul layers")
    check(cnt.get("Add", 0) == 7, "ffnnSIGMOID keeps all 7 Gemm biases (bias fix)")
    check(total > 20, f"graph is non-degenerate ({total} ops, not inputs+weights)")


if __name__ == "__main__":
    for fn in [test_ffnnsigmoid_non_degenerate]:
        print(f"== {fn.__name__} ==")
        fn()
    print("=" * 40)
    print(f"TASO-INGEST TESTS: {_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
