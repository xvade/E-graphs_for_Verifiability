#!/usr/bin/env python3
# Normalize a VNN-COMP ONNX so TASO's narrow/finicky importer can ingest it:
#   - fuse MatMul+Add -> Gemm (TASO skips bare "MatMul", only knows "Gemm")
#   - eliminate no-op Flatten (TASO's load_onnx reorder-assert trips on Flatten)
#   - onnx-simplify (constant-fold Sub/Div/Constant input-norm nodes etc.)
# Then verifies the normalized model is numerically identical to the original
# (~1e-5) so downstream rewrite/verify deltas aren't confounded by this step.
#
# Usage: normalize_for_taso.py <in.onnx> <out.onnx> [in_shape e.g. 1,784]
import sys
import numpy as np
import onnx
import onnxoptimizer
import onnxruntime as ort

def main():
    inp, outp = sys.argv[1], sys.argv[2]
    m = onnx.load(inp)
    try:
        from onnxsim import simplify
        m2, ok = simplify(m)
        if ok:
            m = m2
            print("onnxsim: OK")
    except Exception as e:
        print(f"onnxsim skipped: {e}")
    passes = ["eliminate_nop_flatten", "fuse_matmul_add_bias_into_gemm",
              "fuse_transpose_into_gemm", "eliminate_identity",
              "fuse_consecutive_transposes", "eliminate_deadend"]
    m = onnxoptimizer.optimize(m, passes)
    # Any remaining MatMul with a constant (initializer) second operand -> Gemm
    # (TASO skips bare MatMul but ingests Gemm). Gemm(A,B) = A@B with default
    # alpha=1,beta=0; requires 2D operands, which these FC-style MatMuls are.
    inits = {i.name for i in m.graph.initializer}
    n_conv = 0
    for node in m.graph.node:
        if node.op_type == "MatMul" and len(node.input) == 2 and node.input[1] in inits:
            node.op_type = "Gemm"
            n_conv += 1
    if n_conv:
        print(f"converted {n_conv} constant-weight MatMul -> Gemm")
    m = onnxoptimizer.optimize(m, ["fuse_matmul_add_bias_into_gemm", "eliminate_deadend"])
    onnx.save(m, outp)
    import collections
    print("normalized op counts:", dict(collections.Counter(n.op_type for n in m.graph.node)))

    # numeric equivalence check
    def sess(path):
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    s0, s1 = sess(inp), sess(outp)
    iname = s0.get_inputs()[0].name
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in s0.get_inputs()[0].shape]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    r0 = s0.run(None, {iname: x})[0]
    iname1 = s1.get_inputs()[0].name
    r1 = s1.run(None, {iname1: x})[0]
    diff = float(np.max(np.abs(r0 - r1)))
    print(f"max abs diff original vs normalized: {diff:.2e}")
    assert diff < 1e-4, f"normalization changed semantics (diff={diff})"
    print("OK: normalized model is numerically identical")

if __name__ == "__main__":
    main()
