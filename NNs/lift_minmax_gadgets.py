#!/usr/bin/env python3
"""Automated min/max-gadget lifter: un-waste a ReLU net's min/max encoding.

Generalizes the hand-written `build_tll_lattice.py`. A ReLU MLP that computes
pairwise min/max (e.g. a TLL-compiled net) encodes each `max(a,b)` as a FOUR-ReLU
gadget -- `relu([a+b, -(a+b), b-a, a-b]) . [.5,-.5,.5,.5]` = `0.5(a+b)+0.5|a-b|` --
where HALF the ReLUs merely route the linear term `a+b` through the identity
`x = relu(x)-relu(-x)`. A verifier can't see those are linear, relaxes each as an
unstable ReLU, and the slack compounds through the tree (tll: 628/1020 unstable,
cert_ub 19.59). Re-expressing the SAME function with explicit Min/Max ops makes the
linear parts exact and each min/max one tight neuron (tll: ~90 unstable, 8.26). See
`TLL_RESULT.md` and the session record for the mechanism.

This tool recognizes the gadgets by their WEIGHT SIGNATURE + a numeric check (no
reliance on layer names), rewrites each to an ONNX `Min`/`Max` node, resolves the
one-hot routing layer's aliases so the lattice is a clean tree, and gates the result
numerically against the original. The Min/Max form then flows through the existing
pipeline (taso `load_onnx` maps Min/Max -> OP_EW_MIN/MAX; tensat reassociates).

Recognizes: a hidden ReLU layer's units grouped into quadruples that each compute
`min|max(a,b)` of two upstream values (verified by evaluation), fed by an affine base
and optional one-hot routing, in a single feed-forward chain.
Does NOT (yet) recognize: gadgets with learned nonzero biases that shift the breakpoint
(rejected by the numeric check, logged), non-quadruple min/max encodings, gadgets whose
two operands are themselves non-recognized nonlinear subgraphs, branched/residual graphs.
A model that fails tells you which assumption broke.

    python3 NNs/lift_minmax_gadgets.py <in.onnx> <out.onnx> [--eps E] [--x0 V]
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper

WZERO = 1e-6  # weight-support threshold


# ----------------------------- value DAG --------------------------------------
class Val:
    __slots__ = ("kind", "idx", "op", "l", "r")
    def __init__(self, kind, idx=None, op=None, l=None, r=None):
        self.kind, self.idx, self.op, self.l, self.r = kind, idx, op, l, r

def leaf(i):        return Val("leaf", idx=i)
def node(op, l, r): return Val("mm", op=op, l=l, r=r)


# ------------------------- parse ONNX -> affine/relu chain --------------------
def parse_chain(model):
    """Follow the single data path input->output, emitting ('affine', W, b) with
    out = in@W + b (MatMul[+Add] or Gemm), and ('relu',). Chain nets only."""
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    producer = {o: n for n in model.graph.node for o in n.output}
    inp = model.graph.input[0].name
    out = model.graph.output[0].name
    chain, t, seen = [], out, set()
    while t in producer and t not in seen:
        seen.add(t)
        n = producer[t]; chain.append(n)
        data = [x for x in n.input if x not in inits]
        if not data:
            break
        t = data[0]
    chain.reverse()

    layers, i = [], 0
    while i < len(chain):
        n = chain[i]
        if n.op_type == "Relu":
            layers.append(("relu", None, None)); i += 1
        elif n.op_type == "MatMul":
            W = next(inits[x] for x in n.input if x in inits)
            b = np.zeros(W.shape[1], np.float32)
            if i + 1 < len(chain) and chain[i + 1].op_type == "Add":
                b = next(inits[x] for x in chain[i + 1].input if x in inits).astype(np.float32)
                i += 1
            layers.append(("affine", W.astype(np.float32), b)); i += 1
        elif n.op_type == "Gemm":
            attrs = {a.name: a for a in n.attribute}
            W = next(inits[x] for x in n.input if x in inits)
            if attrs.get("transB") and attrs["transB"].i:
                W = W.T
            bs = [inits[x] for x in n.input if x in inits]
            b = bs[1].astype(np.float32) if len(bs) > 1 else np.zeros(W.shape[1], np.float32)
            layers.append(("affine", W.astype(np.float32), b)); i += 1
        elif n.op_type == "Add":
            # a standalone Add of a constant = a bias shift; fold it into the preceding
            # affine's bias (onnxsim/Gemm-fusion leaves these). Not a residual (that would
            # add two *computed* tensors -- a branch this chain walker doesn't handle).
            cst = [x for x in n.input if x in inits]
            if len(cst) == 1 and layers and layers[-1][0] == "affine":
                k, W, b = layers[-1]
                layers[-1] = (k, W, b + inits[cst[0]].astype(np.float32).reshape(-1))
                i += 1
            else:
                raise NotImplementedError("lifter: non-foldable Add (residual?) in the chain")
        elif n.op_type in ("Flatten", "Reshape", "Identity", "Squeeze", "Unsqueeze"):
            i += 1  # shape-only, no-op for a flat MLP
        else:
            raise NotImplementedError("lifter: unhandled op %r in the chain" % n.op_type)
    return layers


# -------------------------- gadget recognition --------------------------------
def classify_pair(w0, bb0, w1, bb1):
    """Does relu([a,b]@w0 + bb0)@w1 + bb1 compute min or max of (a,b)? Returns
    'min'/'max'/None by evaluating the 4-neuron sub-circuit on random inputs."""
    ab = (np.random.RandomState(0).randn(400, 2) * 4).astype(np.float32)
    outp = np.maximum(ab @ w0 + bb0, 0.0) @ w1 + bb1
    if np.allclose(outp, ab.max(1), atol=1e-4):
        return "max"
    if np.allclose(outp, ab.min(1), atol=1e-4):
        return "min"
    return None

def recognize_bank(frontier, W0, b0, W1, b1, log):
    """[in]--W0-->relu--W1-->[out]. Each output j must read exactly 4 hidden units
    spanning exactly 2 inputs (a,b), computing min|max(val_a,val_b). Returns the new
    frontier (Val nodes) or None if any output isn't a clean pairwise gadget."""
    n_out = W1.shape[1]
    new = [None] * n_out
    for j in range(n_out):
        quad = np.where(np.abs(W1[:, j]) > WZERO)[0]
        if len(quad) != 4:
            log.append("out %d: |hidden support|=%d != 4" % (j, len(quad))); return None
        ins = sorted(set(np.where(np.abs(W0[:, quad]).sum(1) > WZERO)[0].tolist()))
        if len(ins) != 2:
            log.append("out %d: spans %d inputs != 2" % (j, len(ins))); return None
        a, b = ins
        op = classify_pair(W0[[a, b]][:, quad], b0[quad], W1[quad, j], b1[j])
        if op is None:
            log.append("out %d: pair(%d,%d) not min/max (bias0=%s)" % (j, a, b, b0[quad])); return None
        new[j] = node(op, frontier[a], frontier[b])
    return new

def is_one_hot(W):
    """Routing matrix: every output column selects exactly one input (a single 1)."""
    col = np.abs(W) > WZERO
    return bool(np.all(col.sum(0) == 1) and np.all(np.isin(np.round(W[col], 4), [-1.0, 1.0])))


# --------------------------- lift: chain -> (base, root) ----------------------
def lift(layers):
    assert layers and layers[0][0] == "affine", "expected an affine base layer first"
    _, W_lin, b_lin = layers[0]
    frontier = [leaf(k) for k in range(W_lin.shape[1])]
    i, banks, aliases = 1, 0, 0
    while i < len(layers):
        kind = layers[i][0]
        nxt = layers[i + 1][0] if i + 1 < len(layers) else None
        if kind == "affine" and nxt == "affine":
            _, W, b = layers[i]
            if not is_one_hot(W):
                raise NotImplementedError(
                    "lifter: affine layer %d is neither a gadget input nor one-hot routing" % i)
            sel = np.argmax(np.abs(W) > WZERO, axis=0)
            frontier = [frontier[sel[j]] for j in range(W.shape[1])]
            aliases += 1; i += 1
        elif kind == "affine" and nxt == "relu":
            (_, W0, b0), (_, W1, b1) = layers[i], layers[i + 2]
            log = []
            nf = recognize_bank(frontier, W0, b0, W1, b1, log)
            if nf is None:
                raise NotImplementedError(
                    "lifter: layer %d looked like a gadget bank but isn't (%s)" % (i, log[:2]))
            frontier = nf; banks += 1; i += 3
        else:
            raise NotImplementedError("lifter: unexpected layer %d (%s->%s)" % (i, kind, nxt))
    assert len(frontier) == 1, "lattice did not reduce to a single output (%d left)" % len(frontier)
    print("  recognized: %d gadget banks, %d routing layers, base=%s" %
          (banks, aliases, W_lin.shape))
    return (W_lin, b_lin), frontier[0]


# --------------------------- emit torch -> ONNX -------------------------------
def to_onnx(base, root, in_dim, out_path):
    import torch
    W_lin, b_lin = base
    n_base = W_lin.shape[1]

    class Lifted(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(in_dim, n_base)
            with torch.no_grad():
                self.lin.weight.copy_(torch.tensor(W_lin.T))
                self.lin.bias.copy_(torch.tensor(b_lin))
        def forward(self, x):
            z = self.lin(x)
            memo = {}
            def ev(v):
                if v.kind == "leaf":
                    return z[:, v.idx:v.idx + 1]
                key = id(v)
                if key in memo:
                    return memo[key]
                a, b = ev(v.l), ev(v.r)
                res = torch.minimum(a, b) if v.op == "min" else torch.maximum(a, b)
                memo[key] = res
                return res
            return ev(root)

    m = Lifted().eval()
    x = torch.zeros(1, in_dim)
    torch.onnx.export(m, x, out_path, opset_version=11,
                      input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "N"}, "y": {0: "N"}},
                      dynamo=False)  # legacy TorchScript exporter (no onnxscript dep)
    return out_path


# ------------------------------- numeric gate ---------------------------------
def gate(orig_path, lifted_path, in_dim, x0, eps, n=512):
    import onnxruntime as ort
    lo, hi = x0 - eps, x0 + eps
    X = (np.random.RandomState(1).uniform(lo, hi, (n, in_dim))).astype(np.float32)
    def run(p):
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        return s.run(None, {s.get_inputs()[0].name: X})[0].reshape(n, -1)
    yo, yl = run(orig_path), run(lifted_path)
    return float(np.max(np.abs(yo - yl)))


def count_minmax(path):
    g = onnx.load(path).graph
    from collections import Counter
    c = Counter(n.op_type for n in g.node)
    return c.get("Min", 0), c.get("Max", 0)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    inp, outp = args[0], args[1]
    eps = float(sys.argv[sys.argv.index("--eps") + 1]) if "--eps" in sys.argv else 1.0
    x0 = float(sys.argv[sys.argv.index("--x0") + 1]) if "--x0" in sys.argv else 0.0
    model = onnx.load(inp)
    in_dim = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value or 2
    print("lifting %s (in_dim=%d)" % (inp, in_dim))
    layers = parse_chain(model)
    base, root = lift(layers)
    to_onnx(base, root, in_dim, outp)
    nmin, nmax = count_minmax(outp)
    d = gate(inp, outp, in_dim, x0, eps)
    print("  emitted: %d Min + %d Max nodes -> %s" % (nmin, nmax, outp))
    print("  numeric gate (eps=%.3g box): max|orig-lifted| = %.3e" % (eps, d))
    ok = d < 1e-5
    print("  RESULT:", "PASS (function preserved)" if ok else "FAIL (function changed!)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
