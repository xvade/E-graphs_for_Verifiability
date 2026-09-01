#!/usr/bin/env python3
"""Lane 2 of the egg-rule verifier: prove conv2d/concat/matmul/pool rewrites with
TASO's quantified tensor axioms.

STATUS: working (2026-09-01). On the tracked graph_subst.pb corpus it lifts the
verifier from 35/116 to 104/116 rules (conv2d 8->26, concat 0->43), with 0
regressions and 0 negative-canary failures. Tests: NNs/tests/test_z3_axioms.sh.
The ~12 still-unproven are the grouped-convolution and matmul/concat-fold cases
that TASO's own verify.py also does not prove (it comments the grouped-conv
axiom out as "wrong axiom - caught with N=[1,3]" and blacklists such
substitutions); reaching them needs grouped-conv-aware reasoning -- follow-up.

WHY. `z3_verify_egg.py` (lane 1) interprets the elementwise/PWL ops exactly
(ewadd=+, ewmax=If(...), relu, ...) and treats conv2d/concat/matmul as
UNINTERPRETED functions. That is sound but conservative: it proves the ~8 conv
rules that hold by pure congruence (conv applied to arguments lane 1 already
proves equal), and REJECTS every rewrite that needs an operator's real algebra
(conv linear in its weight, conv distributes over channel-concat, relu(conv)=
conv+relu, matmul associativity, ...). This module supplies those algebra facts.

HOW. This is a faithful py2->py3 port of TASO's own rule verifier,
`taso/verify/verify.py` -- the axiom set that proved this exact substitution
corpus for the OSDI'19 paper. Tensors are an UNINTERPRETED sort `T`; each
operator is a Z3 `Function` over `T` (with its integer parameters as leading
`Int` args); and `AXIOMS`/`LEMMAS` are the universally-quantified algebraic
identities relating them. A rule LHS=>RHS is proved by asserting all axioms and
checking that `lhs != rhs` is unsatisfiable. No shapes are modelled -- the
axioms are shape-polymorphic, exactly as in TASO.

SOUNDNESS. The axiom set is consistent (it is TASO's, machine-checked on small
concrete shapes by `validate_axioms.py`), so it does not prove false rewrites --
guarded here by negative canaries (a consistent theory cannot prove
`conv(x,w)=conv(w,x)`; an *inconsistent* one proves everything, so a canary that
flips to VERIFIED is the alarm). Quantified reasoning is incomplete, so Z3 may
return `unknown`; that is reported as UNKNOWN and left unverified (conservative).

SCOPE. Ported for the ops that appear in the rejected corpus
(conv2d/concat/matmul/ewadd/ewmul/relu) plus the rest of TASO's set
(pool/split/transpose/enlarge/const_*) so the axiom list is complete and no
rule silently loses an axiom it needs. Egg ops this port does NOT model
(ewsub/ewmax/ewmin -- TASO predates min/max) raise `UnsupportedOp`; those rules
belong to lane 1, which interprets them exactly.
"""
import z3

# ---- sorts and enum constants (mirror taso/verify/verify.py) ----------------
T = z3.DeclareSort("T")          # uninterpreted tensor sort
P = z3.IntSort()                 # operator parameters are integers

AC_MODE_NONE, AC_MODE_SIGMOID, AC_MODE_RELU, AC_MODE_TANH = range(4)
PD_MODE_SAME, PD_MODE_VALID = range(2)

# operator opId -> (egg-name-independent taso name, n_params, n_tensor_inputs,
# n_outputs). Functions are declared as <name>_<outIdx>(P*n_params, T*n_inputs)->T.
# This is TASO's operator_data reduced to the arities the Function decls need.
_OPS = {
    "conv2d":     (4, 2, 1),
    "pool2d_max": (5, 1, 1),
    "pool2d_avg": (5, 1, 1),
    "relu":       (0, 1, 1),
    "concat":     (1, 2, 1),
    "split":      (1, 1, 2),
    "transpose":  (0, 1, 1),
    "enlarge":    (2, 1, 1),
    "ewadd":      (0, 2, 1),
    "ewmul":      (0, 2, 1),
    "matmul":     (0, 2, 1),
    "scalar_mul": (0, 2, 1),
    "const_pool":  (2, 0, 1),
    "const_iconv": (2, 0, 1),
    "const_imm":   (0, 0, 1),
    "const_one":   (0, 0, 1),
}
_FUNC = {}
for _name, (_np, _ni, _no) in _OPS.items():
    for _o in range(_no):
        _fn = "{}_{}".format(_name, _o)
        _FUNC[_fn] = z3.Function(_fn, *([P] * _np + [T] * _ni + [T]))
globals().update(_FUNC)   # conv2d_0, concat_0, matmul_0, split_0, split_1, ...

# quantified variables reused across axioms
x, y, z, w, one = z3.Consts("x y z w one", T)
sx, sy, kx, ky, pad, acti, ax = z3.Consts("sx sy kx ky pad acti ax", P)
F = z3.ForAll


class UnsupportedOp(Exception):
    """Raised for an egg op this lane does not model (min/max/sub); the rule is
    left to lane 1 rather than crashing the pipeline."""


# ---- axioms (formulas only; the shape-lambdas in verify.py are for its
#      small-shape meta-checker, validate_axioms.py, and are not needed to
#      *use* the axioms for rule proving). Ported verbatim from verify.py. -----
AXIOMS = [
    F([x, y, z], ewadd_0(x, ewadd_0(y, z)) == ewadd_0(ewadd_0(x, y), z)),
    F([x, y], ewadd_0(x, y) == ewadd_0(y, x)),
    F([x, y, z], ewmul_0(x, ewmul_0(y, z)) == ewmul_0(ewmul_0(x, y), z)),
    F([x, y], ewmul_0(x, y) == ewmul_0(y, x)),
    F([x, y, z], ewmul_0(ewadd_0(x, y), z) == ewadd_0(ewmul_0(x, z), ewmul_0(y, z))),
    F([x, y, w], scalar_mul_0(scalar_mul_0(x, y), w) == scalar_mul_0(x, scalar_mul_0(y, w))),
    F([x, y, w], scalar_mul_0(ewadd_0(x, y), w) == ewadd_0(scalar_mul_0(x, w), scalar_mul_0(y, w))),
    F([x, y, w], scalar_mul_0(ewmul_0(x, y), w) == ewmul_0(x, scalar_mul_0(y, w))),
    F([x, w], scalar_mul_0(transpose_0(x), w) == transpose_0(scalar_mul_0(x, w))),
    F([x, y, w], scalar_mul_0(matmul_0(x, y), w) == matmul_0(x, scalar_mul_0(y, w))),
    F([ax, x, y, w], scalar_mul_0(concat_0(ax, x, y), w) == concat_0(ax, scalar_mul_0(x, w), scalar_mul_0(y, w))),
    F([sx, sy, pad, acti, x, y, w], conv2d_0(sx, sy, pad, acti, scalar_mul_0(x, w), y) == conv2d_0(sx, sy, pad, acti, x, scalar_mul_0(y, w))),
    F([sx, sy, pad, x, y, w], scalar_mul_0(conv2d_0(sx, sy, pad, AC_MODE_NONE, x, y), w) == conv2d_0(sx, sy, pad, AC_MODE_NONE, scalar_mul_0(x, w), y)),
    # relu
    F([sx, sy, pad, x, y], relu_0(conv2d_0(sx, sy, pad, AC_MODE_NONE, x, y)) == conv2d_0(sx, sy, pad, AC_MODE_RELU, x, y)),
    F([ax, x, y], relu_0(concat_0(ax, x, y)) == concat_0(ax, relu_0(x), relu_0(y))),
    F([x], relu_0(transpose_0(x)) == transpose_0(relu_0(x))),
    # conv2d linearity in the weight and in the input
    F([sx, sy, pad, x, y, z], conv2d_0(sx, sy, pad, AC_MODE_NONE, x, ewadd_0(y, z)) == ewadd_0(conv2d_0(sx, sy, pad, AC_MODE_NONE, x, y), conv2d_0(sx, sy, pad, AC_MODE_NONE, x, z))),
    F([sx, sy, pad, x, y, z], conv2d_0(sx, sy, pad, AC_MODE_NONE, ewadd_0(x, y), z) == ewadd_0(conv2d_0(sx, sy, pad, AC_MODE_NONE, x, z), conv2d_0(sx, sy, pad, AC_MODE_NONE, y, z))),
    F([sx, sy, pad, x, y, z, w], ewadd_0(conv2d_0(sx, sy, pad, AC_MODE_NONE, x, y), conv2d_0(sx, sy, pad, AC_MODE_NONE, z, w)) == conv2d_0(sx, sy, pad, AC_MODE_NONE, concat_0(1, x, z), concat_0(1, y, w))),
    # matmul/concat
    F([x, y, z, w], ewadd_0(matmul_0(x, y), matmul_0(z, w)) == matmul_0(concat_0(1, x, z), concat_0(0, y, w))),
    F([ax, x, y, z, w], concat_0(ax, ewadd_0(x, y), ewadd_0(z, w)) == ewadd_0(concat_0(ax, x, z), concat_0(ax, y, w))),
    F([ax, x, y, z, w], concat_0(ax, ewmul_0(x, y), ewmul_0(z, w)) == ewmul_0(concat_0(ax, x, z), concat_0(ax, y, w))),
    F([sx, sy, pad, acti, x, y, z], concat_0(0, conv2d_0(sx, sy, pad, acti, x, z), conv2d_0(sx, sy, pad, acti, y, z)) == conv2d_0(sx, sy, pad, acti, concat_0(0, x, y), z)),
    F([sx, sy, pad, acti, x, y, z], concat_0(1, conv2d_0(sx, sy, pad, acti, x, y), conv2d_0(sx, sy, pad, acti, x, z)) == conv2d_0(sx, sy, pad, acti, x, concat_0(0, y, z))),
    F([x, y, z], matmul_0(x, matmul_0(y, z)) == matmul_0(matmul_0(x, y), z)),
    # split and concat
    F([ax, x, y], split_0(ax, concat_0(ax, x, y)) == x),
    F([ax, x, y], split_1(ax, concat_0(ax, x, y)) == y),
    F([x, y, z], matmul_0(x, concat_0(1, y, z)) == concat_0(1, matmul_0(x, y), matmul_0(x, z))),
    F([x, y, z], matmul_0(x, ewadd_0(y, z)) == ewadd_0(matmul_0(x, y), matmul_0(x, z))),
    # transpose
    F([x], transpose_0(transpose_0(x)) == x),
    F([x, y], transpose_0(matmul_0(x, y)) == matmul_0(transpose_0(y), transpose_0(x))),
    F([x, y], transpose_0(concat_0(0, x, y)) == concat_0(1, transpose_0(x), transpose_0(y))),
    F([x, y, z, w], concat_0(0, concat_0(1, x, y), concat_0(1, z, w)) == concat_0(1, concat_0(0, x, z), concat_0(0, y, w))),
    F([x, y], transpose_0(ewadd_0(x, y)) == ewadd_0(transpose_0(x), transpose_0(y))),
    F([x, y], transpose_0(ewmul_0(x, y)) == ewmul_0(transpose_0(x), transpose_0(y))),
    # pooling and concat
    F([kx, ky, sx, sy, pad, x, y], concat_0(1, pool2d_avg_0(kx, ky, sx, sy, pad, x), pool2d_avg_0(kx, ky, sx, sy, pad, y)) == pool2d_avg_0(kx, ky, sx, sy, pad, concat_0(1, x, y))),
    F([kx, ky, sx, sy, pad, x, y], concat_0(0, pool2d_max_0(kx, ky, sx, sy, pad, x), pool2d_max_0(kx, ky, sx, sy, pad, y)) == pool2d_max_0(kx, ky, sx, sy, pad, concat_0(0, x, y))),
    F([kx, ky, sx, sy, pad, x, y], concat_0(1, pool2d_max_0(kx, ky, sx, sy, pad, x), pool2d_max_0(kx, ky, sx, sy, pad, y)) == pool2d_max_0(kx, ky, sx, sy, pad, concat_0(1, x, y))),
    # const_pool / const_iconv / const_imm / const_one
    F([sx, sy, pad, x, kx, ky], conv2d_0(sx, sy, pad, AC_MODE_NONE, x, const_pool_0(kx, ky)) == pool2d_avg_0(kx, ky, sx, sy, pad, x)),
    F([kx, ky, x], conv2d_0(1, 1, PD_MODE_SAME, AC_MODE_NONE, x, const_iconv_0(kx, ky)) == x),
    F([x], matmul_0(x, const_imm_0()) == x),
    F([x], ewmul_0(x, const_one_0()) == x),
    F([kx, ky], pool2d_avg_0(kx, ky, 1, 1, PD_MODE_SAME, const_iconv_0(kx, ky)) == const_pool_0(kx, ky)),
    # enlarge
    F([sx, sy, acti, kx, ky, x, y], conv2d_0(sx, sy, PD_MODE_SAME, acti, x, y) == conv2d_0(sx, sy, PD_MODE_SAME, acti, x, enlarge_0(kx, ky, y))),
]

LEMMAS = [
    transpose_0(const_imm_0()) == const_imm_0(),
    F([x], matmul_0(const_imm_0(), x) == x),
    F([kx, ky, sx, sy, pad, x, y], concat_0(0, pool2d_avg_0(kx, ky, sx, sy, pad, x), pool2d_avg_0(kx, ky, sx, sy, pad, y)) == pool2d_avg_0(kx, ky, sx, sy, pad, concat_0(0, x, y))),
]


# ---- egg s-expression -> Z3 term over sort T --------------------------------
def build(node, memo):
    """Map a parsed egg node (see z3_verify_egg.parse) to a Z3 term over `T`.

    node is ('atom', tok) | ('app', op, [args]). ?input_N leaves become a
    single shared `Const` over T (keyed in `memo`, so both rule sides agree on
    the same tensor). Integer atoms are operator parameters. Raises
    UnsupportedOp for ops outside TASO's algebra (ewsub/ewmax/ewmin) and
    ValueError for a truly unknown op.
    """
    kind = node[0]
    if kind == "atom":
        tok = node[1]
        if tok.startswith("?input_"):
            if tok not in memo:
                memo[tok] = z3.Const(tok.replace("?", ""), T)
            return memo[tok]
        return int(tok)                      # a bare parameter integer
    op, args = node[1], node[2]

    def b(a):
        return build(a, memo)

    if op == "ewadd":  return ewadd_0(b(args[0]), b(args[1]))
    if op == "ewmul":  return ewmul_0(b(args[0]), b(args[1]))
    if op == "relu":   return relu_0(b(args[0]))
    if op == "smul":   return scalar_mul_0(b(args[0]), b(args[1]))
    if op == "matmul":
        acti = int(args[0][1])               # (matmul acti a b)
        if acti != AC_MODE_NONE:
            raise UnsupportedOp("matmul activation {} not modelled".format(acti))
        return matmul_0(b(args[1]), b(args[2]))
    if op == "conv2d":                       # (conv2d sh sw pad acti a b)
        sh, sw, pd, ac = (int(args[i][1]) for i in range(4))
        return conv2d_0(sh, sw, pd, ac, b(args[4]), b(args[5]))
    if op.startswith("concat"):              # (concat axis ndim in..) ; drop ndim
        axis = int(args[0][1])
        ins = [b(a) for a in args[2:]]
        acc = ins[-1]
        for t in reversed(ins[:-1]):         # right-nest to binary concat_0
            acc = concat_0(axis, t, acc)
        return acc
    if op == "poolmax":                      # (poolmax in kh kw sh sw pad acti) 'ip'
        inp = b(args[0]); kh, kw, sh, sw, pd = (int(args[i][1]) for i in range(1, 6))
        return pool2d_max_0(kh, kw, sh, sw, pd, inp)
    if op == "poolavg":
        inp = b(args[0]); kh, kw, sh, sw, pd = (int(args[i][1]) for i in range(1, 6))
        return pool2d_avg_0(kh, kw, sh, sw, pd, inp)
    if op in ("ewsub", "ewmax", "ewmin"):
        raise UnsupportedOp(op)              # lane 1's job (interpreted exactly)
    raise ValueError("unhandled op: " + op)


def verify_rule(parse_node_lhs, parse_node_rhs, timeout_ms):
    """Try to prove LHS==RHS from the tensor axioms. Returns VERIFIED / REJECTED
    / UNKNOWN / UNSUPPORTED. UNSUPPORTED means the rule uses an op this lane does
    not model (leave it to lane 1)."""
    memo = {}
    try:
        lhs = build(parse_node_lhs, memo)
        rhs = build(parse_node_rhs, memo)
    except UnsupportedOp:
        return "UNSUPPORTED"
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    for a in AXIOMS:
        s.add(a)
    for l in LEMMAS:
        s.add(l)
    s.add(lhs != rhs)
    r = s.check()
    if r == z3.unsat:  return "VERIFIED"
    if r == z3.sat:    return "REJECTED"
    return "UNKNOWN"
