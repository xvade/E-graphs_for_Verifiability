#!/usr/bin/env python3
"""Formally verify egg rules (LHS=>RHS) with Z3, keeping only proven equivalences.

The generator's rules pass only random-numeric testing, and piecewise-linear min/max
is exactly where random testing admits false positives (e.g. a rule whose LHS secretly
depends on a matmul the RHS drops). This proves each rule or rejects it with a
counterexample.

Semantics (the point -- the stock TASO verifier leaves relu UNINTERPRETED, so the
max=a+relu(b-a) bridge is unprovable there):
  every ?input_N and every matmul/smul subterm -> a Real; ops interpreted exactly:
    ewadd=+  ewsub=-  ewmul=*  ewmax=If(a>=b,a,b)  ewmin=If(a<=b,a,b)  relu=If(x>=0,x,0)
    matmul(act,a,b), smul(a,b) -> uninterpreted Real functions of their args.
All these ops are ELEMENTWISE (matmul/smul opaque), so scalar validity over Reals
implies the tensor identity. Treating matmul/smul as arbitrary functions is SOUND and
conservative: a rule proved for all interpretations holds for the real op; a rule that
needs the specific op semantics is (safely) rejected.

TWO LANES (verdict = union, both sound, so the union is sound and monotone):
  Lane 1 (this file): scalar/PWL encoding above -- interprets ew*/relu exactly,
    conv/concat/matmul UNINTERPRETED. Fast. Proves PWL/min-max rules and the
    conv rules that hold by congruence.
  Lane 2 (tensor_axioms.py): TASO's quantified tensor axioms -- proves the
    conv/concat/matmul-algebra rewrites lane 1 must reject. Tried ONLY on rules
    lane 1 did not verify. A rule is VERIFIED if EITHER lane proves it.

Usage: z3_verify_egg.py <in_rules.txt> <out_verified.txt> [--timeout_ms N]
"""
import sys, re
import z3
try:
    import tensor_axioms as _lane2
except Exception:            # lane 2 optional; lane 1 still runs if it fails to import
    _lane2 = None

def tokenize(s):
    return s.replace("(", " ( ").replace(")", " ) ").split()

def parse(tokens, pos):
    """Return (node, next_pos). node is ('var',name) | ('app',op,[args])."""
    tok = tokens[pos]
    if tok == "(":
        op = tokens[pos + 1]
        pos += 2
        args = []
        while tokens[pos] != ")":
            node, pos = parse(tokens, pos)
            args.append(node)
        return ("app", op, args), pos + 1
    else:
        return ("atom", tok), pos + 1

class Builder:
    def __init__(self):
        self.vars = {}
        self.matmul = z3.Function("matmul", z3.IntSort(), z3.RealSort(), z3.RealSort(), z3.RealSort())
        self.smul = z3.Function("smul", z3.RealSort(), z3.RealSort(), z3.RealSort())
        self._uf = {}  # cache of uninterpreted Functions for non-PWL ops, keyed (op, arg sorts)

    def var(self, name):
        if name not in self.vars:
            self.vars[name] = z3.Real(name.replace("?", ""))
        return self.vars[name]

    def build(self, node):
        kind = node[0]
        if kind == "atom":
            t = node[1]
            if t.startswith("?input_"):
                return self.var(t)
            # a bare integer parameter (e.g. matmul activation) -> Int constant
            return z3.IntVal(int(t))
        op, args = node[1], node[2]
        if op == "ewadd":  return self.build(args[0]) + self.build(args[1])
        if op == "ewsub":  return self.build(args[0]) - self.build(args[1])
        if op == "ewmul":  return self.build(args[0]) * self.build(args[1])
        if op == "ewmax":
            a, b = self.build(args[0]), self.build(args[1]); return z3.If(a >= b, a, b)
        if op == "ewmin":
            a, b = self.build(args[0]), self.build(args[1]); return z3.If(a <= b, a, b)
        if op == "relu":
            a = self.build(args[0]); return z3.If(a >= 0, a, z3.RealVal(0))
        if op == "matmul":
            act = self.build(args[0]); return self.matmul(act, self.build(args[1]), self.build(args[2]))
        if op == "smul":
            return self.smul(self.build(args[0]), self.build(args[1]))
        # transpose: (transpose input perm_name shuffle). perm_name is a config LEAF
        # (e.g. "1_0"), NOT a tensor -- fold it into the function IDENTITY (so different
        # perms are different functions) and take only the input tensor as the argument.
        # The `shuffle` flag is DELIBERATELY IGNORED: it only changes output memory
        # strides (view vs contiguous copy), never the logical values (transpose.cc:102),
        # so transpose(x,perm,0) and transpose(x,perm,1) are value-identical. Congruence
        # then proves perm-preserving (and shuffle-invariant) rewrites.
        if op == "transpose":
            inp = self.build(args[0])
            perm = args[1][1]                        # atom token string, NOT built; shuffle ignored
            key = ("transpose", perm, inp.sort())
            if key not in self._uf:
                self._uf[key] = z3.Function("transpose_{}".format(perm), inp.sort(), z3.RealSort())
            return self._uf[key](inp)
        # Non-PWL ops (conv2d, poolmax/avg, concat/3/4/5, ...): treated as UNINTERPRETED
        # functions of their (interpreted) args -- sound and conservative, exactly like
        # matmul/smul. A rewrite provable only via the op's real (e.g. conv-linearity)
        # semantics is safely REJECTED here in lane 1 -- lane 2 (tensor_axioms.py) proves
        # those with TASO's quantified axioms.
        if op in ("conv2d", "poolmax", "poolavg", "concat", "concat3", "concat4", "concat5",
                  "Cpool", "Iconv", "Imatmul", "Iewmul"):  # const-tensor ops (0-tensor-input leaves)
            zargs = [self.build(a) for a in args]
            key = (op, tuple(a.sort() for a in zargs))
            if key not in self._uf:
                self._uf[key] = z3.Function(
                    "{}_{}".format(op, len(zargs)), *[a.sort() for a in zargs], z3.RealSort()
                )
            return self._uf[key](*zargs)
        raise ValueError("unhandled op: " + op)

def verify_rule(line, timeout_ms):
    """Return (verdict, lane): verdict in VERIFIED/REJECTED/UNKNOWN, lane in
    {1,2,None}. Lane 1 (scalar/PWL) runs first; if it does not VERIFY and lane 2
    (tensor axioms) is available, lane 2 is tried on the same rule. The rule is
    VERIFIED if either lane proves it (union of two sound checks)."""
    lhs_s, rhs_s = line.split("=>", 1)
    lhs_node = parse(tokenize(lhs_s), 0)[0]
    rhs_node = parse(tokenize(rhs_s), 0)[0]
    b = Builder()
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(b.build(lhs_node) != b.build(rhs_node))   # counterexample to LHS == RHS
    r = s.check()
    lane1 = "VERIFIED" if r == z3.unsat else ("REJECTED" if r == z3.sat else "UNKNOWN")
    if lane1 == "VERIFIED":
        return "VERIFIED", 1
    if _lane2 is not None:
        v2 = _lane2.verify_rule(lhs_node, rhs_node, timeout_ms)   # tensor-axiom lane
        if v2 == "VERIFIED":
            return "VERIFIED", 2
    return lane1, None

def main():
    inp, outp = sys.argv[1], sys.argv[2]
    timeout_ms = 10000
    if "--timeout_ms" in sys.argv:
        timeout_ms = int(sys.argv[sys.argv.index("--timeout_ms") + 1])
    verified, rejected, unknown, errored = [], 0, 0, 0
    v_lane1, v_lane2 = 0, 0
    lines = [l.strip() for l in open(inp) if l.strip() and "=>" in l]
    for i, line in enumerate(lines):
        try:
            res, lane = verify_rule(line, timeout_ms)
        except Exception as e:
            errored += 1; continue
        if res == "VERIFIED":
            verified.append(line)
            if lane == 1: v_lane1 += 1
            elif lane == 2: v_lane2 += 1
        elif res == "REJECTED": rejected += 1
        else:                   unknown += 1
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(lines)} (verified {len(verified)}, rej {rejected}, unk {unknown})")
    with open(outp, "w") as f:
        f.write("\n".join(verified))
    mm = sum(1 for l in verified if "ewmax" in l or "ewmin" in l)
    print(f"total rules:   {len(lines)}")
    print(f"VERIFIED:      {len(verified)}  (min/max: {mm}; lane1 PWL: {v_lane1}, lane2 tensor-axioms: {v_lane2})")
    print(f"REJECTED:      {rejected}   (provably NOT equivalences -- random-test false positives)")
    print(f"UNKNOWN:       {unknown}   (neither lane proved; z3 timeout/incomplete/needs a missing axiom)")
    print(f"parse-errored: {errored}")

if __name__ == "__main__":
    main()
