#!/usr/bin/env python3
"""Direct graph_subst.pb -> egg rule converter (Z3-free).

Replicates verify/read_rules.py's graph traversal (to_z3) but emits egg
S-expressions in tensat's converted.txt format instead of Z3 objects, so the
GENERATOR's rules (including the min/max/sub family) can drive tensat's --rules
without the Z3 verification detour.

Op-type ints follow the XFLOW enum the generator writes (== read_rules.py's
range(26)), extended with EW_SUB/MAX/MIN = 26/27/28 (the generator's xflow shim).
Only rules whose ops are ALL in CLEAN_OPS are emitted -- these have a faithful
1:1 egg arity (name <params> <inputs>). transpose/conv/pool/concat/split/enlarge/
constants are skipped: their egg form needs extra args the pest converter injects.

Usage: pb2egg.py <graph_subst.pb> <out_rules.txt> [--bidir]
Requires rules_pb2 (regenerate from taso/src/core/rules.proto with the container's
protoc; see companion invocation).

Soundness note: the generator's equivalences pass random-numeric testing, NOT Z3.
For piecewise-linear (min/max) ops random testing can pass false equivalences, so
downstream extractions MUST be numeric-checked (as this project already does).
"""
import sys, os, re

def vars_of(s):
    return set(re.findall(r'\?input_\d+', s))

# --- XFLOW OpType enum (range(26)) + appended SUB/MAX/MIN (generator's shim) ---
(OP_INPUT, OP_WEIGHT, OP_ANY, OP_CONV2D, OP_DROPOUT, OP_LINEAR, OP_POOL2D_MAX,
 OP_POOL2D_AVG, OP_RELU, OP_SIGMOID, OP_TANH, OP_BATCHNORM, OP_CONCAT, OP_SPLIT,
 OP_RESHAPE, OP_TRANSPOSE, OP_EW_ADD, OP_EW_MUL, OP_MATMUL, OP_MUL, OP_ENLARGE,
 OP_MERGE_GCONV, OP_CONSTANT_IMM, OP_CONSTANT_ICONV, OP_CONSTANT_ONE,
 OP_CONSTANT_POOL) = range(26)
OP_EW_SUB, OP_EW_MAX, OP_EW_MIN = 26, 27, 28

# --- PMParameter enum (read_rules.py) ---
(PM_OP_TYPE, PM_NUM_INPUTS, PM_NUM_OUTPUTS, PM_GROUP, PM_KERNEL_H, PM_KERNEL_W,
 PM_STRIDE_H, PM_STRIDE_W, PM_PAD, PM_ACTI, PM_NUMDIM, PM_AXIS, PM_PERM,
 PM_OUTSHUFFLE, PM_MERGE_GCONV_COUNT) = range(15)

# op -> (egg_name, [param_keys in egg order], input_arity)
operator_data = {
    OP_RELU:    ('relu',  [], 1),
    OP_EW_ADD:  ('ewadd', [], 2),
    OP_EW_MUL:  ('ewmul', [], 2),
    OP_EW_SUB:  ('ewsub', [], 2),
    OP_EW_MAX:  ('ewmax', [], 2),
    OP_EW_MIN:  ('ewmin', [], 2),
    OP_MATMUL:  ('matmul', [PM_ACTI], 2),
    OP_MUL:     ('smul',  [], 2),
}
# Only emit rules whose every op is convertible with faithful egg arity.
CLEAN_OPS = set(operator_data.keys())

def build(tensor, ops):
    if tensor.opId < 0:
        return "?input_{}".format(-tensor.opId)
    op = ops[tensor.opId]
    name, pkeys, in_ar = operator_data[op.type]
    params = {p.key: p.value for p in op.para}
    args = [str(params[k]) for k in pkeys]
    args += [build(x, ops) for x in op.input]
    return "({} {})".format(name, " ".join(args)) if args else "({})".format(name)

def rule_is_clean(rule):
    return all(o.type in CLEAN_OPS for o in rule.srcOp) and \
           all(o.type in CLEAN_OPS for o in rule.dstOp)

def main():
    inp, outp = sys.argv[1], sys.argv[2]
    bidir = "--bidir" in sys.argv
    sys.path.insert(0, os.path.dirname(os.path.abspath(inp)))
    import rules_pb2
    rules = rules_pb2.RuleCollection()
    rules.ParseFromString(open(inp, "rb").read())

    total = len(rules.rule)
    emitted, skipped_dirty, skipped_multi, skipped_ident, skipped_unbound = [], 0, 0, 0, 0
    seen = set()
    n_minmax = 0
    for rule in rules.rule:
        if len(rule.mappedOutput) != 1:
            skipped_multi += 1; continue
        if not rule_is_clean(rule):
            skipped_dirty += 1; continue
        mo = rule.mappedOutput[0]
        src_t = type('T', (), {'opId': mo.srcOpId, 'tsId': mo.srcTsId})()
        dst_t = type('T', (), {'opId': mo.dstOpId, 'tsId': mo.dstTsId})()
        try:
            lhs = build(src_t, rule.srcOp)
            rhs = build(dst_t, rule.dstOp)
        except (KeyError, IndexError):
            skipped_dirty += 1; continue
        if lhs == rhs:
            skipped_ident += 1; continue
        # egg requires vars(RHS) subset of vars(LHS) (no unbound RHS vars). Many
        # generator rules pair two relabelings of an input-independent expression
        # (RHS has a fresh var) -- emit only direction(s) that satisfy containment.
        cand = []
        if vars_of(rhs) <= vars_of(lhs):
            cand.append((lhs, rhs))
        if bidir and vars_of(lhs) <= vars_of(rhs):
            cand.append((rhs, lhs))
        if not cand:
            skipped_unbound += 1; continue
        for a, b in cand:
            line = "{}=>{}".format(a, b)
            if line not in seen:
                seen.add(line); emitted.append(line)
                if "ewmax" in line or "ewmin" in line:
                    n_minmax += 1
    with open(outp, "w") as f:
        f.write("\n".join(emitted))  # no trailing newline (tensat splits on \n)
    print("total rules in pb:        {}".format(total))
    print("skipped (multi-output):   {}".format(skipped_multi))
    print("skipped (non-clean ops):  {}".format(skipped_dirty))
    print("skipped (identity):       {}".format(skipped_ident))
    print("skipped (unbound RHS var):{}".format(skipped_unbound))
    print("emitted egg rules:        {}{}".format(len(emitted), " (bidir)" if bidir else ""))
    print("  of which min/max rules: {}".format(n_minmax))

if __name__ == '__main__':
    main()
