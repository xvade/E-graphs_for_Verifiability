#!/usr/bin/env python3
"""graph_subst .multi.pb -> tensat multi-pattern rule text converter.

The single-pattern converter (pb2egg.py) routes every rule with
`len(mappedOutput) != 1` to a `.multi.pb` sidecar, because such a rule has NO
single-pattern egg form: its dst graph produces several outputs. This script is
the deferred multi-pattern conversion pb2egg.py's header promised -- it turns
those saved rules into tensat's `--multi_rules` text format.

tensat multi format (main.rs:83, `MultiPatterns::with_rules`): EVERY TWO
CONSECUTIVE LINES form ONE multi-pattern rule. Each line is `LHS=>RHS` in the
same egg S-expression syntax as single-pattern rules. The rule fires only when
BOTH source patterns match with agreeing shared-variable bindings (the merged
subst), then BOTH dst patterns are built and unioned. So for a rule with two
mapped outputs mo[0], mo[1] we emit:
    line 2k   : build(src @ mo[k])  =>  build(dst @ mo[k])          for k in {0,1}
reusing pb2egg.build() verbatim -- the only extension is a `split` root on the
dst side (the multi-OUTPUT fusion the single-pattern build() can't express).

Split encoding (28 of 16,878 rules): the dst roots at a `split` op (OP_SPLIT,
NOUT=2, AXIS=k) whose input is the fused op over a `concat` of the two srcOps.
mo[k].dstTsId selects which split output -> `(split_{tsId} (split {axis} {inner}))`
(stock converted_multi.txt confirms this shape). Everything BELOW the split is a
normal clean op, so pb2egg.build() handles the inner subtree unchanged.

Apply-safety (the multi lane's real panic gate is check_pat, rewrites.rs:355):
the buildable set there is BROADER than pb2egg's single-pattern APPLY_SAFE_EGG_OPS
-- it additionally has arms for split/split_0/split_1/enlarge/sigmoid/tanh (all
confirmed by reading the match). concat3/4/5 are STILL absent -> they hit the
`other => todo!()` catch-all and PANIC tensat at apply time, so a pair whose
EITHER dst uses them is dropped. A pair is atomic: if either line is
unconvertible or apply-unsafe, or the union var-containment fails, the whole
pair is dropped and counted.

A pair whose BOTH lines are identity (X=>X / Y=>Y) is VACUOUS: each line matches
nodes that already exist, passes check_pat and the cycle filter trivially, and
adds zero e-nodes -- yet it still increments `cycle_ok`, so leaving these in makes
the firing funnel uninterpretable (10,655 of 16,878 rules are (id,id), most over
matmul/ewadd that a resnet HAS). We DROP vacuous pairs by default and count them;
`--keep-vacuous` retains them. (id,real) pairs are kept: the identity half is a
real gating precondition ("also require this pattern to exist").

Usage: pb2multi.py <in.multi.pb> <out_multi.txt> [--keep-vacuous]
"""
import sys, os

# Reuse pb2egg's parser table, build(), enums. pb2egg lives alongside this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb2egg
from pb2egg import build, vars_of, OP_SPLIT, PM_AXIS

# The multi lane's apply-safe egg ops = check_pat's arms (rewrites.rs:355-1036),
# read from source: pb2egg's single-pattern set PLUS the ops check_pat handles
# that the single-pattern applier does not. concat3/4/5 remain UNSAFE (no arm).
MULTI_APPLY_SAFE = pb2egg.APPLY_SAFE_EGG_OPS | {
    "split", "split_0", "split_1", "enlarge", "sigmoid", "tanh",
}


def is_multi_apply_safe(egg_line):
    return set(pb2egg._EGG_OP.findall(egg_line)) <= MULTI_APPLY_SAFE


def build_rhs(dst_t, dst_ops):
    """Build the dst (RHS) rooted at dst_t. Adds the split-output root that
    pb2egg.build() lacks; delegates every other op (and the whole subtree below
    a split) to pb2egg.build() unchanged."""
    op = dst_ops[dst_t.opId]
    if op.type == OP_SPLIT:
        params = {p.key: p.value for p in op.para}
        axis = params[PM_AXIS]               # KeyError => dropped, counted
        inner = build(op.input[0], dst_ops)  # the fused op over concat(srcs)
        return "(split_{} (split {} {}))".format(dst_t.tsId, axis, inner)
    return build(dst_t, dst_ops)


def _t(op_id, ts_id):
    return type('T', (), {'opId': op_id, 'tsId': ts_id})()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keep_vacuous = "--keep-vacuous" in sys.argv
    inp, outp = args[0], args[1]
    pb_dir = os.path.dirname(os.path.abspath(inp))
    sys.path.insert(0, pb_dir)
    import rules_pb2
    coll = rules_pb2.RuleCollection()
    coll.ParseFromString(open(inp, "rb").read())

    total = len(coll.rule)
    lines = []
    n_pairs = n_split_pairs = 0
    n_identity_half = 0
    drop_len2 = drop_samesrc = drop_dirty = drop_unsafe = drop_unbound = 0
    drop_vacuous = 0
    n_minmax_pairs = 0
    real_rhs = []   # non-identity RHS halves, for the dedup cross-check

    for rule in coll.rule:
        mo = rule.mappedOutput
        if len(mo) != 2:            # every rule in this file is len 2, but be strict
            drop_len2 += 1; continue
        if mo[0].srcOpId == mo[1].srcOpId:   # degenerate: both lines share one src root
            drop_samesrc += 1; continue

        pair = []
        ok = True
        has_split = False
        for k in (0, 1):
            src_t = _t(mo[k].srcOpId, mo[k].srcTsId)
            dst_t = _t(mo[k].dstOpId, mo[k].dstTsId)
            try:
                lhs = build(src_t, rule.srcOp)
                rhs = build_rhs(dst_t, rule.dstOp)
            except (KeyError, IndexError):
                ok = False; break
            if rule.dstOp[mo[k].dstOpId].type == OP_SPLIT:
                has_split = True
            pair.append((lhs, rhs))
        if not ok:
            drop_dirty += 1; continue

        # Union var-containment: every RHS var must be bound by SOME LHS pattern
        # in the pair (merged subst spans both src patterns), not just its own line.
        lhs_vars = vars_of(pair[0][0]) | vars_of(pair[1][0])
        rhs_vars = vars_of(pair[0][1]) | vars_of(pair[1][1])
        if not (rhs_vars <= lhs_vars):
            drop_unbound += 1; continue

        # Apply-safety is per PAIR: drop if EITHER dst would panic tensat at apply.
        if not (is_multi_apply_safe(pair[0][1]) and is_multi_apply_safe(pair[1][1])):
            drop_unsafe += 1; continue

        # Vacuous pair: BOTH lines are identity -> asserts nothing, adds no e-node,
        # yet would inflate cycle_ok. Drop by default (see header); --keep-vacuous keeps.
        both_identity = (pair[0][0] == pair[0][1]) and (pair[1][0] == pair[1][1])
        if both_identity and not keep_vacuous:
            drop_vacuous += 1; continue

        for lhs, rhs in pair:
            if lhs == rhs:
                n_identity_half += 1
            else:
                real_rhs.append(rhs)
            lines.append("{}=>{}".format(lhs, rhs))
        n_pairs += 1
        if has_split:
            n_split_pairs += 1
        if any("ewmax" in l or "ewmin" in l for l in (pair[0][1], pair[1][1])):
            n_minmax_pairs += 1

    with open(outp, "w") as f:
        f.write("\n".join(lines))    # no trailing newline (tensat pairs consecutive lines)

    # side file: the non-identity RHS halves, for the "already single-pattern?" check
    real_path = outp[:-4] + ".realhalves.txt" if outp.endswith(".txt") else outp + ".realhalves.txt"
    with open(real_path, "w") as f:
        f.write("\n".join(real_rhs))

    print("total multi rules in pb:      {}".format(total))
    print("emitted pairs:                {}  ({} lines)".format(n_pairs, len(lines)))
    print("  of which split-fusion:      {}".format(n_split_pairs))
    print("  identity halves (X=>X):     {}  ({:.0%} of {} lines)".format(
        n_identity_half, (n_identity_half / len(lines)) if lines else 0, len(lines)))
    print("  min/max-bearing pairs:      {}".format(n_minmax_pairs))
    print("dropped (vacuous X=>X pair):  {}{}".format(
        drop_vacuous, "  [--keep-vacuous to retain]" if not keep_vacuous else " (KEPT)"))
    print("dropped (mappedOutput!=2):    {}".format(drop_len2))
    print("dropped (same src both out):  {}".format(drop_samesrc))
    print("dropped (unconvertible op):   {}".format(drop_dirty))
    print("dropped (unbound RHS var):    {}".format(drop_unbound))
    print("dropped (apply-unsafe dst):   {}".format(drop_unsafe))
    print("real (non-identity) RHS ->    {}".format(real_path))


if __name__ == '__main__':
    main()
