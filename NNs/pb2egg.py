#!/usr/bin/env python3
"""Direct graph_subst.pb -> egg rule converter (Z3-free).

Replicates verify/read_rules.py's graph traversal (to_z3) but emits egg
S-expressions in tensat's converted.txt format instead of Z3 objects, so the
GENERATOR's rules (including the min/max/sub family) can drive tensat's --rules
without the Z3 verification detour.

Op-type ints follow the XFLOW enum the generator writes (== read_rules.py's
range(26)), extended with EW_SUB/MAX/MIN = 26/27/28 (the generator's xflow shim).
Two filters gate emission:
  1. PARSE-emittable (CLEAN_OPS): the op has a faithful 1:1 egg arity. Covers
     ew*/relu/matmul/smul, conv2d/pool/concat (tier-1), transpose (perm decoded
     from PM_PERM), and the const_* family (Cpool/Iconv/Imatmul/Iewmul). Still
     un-emittable: enlarge (kernel-based pb vs ref-input-based egg -- a semantic
     mismatch) and split (multi-OUTPUT -> the .multi.pb multi-pattern lane).
  2. APPLY-safe (DEFAULT): the op is one tensat can BUILD during saturation
     (rewrites.rs). A rule using any other op parses and Z3-verifies but PANICS
     tensat when it applies (`todo!()`), so by default such rules are dropped
     (skipped_unapplicable). `--emit-unapplicable` keeps them for consumers that
     never apply rules (Z3 corpus studies). Apply-safe set: see APPLY_SAFE_EGG_OPS.
     Currently apply-UNsafe (gated by default): smul, poolmax/avg, transpose,
     const_*, concat3/4/5. See PROBLEMATIC.md #8 and docs/ADD_AN_OP.md.

Usage: pb2egg.py <graph_subst.pb> <out_rules.txt> [--bidir] [--multi-out <path.pb>]
Requires rules_pb2 (regenerate from taso/src/core/rules.proto with the container's
protoc; see companion invocation).

Multi-output rules (len(mappedOutput) != 1) have NO single-pattern egg form -- their dst
graph produces several outputs (typically via a `split`, whose two outputs are selected by
split_0/split_1). They belong to tensat's multi-pattern lane (PRE_DEFINED_MULTI), not the
single-pattern --rules file. Rather than DROP them, we save the exact source rules to a
filtered RuleCollection protobuf (`--multi-out`, default `<out_rules base>.multi.pb`) so the
multi-pattern conversion can be built against them later without re-running the generator.

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

# op -> (egg_name, [param_keys], order)  where order is the child layout tensat's
# `Mdl` define_language expects, verified against model.rs make() AND `tensat -m
# parse_check`:  'pi' = params (in the listed key order) then inputs; 'ip' = input(s)
# then params. Different ops genuinely differ (conv2d params-first; pool INPUT-first),
# so the order is encoded per-op here rather than assumed in build().
operator_data = {
    OP_RELU:    ('relu',  [], 'pi'),
    OP_EW_ADD:  ('ewadd', [], 'pi'),
    OP_EW_MUL:  ('ewmul', [], 'pi'),
    OP_EW_SUB:  ('ewsub', [], 'pi'),
    OP_EW_MAX:  ('ewmax', [], 'pi'),
    OP_EW_MIN:  ('ewmin', [], 'pi'),
    OP_MATMUL:  ('matmul', [PM_ACTI], 'pi'),         # (matmul acti in1 in2)
    OP_MUL:     ('smul',  [], 'pi'),
    # --- Tier-1 full-op coverage: parametrized SINGLE-output ops. This is the fix for
    #     the non-clean drop that made conv/matmul models look inert. ---
    OP_CONV2D:     ('conv2d', [PM_STRIDE_H, PM_STRIDE_W, PM_PAD, PM_ACTI], 'pi'),  # (conv2d sh sw pad acti input weight)
    OP_POOL2D_MAX: ('poolmax', [PM_KERNEL_H, PM_KERNEL_W, PM_STRIDE_H, PM_STRIDE_W, PM_PAD, PM_ACTI], 'ip'),  # (poolmax input kh kw sh sw pad acti)
    OP_POOL2D_AVG: ('poolavg', [PM_KERNEL_H, PM_KERNEL_W, PM_STRIDE_H, PM_STRIDE_W, PM_PAD, PM_ACTI], 'ip'),
    OP_CONCAT:     ('concat', [PM_AXIS, PM_NUMDIM], 'pi'),  # (concat axis ndim in..); egg name -> concat/concat3/4/5 by input count
    OP_TRANSPOSE:  ('transpose', [], 'special'),  # (transpose input perm_name shuffle); perm decoded from PM_PERM, see build()
    # Constant-tensor ops (0 tensor inputs). egg names differ from the pb enum:
    #   const_pool -> Cpool(kh,kw) [avg-pool kernel]; const_iconv -> Iconv(kh,kw)
    #   [identity conv kernel]; const_imm -> Imatmul [identity matrix]; const_one ->
    #   Iewmul [all-ones]. The generic build() emits nullary ops as "(Imatmul)".
    OP_CONSTANT_POOL:  ('Cpool',   [PM_KERNEL_H, PM_KERNEL_W], 'pi'),
    OP_CONSTANT_ICONV: ('Iconv',   [PM_KERNEL_H, PM_KERNEL_W], 'pi'),
    OP_CONSTANT_IMM:   ('Imatmul', [], 'pi'),
    OP_CONSTANT_ONE:   ('Iewmul',  [], 'pi'),
    # Tier-2 (still dropped, tracked): reshape (0 occurrences in every corpus seen --
    # not present), enlarge (pb is KERNEL-based (PM_KERNEL_H/W + 1 input) but tensat's
    # Enlarge is REF-INPUT-based (2 tensor inputs) -- a semantic mismatch needing graph
    # context, ~8k rules deferred), and split (multi-OUTPUT -- routed to the
    # multi-pattern .multi.pb, not a single pattern). See PROBLEMATIC.md #8.
}
def _decode_perm(perm_idx, numdim):
    """Invert transpose.cc's permutation_to_index (idx = sum_i perm[i]*numdim**(numdim-1-i)):
    read numdim base-`numdim` digits, most-significant first. Returns the perm list, or
    None if it is not a permutation of range(numdim) (so a mis-decode can't emit garbage)."""
    if numdim <= 0:
        return None
    digits, x = [], perm_idx
    for _ in range(numdim):
        digits.append(x % numdim); x //= numdim
    if x != 0:
        return None                       # index too large for numdim
    perm = digits[::-1]
    return perm if sorted(perm) == list(range(numdim)) else None
# Only emit rules whose every op is convertible with faithful egg arity.
CLEAN_OPS = set(operator_data.keys())

def build(tensor, ops):
    if tensor.opId < 0:
        return "?input_{}".format(-tensor.opId)
    op = ops[tensor.opId]
    name, pkeys, order = operator_data[op.type]      # KeyError => non-clean op => rule skipped
    params = {p.key: p.value for p in op.para}
    if op.type == OP_TRANSPOSE:                       # (transpose input perm_name shuffle)
        numdim, permidx = params.get(PM_NUMDIM), params.get(PM_PERM)
        if numdim is None or permidx is None:
            raise KeyError("transpose missing NUMDIM/PERM")
        perm = _decode_perm(permidx, numdim)
        if perm is None:
            raise KeyError("transpose bad perm idx={} numdim={}".format(permidx, numdim))
        perm_name = "_".join(str(p) for p in perm)   # e.g. [1,0] -> "1_0"; a Name leaf in tensat
        shuffle = params.get(PM_OUTSHUFFLE, 0)
        return "(transpose {} {} {})".format(build(op.input[0], ops), perm_name, shuffle)
    pargs = [str(params[k]) for k in pkeys]          # KeyError (missing param) => rule skipped
    iargs = [build(x, ops) for x in op.input]
    if op.type == OP_CONCAT:                          # tensat: concat/concat3/concat4/concat5 by input count
        if len(iargs) not in (2, 3, 4, 5):
            raise KeyError("concat arity {} unsupported".format(len(iargs)))
        name = "concat" if len(iargs) == 2 else "concat{}".format(len(iargs))
    args = pargs + iargs if order == "pi" else iargs + pargs
    return "({} {})".format(name, " ".join(args)) if args else "({})".format(name)

# Ops tensat's APPLY path (rewrites.rs) can actually build during saturation. A rule
# whose egg form uses any op OUTSIDE this set parses (parse_check) and Z3-verifies fine
# but PANICS tensat at application time (rewrites.rs `other => todo!()`), so it is unsafe
# to feed to `tensat --rules`. Confirmed empirically (2-iter saturation probes):
# matmul/ewadd/... saturate; smul/poolmax/poolavg/transpose/Cpool/Iconv/Imatmul/Iewmul/
# concat3/4/5 panic. By DEFAULT pb2egg emits only apply-safe rules; `--emit-unapplicable`
# emits the full parse-valid set for consumers that never apply rules (e.g. Z3 corpus
# studies via z3_verify_egg.py). See PROBLEMATIC.md #8 and docs/ADD_AN_OP.md.
# NOTE: "concat" (binary) is apply-safe; concat3/concat4/concat5 are NOT (only Mdl::Concat
# has an apply arm), so they are excluded here by name.
APPLY_SAFE_EGG_OPS = {
    "relu", "ewadd", "ewmul", "ewsub", "ewmax", "ewmin", "matmul", "conv2d", "concat",
    "Iewmul",   # const all-ones: tensat make/apply resolve it via its ewmul consumer
}
_EGG_OP = re.compile(r"\(([A-Za-z][A-Za-z0-9_]*)")   # op name = token right after "("

def is_apply_safe(egg_line):
    """True iff every op in the egg rule is one tensat can build at apply time (so the
    rule can be fed to `tensat --rules` without a todo!() panic)."""
    return set(_EGG_OP.findall(egg_line)) <= APPLY_SAFE_EGG_OPS

def rule_is_clean(rule):
    return all(o.type in CLEAN_OPS for o in rule.srcOp) and \
           all(o.type in CLEAN_OPS for o in rule.dstOp)

def main():
    inp, outp = sys.argv[1], sys.argv[2]
    bidir = "--bidir" in sys.argv
    emit_unapplicable = "--emit-unapplicable" in sys.argv  # keep tensat-unapplicable rules (Z3 studies)
    if "--multi-out" in sys.argv:
        multi_out = sys.argv[sys.argv.index("--multi-out") + 1]
    else:  # default sidecar: strip a trailing .txt, append .multi.pb
        multi_out = (outp[:-4] if outp.endswith(".txt") else outp) + ".multi.pb"
    sys.path.insert(0, os.path.dirname(os.path.abspath(inp)))
    import rules_pb2
    rules = rules_pb2.RuleCollection()
    rules.ParseFromString(open(inp, "rb").read())

    total = len(rules.rule)
    emitted, skipped_dirty, skipped_multi, skipped_ident, skipped_unbound = [], 0, 0, 0, 0
    skipped_unapplicable = 0
    multi_coll = rules_pb2.RuleCollection()   # multi-output rules, saved not dropped
    seen = set()
    n_minmax = 0
    for rule in rules.rule:
        if len(rule.mappedOutput) != 1:
            skipped_multi += 1
            multi_coll.rule.add().CopyFrom(rule)   # preserve the exact source rule
            continue
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
            if not emit_unapplicable and not (is_apply_safe(a) and is_apply_safe(b)):
                skipped_unapplicable += 1; continue   # would panic tensat at apply time
            if line not in seen:
                seen.add(line); emitted.append(line)
                if "ewmax" in line or "ewmin" in line:
                    n_minmax += 1
    with open(outp, "w") as f:
        f.write("\n".join(emitted))  # no trailing newline (tensat splits on \n)
    if skipped_multi:                # save (don't drop) multi-output rules for the multi-pattern lane
        with open(multi_out, "wb") as f:
            f.write(multi_coll.SerializeToString())
    print("total rules in pb:        {}".format(total))
    print("saved multi-output -> {}: {}".format(multi_out, skipped_multi) if skipped_multi
          else "saved multi-output:       0 (none present)")
    print("skipped (multi-output):   {}".format(skipped_multi))
    print("skipped (non-clean ops):  {}".format(skipped_dirty))
    print("skipped (identity):       {}".format(skipped_ident))
    print("skipped (unbound RHS var):{}".format(skipped_unbound))
    print("skipped (tensat-unapplicable ops): {}{}".format(
        skipped_unapplicable, " [--emit-unapplicable to keep]" if skipped_unapplicable else ""))
    print("emitted egg rules:        {}{}".format(len(emitted), " (bidir)" if bidir else ""))
    print("  of which min/max rules: {}".format(n_minmax))

if __name__ == '__main__':
    main()
