#!/usr/bin/env python3
"""Classify which CONVERTED multi-pattern rules fired, and what their real content is.

Input: a firing-probe log (multi_firing_probe.sh output) + the multi corpus that was
run + the single-pattern dedup corpus. For every rule[i] with cycle_ok>0 whose index is
OURS (i < pairs-in-file; predefined multi are appended after), we pull its two lines,
take the NON-identity (real) halves, alpha-canonicalize them (rename ?input_N by order of
first appearance -- the same canonicalization used to collapse the 14,913 emitted pairs to
115 distinct reals), and bucket each real half as:
    already-single-pattern : its alpha-canon form (either orientation) is in the dedup corpus
    novel                  : not in dedup (split-fusion multi-output, or min/max AC the
                             single-pattern lane never emitted)
Reports distinct rules fired separately from total applications (the (id,real) cross-product
inflates applications by |matching subgraphs|).

Usage: classify_fired_multi.py <probe.log> <multi_corpus.txt> <singlepattern_dedup.txt>
"""
import sys, re

VAR = re.compile(r'\?input_\d+')

def alpha_canon(expr, order=None):
    """Rename ?input_N by order of first appearance -> ?v0, ?v1, ...
    A rule's LHS and RHS SHARE variables, so pass a single `order` dict across
    both sides -- canonicalizing each side independently would collapse a
    reassociation a+(b+c)=>c+(b+a) into v0+(v1+v2)=>v0+(v1+v2), a false identity."""
    if order is None:
        order = {}
    def repl(m):
        v = m.group(0)
        if v not in order:
            order[v] = "?v{}".format(len(order))
        return order[v]
    return VAR.sub(repl, expr)

def canon_rule(line):
    lhs, rhs = line.split("=>")
    order = {}                       # one shared order spanning LHS then RHS
    return alpha_canon(lhs, order) + "=>" + alpha_canon(rhs, order)

def flip(cr):
    l, r = cr.split("=>")
    order = {}                       # shared order over the flipped orientation
    return alpha_canon(r, order) + "=>" + alpha_canon(l, order)

def main():
    logp, multip, dedupp = sys.argv[1], sys.argv[2], sys.argv[3]

    mlines = open(multip).read().split("\n")
    n_pairs = len(mlines) // 2

    # single-pattern dedup corpus, both orientations, alpha-canon
    dedup = set()
    for ln in open(dedupp).read().split("\n"):
        ln = ln.strip()
        if "=>" not in ln:
            continue
        cr = canon_rule(ln)
        dedup.add(cr)
        dedup.add(flip(cr))

    # parse fired indices from the log
    fired = {}   # idx -> cycle_ok count
    rule_re = re.compile(r'rule\[(\d+)\].*this_rule=\((\d+), *(\d+), *(\d+), *(\d+)\)')
    for ln in open(logp, errors="replace"):
        m = rule_re.search(ln)
        if not m:
            continue
        idx = int(m.group(1)); cyc = int(m.group(5))
        # this_rule counters are cumulative-per-rule and printed on every change,
        # so the LAST (== max, monotonic) value per index is the final count.
        # Summing occurrences would double-count the running snapshots.
        if cyc > 0:
            fired[idx] = max(fired.get(idx, 0), cyc)

    ours = {i: c for i, c in fired.items() if i < n_pairs}
    predef = {i: c for i, c in fired.items() if i >= n_pairs}

    print("multi corpus pairs (ours): {}".format(n_pairs))
    print("fired rule indices total : {}  (ours: {}, predefined: {})".format(
        len(fired), len(ours), len(predef)))
    print("total applications (ours): {}".format(sum(ours.values())))
    print()

    # classify the real content of each fired OUR rule
    distinct_real = {}   # canon real half -> {"idx":set, "apps":int, "split":bool}
    for idx, cyc in sorted(ours.items()):
        l0, l1 = mlines[2*idx], mlines[2*idx+1]
        for line in (l0, l1):
            lhs, rhs = line.split("=>")
            if lhs == rhs:
                continue                  # identity gating half, no content
            cr = canon_rule(line)
            d = distinct_real.setdefault(cr, {"idx": set(), "apps": 0, "split": "(split_" in rhs})
            d["idx"].add(idx); d["apps"] += cyc

    already = novel = 0
    print("===== distinct REAL rewrites among fired OUR rules =====")
    for cr, d in sorted(distinct_real.items(), key=lambda kv: -kv[1]["apps"]):
        in_dedup = cr in dedup
        tag = "SPLIT-FUSION" if d["split"] else ("already-single-pattern" if in_dedup else "NOVEL")
        if d["split"] or not in_dedup:
            novel += 1
        else:
            already += 1
        print("  [{:<22}] apps={:<5} rules={:<3} {}".format(tag, d["apps"], len(d["idx"]), cr))
    print()
    print("distinct real rewrites fired: {}  (already-single-pattern: {}, novel/split: {})".format(
        len(distinct_real), already, novel))

if __name__ == "__main__":
    main()
