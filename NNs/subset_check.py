#!/usr/bin/env python3
"""Subset check: did the generator's substitution-dedup (RELAX_SUBST off) DROP any
wanted rewrite relative to the un-deduped baseline (RELAX_SUBST=1)?

Both inputs are prededup.py outputs, i.e. already alpha-canonical (?input_0,?input_1,...
in first-appearance order), so set relations are exact string ops. But the GENERATOR's
same_via_subst dedup is SUBSUMPTION, not alpha-equivalence: it never checks the variable
map for injectivity, so it drops a transfer when it is a variable-identifying INSTANCE of
one already kept (e.g. add(x,x) is an instance of add(x,y) via y->x). Such an instance is
fully redundant in egg -- the general LHS matches everywhere the instance does. So B\S is
expected to be non-empty for a benign reason; the real question is:

  for every r in B\S, is r a var-to-var INSTANCE of some s in S (same op skeleton, and
  s's variable-equality partition is COARSER: vs[i]==vs[j] => vr[i]==vr[j])?

All subsumed  -> nothing lost, subst-dedup is strictly a cleaner upstream prune.
Any residue  -> genuinely dropped rules; listed (the only case that changes the verdict).
Headline run on the min/max family, since that is what matters for verifiability.

    subset_check.py <S=subst_dedup.txt> <B=baseline_dedup_CURRENT.txt>
"""
import sys, re

VAR = re.compile(r"\?input_\d+")

def skel_and_seq(line):
    """(skeleton with vars blanked, tuple of var-index ints in appearance order)."""
    seq = [int(m.group(0).rsplit("_", 1)[1]) for m in VAR.finditer(line)]
    return VAR.sub("?", line), tuple(seq)

def is_instance(vr, vs):
    """Is r an instance of s? Same skeleton assumed. r identifies (>=) as many var
    positions as s: wherever s equates two positions, r must too. i.e. r's partition
    refines s's. Equivalently the map s-var -> r-var is well defined."""
    m = {}
    for a, b in zip(vs, vr):          # a = s's var at this pos, b = r's var
        if a in m:
            if m[a] != b:
                return False
        else:
            m[a] = b
    return True

def load(path):
    return [l.strip() for l in open(path) if l.strip()]

def analyze(S_lines, B_lines, tag):
    S = set(S_lines); B = set(B_lines)
    S_only = S - B
    B_only = B - S
    # index S by skeleton for instance lookup
    from collections import defaultdict
    S_by_skel = defaultdict(list)
    for l in S:
        sk, sq = skel_and_seq(l)
        S_by_skel[sk].append(sq)
    subsumed, residue = [], []
    for r in B_only:
        sk, vr = skel_and_seq(r)
        if any(is_instance(vr, vs) for vs in S_by_skel.get(sk, ())):
            subsumed.append(r)
        else:
            residue.append(r)
    print(f"[{tag}]  |S|={len(S)}  |B|={len(B)}")
    print(f"[{tag}]  S\\B (in subst, not baseline) = {len(S_only)}   "
          "(MUST be 0: same enumeration, baseline emits every match)")
    print(f"[{tag}]  B\\S (in baseline, not subst) = {len(B_only)}")
    print(f"[{tag}]    of which var-to-var INSTANCES of a surviving S rule = {len(subsumed)} (benign, redundant in egg)")
    print(f"[{tag}]    GENUINELY DROPPED (not subsumed)                    = {len(residue)}")
    for r in residue[:12]:
        print(f"[{tag}]      DROPPED: {r}")
    return S_only, residue

def is_minmax(l):
    return ("ewmax" in l) or ("ewmin" in l)

def main():
    S = load(sys.argv[1]); B = load(sys.argv[2])
    print("=== FULL corpus ===")
    S_only_all, residue_all = analyze(S, B, "all")
    print("\n=== MIN/MAX family (verifiability-relevant) ===")
    analyze([l for l in S if is_minmax(l)], [l for l in B if is_minmax(l)], "minmax")
    # verdict
    print("\n=== VERDICT ===")
    if S_only_all:
        print("WARN: S has %d rules absent from baseline -- unexpected drift, investigate." % len(S_only_all))
    if not residue_all:
        print("CLEAN: every baseline rule absent from subst is a var-to-var instance of a "
              "surviving subst rule. Subst-dedup dropped NOTHING wanted (instances are "
              "redundant in egg). Subst-dedup is strictly a cleaner upstream prune.")
    else:
        print("REVIEW: %d baseline rules are genuinely dropped (not instances). Listed above." % len(residue_all))

if __name__ == "__main__":
    main()
