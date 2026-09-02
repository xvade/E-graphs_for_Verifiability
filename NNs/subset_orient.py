#!/usr/bin/env python3
"""Orientation-aware refinement of subset_check.

The generator's same_via_subst dedup (generator.cc branches 2/4) matches a NEW transfer
against an existing one with LHS/RHS SWAPPED -- so it treats `A=>B` and `B=>A` as one
transfer and keeps a single orientation. The un-deduped baseline (RELAX_SUBST=1) keeps
both. tensat's rules_from_str (rewrites.rs:154) builds ONE directed Rewrite per line, so a
reverse orientation present in the baseline but absent from S is a genuinely-absent
directed rule. This script quantifies how much of the same-orientation "residue" from
subset_check is explained by S carrying the OPPOSITE orientation.

For each residue rule r=L=>R (in B', not a same-orientation var-instance of any S rule):
  reverse r to R=>L, canonicalize (prededup.canon), and test whether that canonical
  reverse is an instance of some S rule (same skeleton, well-defined var map). If yes, S
  carries r's equivalence in the opposite direction.

    subset_orient.py <S=subst_dedup.txt> <B=baseline_dedup_CURRENT.txt>
"""
import sys, re
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from prededup import canon

VAR = re.compile(r"\?input_\d+")

def skel_and_seq(line):
    seq = [int(m.group(0).rsplit("_", 1)[1]) for m in VAR.finditer(line)]
    return VAR.sub("?", line), tuple(seq)

def is_instance(vr, vs):
    m = {}
    for a, b in zip(vs, vr):
        if a in m:
            if m[a] != b:
                return False
        else:
            m[a] = b
    return True

def reverse(line):
    l, r = line.split("=>", 1)
    return canon(r + "=>" + l)

def load(path):
    return [x.strip() for x in open(path) if x.strip()]

def main():
    S = load(sys.argv[1]); B = load(sys.argv[2])
    Sset, Bset = set(S), set(B)
    S_by = defaultdict(list)
    for l in S:
        sk, sq = skel_and_seq(l); S_by[sk].append(sq)

    def subsumed_by_S(line):
        sk, v = skel_and_seq(line)
        return any(is_instance(v, vs) for vs in S_by.get(sk, ()))

    # same-orientation residue (identical to subset_check's "genuinely dropped")
    residue = [r for r in (Bset - Sset) if not subsumed_by_S(r)]
    rev_exact = rev_inst = truly_novel = 0
    novel_examples = []
    novel_all = []
    for r in residue:
        rr = reverse(r)
        if rr in Sset:
            rev_exact += 1
        elif subsumed_by_S(rr):
            rev_inst += 1
        else:
            truly_novel += 1
            novel_all.append(r)
            if len(novel_examples) < 15:
                novel_examples.append(r)
    # optional 3rd arg: dump ALL truly-novel rules (to measure their sound count via verify)
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as f:
            f.write("\n".join(novel_all) + "\n")
        print(f"[dumped {len(novel_all)} truly-novel rules -> {sys.argv[3]}]")
    print(f"same-orientation residue (subset_check 'dropped')     = {len(residue)}")
    print(f"  reverse orientation is EXACTLY in S                 = {rev_exact}")
    print(f"  reverse orientation is a var-INSTANCE of an S rule   = {rev_inst}")
    print(f"  TRULY novel (neither orientation covered by S)      = {truly_novel}")
    mm = [r for r in novel_examples if 'ewmax' in r or 'ewmin' in r]
    print(f"\ntruly-novel examples (up to 15; {len(mm)} of them min/max):")
    for r in novel_examples:
        tag = "  [min/max]" if ('ewmax' in r or 'ewmin' in r) else ""
        print("   ", r + tag)
    # min/max-only breakdown
    mmres = [r for r in residue if 'ewmax' in r or 'ewmin' in r]
    mm_rev = sum(1 for r in mmres if reverse(r) in Sset or subsumed_by_S(reverse(r)))
    print(f"\nmin/max residue: {len(mmres)}; reverse-covered by S: {mm_rev}; "
          f"truly novel: {len(mmres) - mm_rev}")

if __name__ == "__main__":
    main()
