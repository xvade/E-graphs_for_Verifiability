#!/usr/bin/env python3
"""Cheap SYNTACTIC pre-dedup of egg rules: collapse rules that are the SAME rule up to
exact match or a consistent variable RENAMING (alpha-equivalence). No saturation.

This is the first pass of the relaxed-generation pipeline: RELAX_SUBST makes the
generator re-emit the AC/cost-neutral families it used to quotient away, but also floods
out the input-class copies (the identical rewrite over x-, w-, i-inputs). Canonically
renaming each rule's variables to ?input_0, ?input_1, ... in first-appearance order (kept
prefix so z3_verify_egg.py still recognizes them) and deduping the result collapses those
copies to one representative while KEEPING genuinely distinct rewrites (comm vs assoc
canonicalize differently). The expensive AC-modulo reasoning is left to the downstream
saturation-based redundancy prune.

Usage: prededup.py <in_rules.txt> <out_rules.txt>
"""
import sys, re

VAR = re.compile(r"\?[A-Za-z0-9_]+")

def canon(rule: str) -> str:
    seen = {}
    def repl(m):
        v = m.group(0)
        if v not in seen:
            seen[v] = f"?input_{len(seen)}"
        return seen[v]
    return VAR.sub(repl, rule)

def main():
    inp, outp = sys.argv[1], sys.argv[2]
    lines = [l.strip() for l in open(inp) if l.strip()]
    seen = set()
    out = []
    for l in lines:
        c = canon(l)
        if c not in seen:
            seen.add(c)
            out.append(c)  # emit the canonical form (unambiguous, still ?input_N)
    with open(outp, "w") as f:
        f.write("\n".join(out))
    print(f"pre-dedup: {len(lines)} -> {len(out)} unique (exact + alpha-equivalence)")

if __name__ == "__main__":
    main()
