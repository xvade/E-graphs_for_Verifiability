#!/usr/bin/env python3
"""Convert egg S-expr rules  (op a b)=>(op c d)  with ?input_N vars into the
whitespace-free functional  op(a,b)==op(c,d)  notation that tensat's
`-m verify` (parse_rules / equation.pest) consumes. Strips the leading ? on
vars (verify treats each input_N as an opaque ground constant).

Usage: sexpr_to_functional.py <in_sexpr.txt> <out_functional.txt>
"""
import sys

def tokenize(s):
    return s.replace("(", " ( ").replace(")", " ) ").split()

def parse(toks, pos):
    if toks[pos] == "(":
        op = toks[pos + 1]
        pos += 2
        args = []
        while toks[pos] != ")":
            node, pos = parse(toks, pos)
            args.append(node)
        return (op, args), pos + 1
    return (toks[pos], None), pos + 1

def emit(node):
    head, args = node
    if args is None:                      # atom
        return head.lstrip("?")           # ?input_1 -> input_1
    return "{}({})".format(head, ",".join(emit(a) for a in args))

def main():
    inp, outp = sys.argv[1], sys.argv[2]
    eqs, dropped = [], 0
    for line in open(inp):
        line = line.strip()
        if not line or "=>" not in line:
            continue
        lhs_s, rhs_s = line.split("=>", 1)
        lnode = parse(tokenize(lhs_s), 0)[0]
        rnode = parse(tokenize(rhs_s), 0)[0]
        # Skip degenerate sides that emit a bare atom: whitespace-free
        # concatenation (equation.pest has no WHITESPACE rule) would merge a
        # bare-atom RHS with the next rule's op-name into one token. Every kept
        # equation therefore ends in ')'.
        if lnode[1] is None or rnode[1] is None:
            dropped += 1
            continue
        eqs.append("{}=={}".format(emit(lnode), emit(rnode)))
    with open(outp, "w") as f:
        f.write("".join(eqs))
    print("converted {} rules -> {} (dropped {} bare-atom-sided)".format(len(eqs), outp, dropped))

if __name__ == "__main__":
    main()
