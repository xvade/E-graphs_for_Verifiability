#!/usr/bin/env python3
"""Diagnose the 'genuinely dropped' residue: for each residue rule r (in B' not S, not a
var-instance of any S rule), find same-skeleton rules in S and B', and test instance
relations both directions. Answers: does the more-general survivor exist in S? in B' only?
nowhere? Is subst keeping the LESS general (reverse instance)?"""
import sys, re
from collections import defaultdict
VAR = re.compile(r"\?input_\d+")

def sk_seq(line):
    seq = [int(m.group(0).rsplit("_",1)[1]) for m in VAR.finditer(line)]
    return VAR.sub("?", line), tuple(seq)

def instance(vr, vs):  # r is instance of s (s more general): map s-var->r-var well-defined
    m={}
    for a,b in zip(vs,vr):
        if a in m:
            if m[a]!=b: return False
        else: m[a]=b
    return True

def load(p): return [l.strip() for l in open(p) if l.strip()]

S=load(sys.argv[1]); B=load(sys.argv[2])
Sset=set(S); Bset=set(B)
S_by=defaultdict(list); B_by=defaultdict(list)
for l in S: sk,sq=sk_seq(l); S_by[sk].append((sq,l))
for l in B: sk,sq=sk_seq(l); B_by[sk].append((sq,l))

# residue = B\S not an instance of any S rule (same defn as subset_check)
residue=[]
for r in Bset-Sset:
    sk,vr=sk_seq(r)
    if not any(instance(vr,vs) for vs,_ in S_by.get(sk,())):
        residue.append(r)
print("residue size:", len(residue))

cat=defaultdict(int); examples=defaultdict(list)
for r in residue:
    sk,vr=sk_seq(r)
    s_mates=S_by.get(sk,[])            # same-skeleton rules in S
    b_mates=B_by.get(sk,[])
    rev = any(instance(vs,vr) for vs,_ in s_mates)  # some S rule is an instance of r (S less general)
    # does a general parent of r exist in B' (not S)?
    genB = [l for vs,l in b_mates if l!=r and instance(vr,vs)]
    if not s_mates:
        c="no same-skeleton rule in S at all"
    elif rev:
        c="S has a rule that is a var-INSTANCE of r (survivor LESS general than r)"
    elif genB:
        c="general parent of r exists in B' but NOT in S (survivor missing from S)"
    else:
        c="same-skeleton S rules exist but unrelated by var-instance"
    cat[c]+=1
    if len(examples[c])<3:
        examples[c].append((r, genB[0] if genB else (s_mates[0][1] if s_mates else None)))

for c,n in sorted(cat.items(), key=lambda x:-x[1]):
    print(f"\n[{n:4d}] {c}")
    for r,other in examples[c]:
        print("   r      :", r)
        if other: print("   related:", other)
