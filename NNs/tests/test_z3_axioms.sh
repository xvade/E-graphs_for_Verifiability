#!/bin/bash
# Tests for the Z3 tensor-axiom lane (NNs/tensor_axioms.py) + its union with the
# PWL lane in z3_verify_egg.py. Runs with the taso_py env (z3 lives there, NOT in
# the container's /opt/conda python -- see PROBLEMATIC.md #5), so this is a
# SEPARATE host-run suite from NNs/tests/run_tests.sh (which is in-container).
#
#   bash NNs/tests/test_z3_axioms.sh
#
# Exits nonzero if any assertion fails. Plain asserts, no pytest.
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
PY=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/miniconda3/envs/taso_py/bin/python3
export PYTHONPATH="$REPO/NNs:/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/z3pkg"
cd "$REPO"
$PY - "$REPO/NNs/tests/z3_canaries_false.txt" <<'PY'
import sys
import z3_verify_egg as L1
import tensor_axioms as L2
TIMEOUT = 8000
def parse(s): return L1.parse(L1.tokenize(s), 0)[0]
PASS = [0]; FAIL = [0]
def ok(m):  print("  PASS:", m); PASS[0]+=1
def bad(m): print("  FAIL:", m); FAIL[0]+=1

# --- soundness: false conv/concat/matmul rewrites must NOT verify (either lane).
# A consistent axiom set cannot prove them; a canary that flips to VERIFIED means
# the axiom set became inconsistent (then it would prove everything).
print("== canaries: false rewrites stay unproven (soundness) ==")
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or "=>" not in line: continue
    v, lane = L1.verify_rule(line, TIMEOUT)   # full union (lane1 then lane2)
    (ok if v != "VERIFIED" else bad)("canary not VERIFIED: {} -> {}".format(line[:48], v))

# --- flips: rules lane 1 must REJECT/UNKNOWN, lane 2 proves (the whole point).
print("== flips: tensor-axiom lane proves what the PWL lane cannot ==")
flips = [
    ("conv linear in weight",
     "(conv2d 1 1 0 0 ?input_1 (ewadd ?input_2 ?input_3))",
     "(ewadd (conv2d 1 1 0 0 ?input_1 ?input_2) (conv2d 1 1 0 0 ?input_1 ?input_3))"),
    ("relu(conv NONE) = conv RELU",
     "(relu (conv2d 1 1 0 0 ?input_1 ?input_2))",
     "(conv2d 1 1 0 2 ?input_1 ?input_2)"),
    ("relu over concat",
     "(relu (concat 1 4 ?input_1 ?input_2))",
     "(concat 1 4 (relu ?input_1) (relu ?input_2))"),
    ("2-D transpose involution",
     "(transpose (transpose ?input_1 1_0 0) 1_0 0)",
     "?input_1"),
    ("matmul . identity-matrix (Imatmul)",
     "(matmul 0 ?input_1 (Imatmul))", "?input_1"),
    ("conv . identity-conv (Iconv) = id",
     "(conv2d 1 1 0 0 ?input_1 (Iconv 3 3))", "?input_1"),
]
for name, lhs, rhs in flips:
    ln, rn = parse(lhs), parse(rhs)
    import z3 as _z3
    # lane 1 alone must NOT verify:
    b = L1.Builder(); s = _z3.Solver(); s.set("timeout", TIMEOUT)
    s.add(b.build(ln) != b.build(rn))
    l1 = "VERIFIED" if s.check()==_z3.unsat else "not"
    v2 = L2.verify_rule(ln, rn, TIMEOUT)
    (ok if (l1 != "VERIFIED" and v2 == "VERIFIED") else bad)(
        "{}: lane1={}, lane2={}".format(name, l1, v2))

# --- regression: a PWL rule still proven by lane 1 alone (union must not lose it).
print("== shuffle invariance: transpose shuffle flag is value-invariant ==")
# transpose(x,perm,0) == transpose(x,perm,1): shuffle changes only strides, not values
# (transpose.cc). Both lanes must ignore shuffle -- guards against re-keying on it.
for name, lhs, rhs in [
    ("2-D transpose shuffle 0==1",
     "(matmul 0 ?input_1 (transpose ?input_2 1_0 0))",
     "(matmul 0 ?input_1 (transpose ?input_2 1_0 1))"),
    ("3-D transpose shuffle 0==1",
     "(transpose ?input_1 1_2_0 0)", "(transpose ?input_1 1_2_0 1)"),
]:
    v, lane = L1.verify_rule("{}=>{}".format(lhs, rhs), TIMEOUT)
    (ok if v == "VERIFIED" else bad)("{}: {} (lane {})".format(name, v, lane))

print("== regression: PWL lane still proves min/max + ewadd assoc ==")
for name, lhs, rhs in [
    ("ewadd assoc", "(ewadd ?input_1 (ewadd ?input_2 ?input_3))", "(ewadd (ewadd ?input_1 ?input_2) ?input_3)"),
    ("ewmax comm",  "(ewmax ?input_1 ?input_2)", "(ewmax ?input_2 ?input_1)"),
]:
    v, lane = L1.verify_rule("{}=>{}".format(lhs, rhs), TIMEOUT)
    (ok if (v == "VERIFIED" and lane == 1) else bad)("{}: {} (lane {})".format(name, v, lane))

print("======================================")
print("Z3 AXIOM TESTS: {} passed, {} failed".format(PASS[0], FAIL[0]))
sys.exit(0 if FAIL[0]==0 else 1)
PY