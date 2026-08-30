# TENSAT-driven verifiability improvement on a maxout net (headline result)

**Claim achieved:** tensat's Z3-verified min/max reassociation rules, applied to a
real single-input maxout model, produce semantics-preserving rewrites that measurably
IMPROVE the certified bound -- the project's central goal.

## Setup
- Model: single-input maxout `g(x)=max_i(W_i.x+b_i)`, M=16 affine maps, d=8 input,
  realized as a BALANCED ewmax tree (the loose end, per the PoC). Built as M vector
  Linear(d,d) (row 0 = W_i,b_i; component 0 == the scalar maxout) -> onnx -> taso.
  Same W,b as PoC N=16 rep0 (seed 0), so the hand-built chain/balanced envelope
  brackets the forms. L-inf box eps=0.5.
- Rewrite: 4 Z3-VERIFIED ewmax rules (associativity both ways, commutativity, one
  generator rule). tensat saturated + extracted 40 diverse forms.
- Reconstruction lowers ewmax->relu (max(a,b)=a+relu(b-a)), so ab-CROWN sees the ReLU
  topology. Certified upper bound on component 0 via alpha-CROWN.

## Result
- **All 40 forms pass the numeric gate** (max|onnx - true_max| ~5e-7): semantics preserved.
- **Pipeline cross-validated EXACTLY:** reconstructed input bound = 12.0257 =
  hand-built balanced envelope 12.0257.
- Envelope: chain(tight)=11.7978, balanced(input,loose)=12.0257.
- **tensat forms span certified ub [10.5620, 13.2213].**
- **Best form = 10.5620 -> +1.4637 (12%) tighter than the balanced input (12.0257).**
- Tightest forms have fewest unstable ReLUs (7-9 of 120) -- the PoC stability mechanism.
- Best forms beat even the hand-built chain (11.80) because tensat explores leaf
  REORDERING (commutativity), a richer space than the PoC's associativity-only.
- depth vs ub correlation -0.31 (deeper weakly tighter; unstable-count is the stronger driver).

## Honest scope
- The best form was selected POST-HOC by measurement; tensat PRODUCED it but did not
  autonomously CHOOSE it -- that needs a verifiability-aware extraction cost (follow-on).
- The spread includes forms WORSE than the input (up to 13.22): rewriting is not
  uniformly good; the BEST rewrite helps. Steering (a verifiability cost) is the payoff lever.
- 40 samples is partial exploration; the true optimum may be tighter.

Artifacts: build_maxout.py, envelope_maxout.py, bound_maxout_forms.py,
maxout_reconstruct.sh, maxout_bounds_table.txt, maxout_wbx.npz, maxout_forms/.
