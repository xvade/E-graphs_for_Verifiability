# tll (VNN-COMP Two-Level Lattice): first REAL-WORLD verifiability win

**Headline: lifting the VNN-COMP tll net to an explicit max-of-min lattice and
reassociating it yields a certified upper bound of 8.26 vs the shipped model's 19.59 --
a 58% tighter certificate on the SAME function** (L-inf box x0=0, eps=1.0).

## The problem the lift solves
The tllBench ONNX is TLL-compiled to a sequential MatMul/Add/Relu MLP (18 MatMul, 18 Add,
8 Relu) -- min/max are baked into the weights, not present as ops. A mechanical importer
would leave it inert. BUT the architecture is recoverable from the named weights (verified
numerically, 4.8e-7): linearLayer [2,16] = 16 local affine fns; selectionLayer [16,256]
one-hot = 16 groups x 16 members; minBanks/maxBanks = the reduction. Net semantics:
**max_g min_{k in group g} (W_k.x + b_k)** -- a genuine max-of-min two-level lattice.
`NNs/build_tll_lattice.py` rebuilds that explicitly (ewmax/ewmin, vector trick width-8 to
dodge taso's small-N SGEMM), gated to 4.8e-7 vs the ONNX.

## Result (alpha-CROWN certified upper bound, same box)
| form | cert_ub | unstable | vs baseline | numchk |
|---|---|---|---|---|
| **BASELINE: original compiled tll MLP** | **19.5945** | 628/1020 | -- | 0 |
| lifted lattice, typical n_diverse form   |  ~10.06 | ~92/912 | +9.5 (48%) | 4.8e-7 |
| **lifted + best AC-reassociation (form0)** | **8.2606** | **89/904** | **+11.33 (58%)** | 3.6e-7 |

Pipeline: build_tll_lattice.py -> taso ingest (Min/Max/Gemm, no SGEMM) -> tensat with
pwl_rules_ac.txt (the 621 + AC-closure) n_diverse 16 -> reconstruct(->relu) -> alpha-CROWN.
All forms numeric-gated equivalent to tll.

## Honest decomposition of the 58%
- **The LIFT does most of it (19.59 -> ~10.06, ~48%).** The compiled TLL gadget MLP has
  628/1020 UNSTABLE ReLUs; the explicit reconstructed lattice has only ~90/900. The
  min/max-via-relu gadgets in the compiled form are far less stable than the direct
  max(a,b)=a+relu(b-a) topology. This is a decompilation using TLL domain knowledge --
  TLL-family-specific, not a general graph rewrite.
- **REASSOCIATION (the general tensat/AC contribution) refines 10.06 -> 8.26 (~18%).**
  Among the lifted forms, the best AC-reassociation (form0) beats the near-balanced ones
  (~10.06). BOTH lattice levels (min-16, max-16) are reassociable here (unlike the G=2
  synthetic), so the AC-closure rules -- ewmin included -- are load-bearing for the first
  time.

## Significance
First verifiability improvement on a real VNN-COMP model (all prior wins were synthetic:
maxout, the hand-built lattice). It's also large. Caveat: the lift assumes we know the net
is a TLL (true for this benchmark family); the reassociation half is general. The whole
chain is sound (every step numeric-gated). Baseline box x0=0, eps=1.0; not the vnnlib
property (a certified-ub comparison in the project's standard protocol).

## Infra unblocked along the way
- **taso MatMul-casing bug fixed** (xf_operators['MatMul']): taso registered only the
  lowercase 'Matmul' key, so every standard-ONNX 'MatMul' was skipped -> pure FC/MatMul
  graphs degenerated to inputs+weights, zero compute. This was the real "tll degenerate
  ingestion" barrier. See BUGS.md. (Residual: taso's SGEMM cost-measurement still aborts
  on small-N matmuls -- tll's width-1 output layers -- a separate pre-known bug the vector
  trick sidesteps; the lift avoids it, so the mechanical importer for scalar-output FC
  nets remains blocked on that until taso's cost measurement is guarded.)
- derive_weight_names_baseline.py: guard empty param lines (elementwise ops).
