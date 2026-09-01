# Reverify all models with the 1097 PWL+matmul core (relaxed_d3_core.txt)

Status: IN PROGRESS. This records the run so far + the diagnosis. (2026-08-31)

## Result of the first pass (1097 core, n_diverse 20, n_iter 12, n_sec 120)
| model | all-forms cert_ub | unstable | baseline | verdict |
|---|---|---|---|---|
| mnist_tiny_mlp | 0.9710 (uniform, n=20) | 2/20 | — | **NEUTRAL control CONFIRMED** (matmul reassoc between fixed ReLUs can't move pre-activations) |
| maxout | 12.0257 (uniform, n=20) | 14/120 | input 12.0257 | no improvement (prior wins were 10.56 diverse / 9.65 verif_cost) |
| lattice | 8.5019 (uniform, n=20) | 14/120 | plateau 8.50 | no improvement |

All forms numeric-gated equivalent (dmax < 1e-6). The bound HARNESS is intact: 12.0257 and
8.5019 reproduce the recorded baselines (VERIF_COST_RESULT, MAXOUT_RESULT) to 4 decimals, and
14/120 unstable matches the recorded lattice numbers digit-for-digit.

## Diagnosis (why flat, not the prior wins)
- **Extraction gap, not harness.** `n_diverse` collapses: `"0 new enodes added"` after ~3
  samples, for BOTH the old 632 (`142 86 85 0 0...`) and new 1097 (`134 93 53 0 0...`) rule
  sets. So the improving reassociation is likely in the e-graph but not reached by diverse
  sampling under these flags.
- **Invocation is confounded with rules.** The prior maxout win used prefix `maxout_out`,
  **40** forms, rules `pwl_rules_ac.txt` (632); this pass used n_diverse 20 and the 1097 core.
  1097 rules ~= 2x per-iteration saturation cost -> saturation stopped on TimeLimit at 9 iters,
  so the same `n_sec` buys LESS saturation than the old runs. Budget confounded with rule set.
- **lattice 8.5019 is the pre-AC-fix plateau** (commit 17ae685: "full 621: NULL, all forms
  8.50"). Lattice only ever moved (+10.7%) after the 12 hand-added AC rules. The 1097 core
  documentedly lacks bare binary commutativity (enumeration-order artifact) and its reassoc
  rules are root-order-canonicalized -> AC-gap hypothesis has direct historical support. The
  GEN_COMMUTE regeneration (commute_d3, in flight) is the systemic fix.
- maxout, by contrast, won (+1.46) with the PLAIN 621 pre-AC -> its failure under the 1097
  core points at the budget-4 PRUNE or the invocation, not AC. Different mechanism per model.
- Identical verif_cost gap-cost (35482788) for maxout & lattice is LEGIT: the interval files
  are byte-identical (lattice built from the same seed-0 W,b operating point as maxout).

## CORRECTION (2026-08-31, verif_cost + reconstruct-fixed): NOT an extraction regression -- the 1097 core is the weaker rule set
The `--n_diverse` "extraction regressed" story below is SUPERSEDED. Two bugs masked the real
picture: (1) reconstruct crashed on Cuda-35 (taso python ext RPATH->build_gpu; see BUGS.md),
zeroing the bound; (2) I led with `--n_diverse` instead of the deterministic `--verif_cost`.
With BOTH fixed, the decisive control is clean:

| rule set (verif_cost, maxout, current binary + fixed CPU reconstruct) | gap-cost | form depth | cert_ub | unstable |
|---|---|---|---|---|
| **632 `pwl_rules_ac.txt`** | 12,807,726 | **18** | **9.6236** | **5/120** |
| **1097 core `relaxed_d3_core.txt`** | 35,482,788 | 7 (=input) | 12.0257 | 14/120 |

632 reproduces the original +2.37 win (record 9.6519, depth-18, 5/120) via the SAME binary and
reconstruct path. So the verif_cost EXTRACTION and the binary are NOT regressed -- the 1097 core
just fails to steer to the deep form. (The `--n_diverse` collapse is a separate, real issue but
is NOT what blocked the win.)

**Why the 1097 core fails (hypothesis, not yet proven vs starvation):** the 1097 core has MORE
min/max rules than 632 (592 vs 335 ewmax/ewmin lines), so it does NOT lack reassoc rules by
count. Both runs stop on TimeLimit, but 632 reaches 19 iters@depth-18 while the 1097 core (~2x
per-iter saturation cost) reaches only 13 iters@depth-7. This matches the breadth-first
starvation mechanism (egraph-breadth-first-limitation memory): a larger rule set explores
broader-and-shallower per iteration, starving the narrow-deep depth-18 chain under fixed time.
NOTE: exact-line diff of the two rule files is uninformative -- they use different variable-name
canonicalization (comm shows 0 overlap, a naming artifact, not semantic absence).

**DECISIVE NEXT TEST (cheap): give the 1097 core a big time budget** (`--n_sec 600 --n_iter 60`)
on maxout verif_cost. If it then reaches depth ~18 / ~9.6 -> pure starvation (fix: budget or the
narrow-deep extraction lever, MCTS/heuristic firing). If it plateaus at depth 7 even with ample
time -> the specific composing chain is genuinely absent from the 1097 core (fix: rule
generation/pruning -- the budget-4 redundancy prune or the relaxed toggles dropped a bridge).

Infra unblocked to get here: reconstruct now runs CPU-only after binary-patching the taso python
ext RPATH build_gpu->build (BUGS.md). `NNs/vc_control_632.sh`, `NNs/vc_recon.sh` are the drivers.
Results table in `verif_cost_reverify_results.txt`.

## [SUPERSEDED] earlier claim: diverse EXTRACTION regressed since 2026-08-29 (not the 1097 core)
Consistent DAG-depth metric on maxout diverse forms (n_diverse 40, same invocation):
| source | forms | depth histogram |
|---|---|---|
| **PRIOR maxout_out (2026-08-29)** | 40 | **{10:1, 11:6, 12:8, 13:7, 14:11, 15:3, 16:2, 17:1, 18:1}** -- WIDE, deep, diverse |
| new 632 (pwl_rules_ac)  | 25 | {9:24, 14:1} -- collapsed to one shallow depth |
| new 1097 core           | 40 | {9:1, 10:38, 12:1} -- collapsed to one shallow depth |

**Both the OLD 632 and the NEW 1097 rule sets now collapse to a single shallow depth**, while
the Aug-29 forms reached depths 10-18 with real diversity. So the failure is NOT the 1097 core
-- it's the DIVERSE EXTRACTION itself (the `--n_diverse` sampler reports "0 new enodes added"
after ~3 samples and returns duplicates). This regressed between the Aug-29 binary and the
Aug-31 rebuild. Suspects (this window's tensat commits touching cost/extraction, git log):
`f52cc16` (added ewsub/ewmax/ewmin to the COST), `e9b139a` (min/max in CheckApply), the
VerifCost commits. The 1097 core is UNPROVEN either way -- it was never given a working
extraction to show its reach. mnist stays a valid NEUTRAL result regardless (its bound is
extraction-independent: matmul reassoc can't move pre-activations).

## NEXT (a focused bisect, separate task): build the Aug-29 tensat commit, re-run maxout
## n_diverse, confirm depths 10-18 return -> then git-bisect the DiverseCost regression.
## Only after extraction is restored does the 1097-core-vs-632-vs-2658-vs-+12AC matrix mean anything.

## Experiment matrix (deferred until extraction is fixed; maxout, ALL on CPU)
1. 632 (pwl_rules_ac.txt) w/ prior invocation (n_diverse 40, generous n_sec) -> reproduce
   [10.56, 13.22]? validates the invocation.
2. 2658 pre-prune (relaxed_d3_verified.txt) vs 1097 core -> did budget-4 prune trade away
   reachability? (the tunable knob).
3. core+12AC vs core -> is commutativity load-bearing for maxout too?
Then lattice: +-12AC pair + `--query_chain` (lattice-specific diagnostic) on the saturated egraph.

## Infra fixed to get here
- taso `export_op` (ops.cc:1031) had NO cases for EW_SUB/MAX/MIN (op types 26/27/28) -> any
  extracted form containing an ewmin/ewmax/ewsub crashed on export (`Assertion 'false'`). Added
  the three cases; rebuilt the CPU taso .so (manual g++, cmake install is broken). See BUGS.md.
- taso GPU lib (build_gpu) still fails `Cuda failure 35` at ops_cudnn.cu:24 (cuDNN init) even
  though the bare CUDA runtime works now -> use CPU taso (build) for stages 1-2 (structural,
  no GPU needed); abcrown venv (GPU or CPU) for stage 3.
- generator GEN_COMMUTE flag added (both operand orders for commutative ops) -> commutativity
  can now be generated; commute_d3 regeneration in flight.
