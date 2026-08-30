# Verifiability-aware extraction cost (VerifCost) -- result

VerifCost (tensat/src/optimize.rs) scores each ewmax/ewmin enode by the TRIANGLE
RELAXATION AREA of its lowered ReLU's pre-activation interval (IBP over the input box),
computed by a lazy memoized interval walk at extraction time; leaf intervals injected
by weight-name key from a sidecar (gen_leaf_intervals.py). ONE deterministic extraction
(`--verif_cost --interval_file`), no sampling.

## Maxout (pure max) -- SUCCESS
| form | cert_ub | unstable | vs input |
|---|---|---|---|
| input (balanced)              | 12.0257 | 14/120 | -- |
| best of 40 random samples     | 10.5620 |  9/120 | +1.46 |
| hand-built chain envelope     | 11.7978 | 11/15  | +0.23 |
| **VerifCost (1 extraction)**  | **9.6519** | **5/120** | **+2.37 (20%)** |

VerifCost beats random sampling AND the hand-built chain, in a single deterministic
extraction -- it steers to a depth-18 form with only 5/120 unstable ReLUs (it exploits
leaf REORDERING + depth together). 16/16 leaf intervals matched; numeric gate 1.4e-6.

## Lattice (min-of-max) -- LIMIT (informative)
VerifCost form: cert_ub = 8.5019 = input (14/120 unstable), numeric gate 7.2e-7. NO
improvement -- pinned at the input, same as the 40 random samples. The chain form
(envelope 7.80) is not reached. This is the additive surrogate's LIMIT, now empirical:
gap-AREA summed over all ReLUs captures per-ReLU error but NOT the critical-path
weighting the min-dominated lattice bound needs (the outer-min ReLU governs the bound;
reducing other ReLUs' gaps doesn't move it). It even minimizes to 14 unstable = the
lattice's floor, yet the bound doesn't move.

## Net
The verifiability-aware cost turns "sample 40, pick best post-hoc" into "one steered
extraction that beats them" ON PURE-MAX structure (+2.37). On min-dominated structure
it hits the documented additive-cost limit -- the empirical case for the next lever, a
per-ReLU SENSITIVITY WEIGHT (one backward-CROWN pass on a reference form) so the cost
targets critical-path ReLUs, not all ReLUs equally. (Not built yet, per scope.)
