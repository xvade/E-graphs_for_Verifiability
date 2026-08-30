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

## Sensitivity-weighted VerifCost (--sensitivity_file) -- tried, does NOT unlock lattice
Weights each ReLU's gap-area by its backward-CROWN sensitivity |lambda| (gen_sensitivity.py:
top-down CROWN-lite pass on a reference form; min-ReLU=1.0 decaying to 0.03; keyed by
weight-name set). Result:
- **maxout: 9.6519 (5/120) = unweighted -- NEUTRAL** (single max-reduction has near-uniform
  sensitivity [0.118,1.0]; no regression -- the mechanism is sound).
- **lattice: 8.5019 (14/120) = unweighted -- NO IMPROVEMENT.**

Why it fails on the lattice (two compounding reasons, both informative):
1. The lattice's critical-path ReLU is the OUTER MIN, which is STRUCTURALLY FIXED
   (min(maxA,maxB) -- 2 groups, no reassociation freedom). No rewrite reduces its gap, so
   up-weighting it changes nothing. The bound gain (envelope chain 7.80) needs reassociating
   the INNER max-groups -- which sensitivity DOWN-weights (they're behind the min). Their
   benefit is a SECOND-ORDER effect (tighter group intervals -> tighter min inputs -> tighter
   output) that first-order |lambda|-gap-area does not see.
2. REACHABILITY: every tensat form (40 sampled + unweighted-verif + weighted-verif) plateaus
   at exactly 8.5019. The hand-built chain (7.80) is a DIFFERENT realization that tensat's
   rules + caps apparently don't generate. So no extraction cost could reach it -- the limit
   is partly the e-graph, not just the cost.

Net: the verifiability-aware cost is a real win on pure-max (maxout +2.37, deterministic).
The min-of-max lattice resists BOTH the local cost (min unmovable, second-order gains) AND
tensat's reachable rewrite set. The next lever is NOT more cost engineering: it's either
rules that create the tight lattice form (reachability) or a genuinely global/second-order
objective -- both bigger than a per-enode weight.

## E-graph chain-query (reachability diagnostic, added later)
Added `--query_chain` to tensat/main.rs: after saturation (EXACT verif-cost config:
`-r pwl_rules_verified.txt --n_iter 12 --n_sec 120 --n_nodes 500000 --no_cycle`), it
looks up (non-mutating `egraph.lookup`) whether the left-deep CHAIN association of the
lattice is materialized. Order-independent (subset-closure bitmask per group) + natural
-order break-depth + blacklist check + root-equivalence.

RESULT on lattice.taso (2 groups x 8 leaves):
- **Stopped: TimeLimit(120.2s), iter 10/12, 37806 nodes / 15808 classes.** Saturation
  did NOT complete (and node count is 7% of the 500k limit -> it's the WALL-CLOCK
  budget, not node count, not iter limit).
- Natural-order chain BREAKS at depth 2/7 in BOTH groups (assoc DOES fire -- shallow
  spines exist -- but the depth-7 running-max spine never forms).
- Order-INDEPENDENT subset closure: NO left-deep chain over all 8 leaves in EITHER
  group, in ANY leaf order.
- No blacklist hit (chain never built, so nothing to blacklist).

VERDICT (maps to the 4-row table): chain ABSENT + Stopped=TimeLimit => **saturation
BUDGET**, specifically wall-clock. NOT a rule gap (associativity demonstrably fires),
NOT the cycle blacklist, NOT the extraction heuristic (there is nothing in the e-graph
for any cost to pick). This CONFIRMS the earlier "reachability, not cost" framing and
pins the cause to the 120s time limit. Untested next lever: raise --n_sec (e.g. 1200)
and re-query to see whether the depth-7 spine forms with more time; if it Saturates
without the chain, THEN check ewmax assoc-rule directions for a genuine rule gap.
