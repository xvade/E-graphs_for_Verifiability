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

INITIAL (WRONG) VERDICT -- superseded, see CORRECTION below: I read chain-ABSENT +
Stopped=TimeLimit as "saturation budget / breadth-first". That was wrong.

**CORRECTION (rule-set gap, not budget).** The advisor flagged the tell: the k=2 node
max(max(g0,g1),g2) is ONE associativity application from the INITIAL graph, so its
absence after many iterations cannot be a budget/breadth story -- crowding slows a
frontier, it doesn't freeze it at input depth. Grep of pwl_rules_verified.txt settles
it: **the 621 contains NO pure ewmax associativity and NO pure ewmax commutativity.**
The only nested-ewmax rules are line 446 (idempotent: needs a SHARED operand on both
inner maxes) and lines 540-615 (min/max DISTRIBUTIVE laws). None can re-associate a max
over DISTINCT leaves -- so the lattice's inner group-max trees are literally frozen; the
chain can never form under the 621. My "assoc demonstrably fires" claim was false: the
depth-1 node max(g0,g1) I saw is an INPUT node, not a rewrite product. The 6x node
growth in the 1200s run was OTHER rules (bridges/distributive) firing. The maxout
succeeded only because it used ewmax_run1_verified.txt (4 rules that DO include pure
assoc+comm), NOT the 621. This vindicates the user's original read ("limitation of the
available rewrite rules"). Fix: add pure ewmax/ewmin assoc+comm (trivially Z3-valid) to
the rule set and re-run. (Breadth-first may still bind AFTER that -- but it is NOT what
blocks this run.)

### Cheap-gate re-run: 10x budget (n_sec 1200, n_iter 30)
Same rules/model/node-limit; raised time 120s->1200s and iter cap 12->30.
- **Stopped: TimeLimit(1203s), iter 13, 224318 nodes / 92080 classes.** Still TIME-bound
  (nodes = 45% of 500k limit; iter 13 << cap 30).
- Chain STILL breaks at depth 2/7 (both groups); order-independent closure STILL finds
  NO left-deep chain over all 8 leaves.
- Nodes grew ~6x (37806 -> 224318) but the chain frontier did NOT advance at all, and
  10x wall-clock bought only +3 iterations (10 -> 13): per-iteration cost is exploding
  super-linearly as the e-graph widens.
VERDICT (CORRECTED): the frozen frontier (depth unchanged despite 6x nodes) is NOT
breadth-first starvation -- it is the RULE GAP above. With no pure ewmax assoc/comm in
the 621, the group max-trees cannot be re-associated at all; the 6x node growth was
unrelated rules. So raising wall-clock was never going to help, but the reason is the
missing rule, not width-starvation. See CORRECTION under the 120s section.

### CONFIRMING TEST: lattice + ewmax assoc+comm (4-rule file) -> chain MATERIALIZES
Ran lattice.taso with ewmax_run1_verified.txt (the 4 rules that DO include pure ewmax
associativity + commutativity), --query_chain:
- **Stopped: Saturated, iter 20, 12218 nodes / 612 classes.** (vs 621: TimeLimit, 224318
  nodes -- the 621's distributive/bridge rules churn without ever saturating.)
- group 0 & 1 natural-order chain PRESENT at depth 7; order-independent closure: SOME
  left-deep chain over all 8 leaves EXISTS; FULL chain lattice PRESENT, root-equivalent
  = true.
PROVES the rule gap is THE cause: swap in pure assoc+comm and the tight depth-7 chain
appears and the e-graph saturates cleanly at 612 classes. Not budget, not breadth-first.
Actionable next step: add pure ewmax/ewmin assoc+comm to the lattice's rule set (or use
the 621 UNION these 4), run --verif_cost extraction, and bound -- expected ~7.80 (the
measured chain envelope), i.e. the lattice's first real verifiability improvement.

### RESOLVED: add the missing assoc rules -> lattice IMPROVES (first real gain)
Added pure ewmax assoc+comm to the rule set, ran --verif_cost extraction + reconstruct +
bound (alpha-beta-CROWN venv python; native-minmax lattice.onnx can't be bounded by
auto_LiRPA's minmax op, but the relu-lowered reconstructions bound fine):
| form | cert_ub | unstable | vs input | numchk |
|---|---|---|---|---|
| input (balanced)                 | 8.5019 | 14/120 | --          | -- |
| hand-built pure chain (envelope) | 7.8026 | --     | +0.70       | -- |
| assoc-only (4 rules, Saturated)  | 7.7092 |  9/120 | +0.79       | 4.8e-7 |
| **UNION 621 + assoc (TimeLimit)**| **7.5913** | **9/120** | **+0.91 (10.7%)** | 7.2e-7 |
- The UNION (full corpus + the 3 missing rules) gives the BEST bound, beating even the
  hand-built pure chain -- verif_cost combines assoc + commutativity + the 621's
  distributive laws to find a topology with fewer unstable ReLUs (9 vs 14). Semantics
  preserved (gate 7.2e-7). The union e-graph TimeLimit'd but the tight forms built early
  (assoc fires fast), so extraction still found them.
- This turns the documented min-of-max NULL into a real +0.91 (10.7%) verifiability
  improvement -- the lattice's first. Rule file: pwl_rules_plus_assoc.txt.

### AC-CLOSURE corpus (pwl_rules_ac.txt = 621 + 12 Z3-verified AC rules)
Manually added assoc(both dirs)+comm for ewmax/ewmin/ewadd/ewmul (12 rules, ALL absent
from the 621 for every op; all 12 Z3-proven), deduped-unioned with the 621 -> 632 rules.
pwl_rules_ac.txt SUPERSEDES pwl_rules_plus_assoc.txt (which had only the 3 ewmax rules).
verif_cost extraction (n_iter 20 / n_sec 180), reconstruct, bound:
| model | input | AC-corpus best | vs input | unstable |
|---|---|---|---|---|
| maxout  | 12.0257 | **9.6236** | **+2.40 (20%)** |  5/120 |
| lattice |  8.5019 | **7.6167** | **+0.89 (10.4%)** | 8/120 |
- maxout 9.6236 is a hair better than the 4-rule-file best (9.6519) -- new best.
- lattice 7.6167 ~= the ewmax-only union (7.5913); the extra ewmin/ewadd AC rules don't
  help the G=2 lattice (outer min is single-node, no residual adds) and only perturb which
  form a TimeLimit'd saturation lands on. Both ~10.5% wins.
- Both extractions numeric-gated (maxout 9.5e-7, lattice 7.2e-7): semantics preserved.

### AC corpus on the 4 Conv/Matmul models -- still INERT (as predicted)
n_diverse 8 with pwl_rules_ac.txt: mnist_tiny 8->1 distinct (=input), mnist_cnn_a 8->1
(=input), resnet2b 8->1 (!=input), inception 8->1 (!=input). AC rules add NO new
structure vs the 621 breadth: commutativity gives mirror-identical ReLU topology (same
bound), and residual ewadd's are 2-operand (no >=3-leaf chain to reassociate). So bounds
unchanged -- consistent with the prior neutral/barrier findings. The AC lever remains
specific to min/max-reduction-shaped models (maxout, lattice).
