# Where TASO assumes rewrites serve SPEED (audit)

TASO is an inference-runtime optimizer: it searches for a graph that computes the same
function *faster*. Every design point below bakes in "reward = lower measured runtime."
For verifiability we want the opposite kind of reward (tighter certified bound), and the
relevant rewrites are typically runtime-NEUTRAL (associativity/commutativity reshapes a
ReLU tree without changing FLOPs). So each site is BOTH a likely issue point AND a place
to insert a verification-centric objective. Ordered by how directly it blocks us.

Legend: **BLOCKS** = actively prevents verifiability rewrites; **ORACLE** = defines "cost"
as speed; **BIAS** = tilts toward speed but not fatal.

---

## 1. [BLOCKS, deepest] Cost-based backtracking search only accepts STRICT speedups
`src/core/ops.cc:438-471` (`Graph::optimize`), best updated at `:441` `if (subGraph->
total_cost() < bestCost)`, candidates ordered by `GraphCompare` (cheapest-first PQ).
- `bestGraph` is replaced ONLY on a STRICT runtime decrease. A cost-NEUTRAL rewrite (equal
  runtime) can never become the returned graph. Since our verifiability rewrites are
  cost-neutral AC-rearrangements, **TASO::optimize can never output one, regardless of the
  rule set.** This is the structural reason the project uses tensat/egg (equality
  saturation keeps ALL equivalent forms) + our own extraction, instead of TASO's optimizer.
- Verification-centric replacement: rank/accept by a verifiability score, not `total_cost`.
  We already do this on the egg side (VerifCost); on the TASO side it would mean replacing
  the PQ comparator + the `<bestCost` acceptance test with a bound-aware objective.

## 2. [BLOCKS] The alpha pruning threshold discards any runtime-increasing candidate
`src/core/ops.cc:466` `xfers[i]->run(0, subGraph, candidates, hashmap, bestCost*alpha, ...)`
applied at `src/core/substitution.cc:1057` and `:1321` `if (newGraph->total_cost() <
threshold ...) candidates.push(newGraph)`.
- Only graphs with `total_cost < alpha*bestCost` are ever enqueued. Default `alpha = 1.0`
  (`python/taso/__init__.py:969`), so even cost-neutral rewrites are pruned before
  exploration; the paper's 1.05 would explore-but-never-select them (see #1). Either way a
  bound-improving-but-slower rewrite is unreachable.
- Insertion point: this `threshold` comparison is the single most surgical place to swap in
  a verifiability budget (allow candidates that raise runtime if they tighten the bound).

## 3. [ORACLE] `total_cost` = sum of measured GPU kernel times
`src/core/ops.cc:1122` `Graph::total_cost` = sum of `op.ptr->runtime`. Each op's `runtime`
is set by `measure_*_cost` (`src/core/element.cc:65`, `matmul.cc:65`, `conv2d.cc:138`,
`pool2d.cc:111`, `activation.cc:68`, ...) which run the kernel and time it with
`cudaEventElapsedTime` (`src/cudnn/element_kernel.cu`, `conv2d_kernel.cu`, etc.).
- The cost oracle is literally wall-clock ms on the GPU. It is blind to ReLU stability,
  interval width, or anything verification cares about; a chain and a balanced ReLU tree
  have ~identical `total_cost` yet very different certified bounds.
- Replacement: a `verif_cost(op)` oracle (e.g. per-ReLU triangle-relaxation gap-area over
  an input interval, as in tensat/src/optimize.rs VerifCost) summed the same way.

## 4. [BLOCKS, upstream] Generator is AC-blind (already diagnosed)
`taso/src/generator/generator.cc`: `variable_ordering` (`:1226`), `same_via_subst`
(`:1257`), `pass_checks` common sub/super-graph pruning (`:1239`).
- Canonicalizes associative/commutative operators, so pure assoc/comm are never emitted as
  rules (empirically: 0 in the 621 for ewmax/ewmin/ewadd/ewmul; only the idempotent
  shared-operand max rule survives). These cost-neutral rewrites are exactly our lever.
- This is the proven exemplar of the failure class -- see [[taso-generator-is-AC-blind]] /
  EGRAPH + VERIF_COST docs. Fix already applied downstream: hand-add + Z3-verify the
  AC-closure (pwl_rules_ac.txt).

## 5. [BIAS] Hardcoded fusion transforms reward fewer/larger kernels
`src/core/ops.cc:392-397` (`create_enlarge_merge_convs`, `create_merge_group_convs`),
`create_conv_relu` (`src/core/substitution.cc:61`).
- Fusion = fewer kernel launches = faster, and it is the built-in (non-pb) xfer set. Our
  own sweep found fusion verification-neutral-to-negative (fused InceptionMNIST never beat
  unfused). So fusion-preference is speed-motivated and not helpful (sometimes harmful) for
  bounds. Not fatal, but a bias to be aware of when a model arrives pre-fused.

## 6. [ORACLE, API] The public API only exposes speed knobs
`python/taso/__init__.py:969` `def optimize(graph, alpha=1.0, budget=1000, ...)`.
- The two tunables a caller gets -- `alpha` (speed slack) and `budget` (search steps) --
  are both speed-search parameters. There is no verifiability knob. A verification-centric
  API would take an interval/spec and a bound objective.

## Cross-ref: the tensat side has the SAME assumption, already replaced
`tensat/src/optimize.rs`: `TensorCost` / greedy / ILP extraction all minimize a runtime
proxy. We added `VerifCost` (per-ReLU gap-area, optional sensitivity weight) as the
verification-centric extraction objective -- the concrete template for what "insert our own
optimization" looks like at each ORACLE site above.

## The two the user will act on
- **#2 (alpha threshold, substitution.cc:1057)** -- surgical insertion point for a
  bound-aware acceptance test if we ever want TASO-native verifiability search.
- **#4 (generator AC-blindness)** -- the rule-source fix; already handled by hand-adding
  the AC-closure, but the general lesson (audit what canonicalization discards) stands.
