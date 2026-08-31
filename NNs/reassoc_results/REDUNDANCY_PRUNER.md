# Redundancy pruner (tensat `-m redundancy`)

Greedily removes rules whose LHS=RHS equality is **re-derivable from the other kept
rules within a budget**, measured in the SAME sound engine that will use them. Preserves
the equational closure (only ever removes); sound even in the untyped egg (removal can't
add a false equality).

## Mechanism (`prune_redundant` / `ground_side` in tensat/src/main.rs)
For each rule (largest-LHS first, so simple generators survive): ground its LHS and RHS by
mapping each pattern `?var` to a fresh Input leaf of uniform shape [4,4] (shared across
both sides), join them with a `Noop`, saturate with the other kept rules for
`--redundancy_iters` iterations, and test whether the two sides land in one e-class. If
so, the rule is redundant -> drop it. Only elementwise/PWL rules (ewadd/ewsub/ewmax/ewmin/
ewmul/relu) are grounded; any rule with a non-elementwise op is conservatively KEPT.

## The knob = reachability/budget tradeoff
`--redundancy_iters B` (default 4): a rule is pruned only if the rest re-derive it within
B e-graph iterations. Small B -> prune only SHORT-derivation redundancies (keeps
shortcuts, preserves budget-limited reachability). Large B -> prune aggressively (smaller
set, worse reachability). This is the tunable line to sweep later.

## Validation (4 AC rules) -- CORRECT, and instructive
Input: assoc L->R, assoc R->L, comm, idempotent. Result: **pruned 1 (assoc L->R), kept 3.**
assoc-L->R IS derivable from {comm, assoc-R->L}: max(max(a,b),c) =comm= max(c,max(a,b))
=assocR->L= max(max(c,a),b) =comm= ... = max(a,max(b,c)). So the minimal generating set of
the AC-closure is 3 rules (one assoc direction + comm), not 4 -- the pruner found it, and
kept the genuinely-independent idempotent rule. (This corrected the earlier "both
directions needed" assumption.)

## Full corpus (pwl_rules_ac.txt = 621 + 12 AC = 632)
**pruned 515 redundant -> kept 117** (623 groundable, 9 non-groundable kept), budget 4.
The current corpus is ~82% redundant: 632 rules collapse to a 117-rule generating core
with the same closure (up to 4-step derivations). Output: pwl_rules_ac_pruned4.txt.
Directly attacks the saturation-budget problem -- a 117-rule set saturates far lighter
than 632. (Next: confirm the pruned set reaches the same forms on maxout/lattice/tll, and
sweep the budget knob.)

## NOTE re the pipeline plan
This validates the pruner. The intended use is NOT to prune the 621 (which already lost
rules to TASO's quotient) but to: RELAX the generator's quotient (emit the full closure
incl. AC), regenerate, then prune back to a minimal-complete core. That regeneration is
the next phase.
