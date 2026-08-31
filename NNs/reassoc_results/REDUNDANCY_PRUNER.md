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

## Phase 2 progress: quotient relaxation (generator toggles) -- INTUITION BUILT
Made the generator's 4 quotient steps env-toggleable (taso 03825ff): RELAX_SUBGRAPH,
RELAX_SUPERGRAPH, RELAX_VARORDER, RELAX_SUBST. Depth-2 PWL intuition run:
| config | transfers | vs baseline |
|---|---|---|
| baseline (original)            |  34 | 1x |
| RELAX_VARORDER only            |  34 | 1x (inert alone) |
| **RELAX_SUBST only**           | 260 | **7.6x (the main lever)** |
| RELAX_VARORDER+SUBST           | 438 | 12.9x |
| all four relaxed               | 488 | 14.4x |
- **RELAX_SUBST (drop the renaming-dedup) is what recovers the AC family.** Associativity
  re-emerges as e.g. EWMax(x3,EWMax(x1,x2)) => EWMax(x1,EWMax(x2,x3)) (root operand order
  is still canonicalized, so it's not literally EWMax(EWMax(..),x3)); plus reorderings.
  Emitted redundantly across input classes (x/w/i) -- exactly what the redundancy pruner
  collapses.
- Blowup is ~14x at depth 2; depth-3 compounds (baseline depth-3 PWL was ~790 transfers,
  so all-relaxed depth-3 could be ~10^4+). That's the explosion to watch.

## Phase 2 big-run plan (needs a fresh allocation)
1. Rebuild generator all-relaxed at depth 3 (PWL op set first, then full op set), measure
   the transfer count -- gate on tractability (if >~50k, do RELAX_SUBST-only first).
2. pb2egg the relaxed graph_subst.pb -> egg rules.
3. Z3-verify (the count is the cost driver; ~10^4 rules is feasible but slow).
4. Redundancy-prune (tensat -m redundancy, budget 4) to a minimal-complete core.
5. Validate the pruned core reaches the same forms on maxout/lattice/tll; sweep budget.

## Phase 2 depth-3 GATE (tractability) -- intuition complete, found a scaling wall
- Generation runtime is ENUMERATION-bound, NOT relaxation-bound: relaxing only changes
  which transfers are KEPT, not the graph search. Baseline depth-3 PWL ~= 1536 transfers,
  ~20 min. RELAX_SUBST depth-3 = SAME ~20-min enumeration but emits >=7936 (still climbing;
  ~7-10x baseline).
- So the "explosion" is in OUTPUT COUNT (-> Z3 + prune load), not generation time.
- **Scaling wall at the PRUNE step:** the redundancy pruner does ~1 saturation per rule
  (up to 5s). ~10k relaxed rules -> up to ~14 h. NOT tractable as-is.
- **Refined pipeline (the fix):** relaxed-gen -> pb2egg -> CHEAP syntactic pre-dedup
  (exact-match + variable-renaming canonicalization; no saturation -- collapses the x/w/i
  renaming-copies and exact dups, ~10k -> ~hundreds) -> Z3-verify survivors -> the
  expensive derivability-prune on the small set -> minimal-complete core. The pre-dedup is
  a new prerequisite the intuition phase surfaced; build it before the full run.

## FULL depth-3 pipeline RUN (all cost-neutral families) -- through verification
Ran gen(all 4 relaxations, depth 3) -> pb2egg -> pre-dedup -> Z3 verify -> prune:
| stage | count |
|---|---|
| generator transfers (all-relaxed depth 3) | **849,839** |
| pb2egg valid egg rules | 36,976 |
| **pre-dedup (alpha-equivalence)** | **3,757** (10x collapse -- the safety valve worked) |
| **Z3-verified** | **2,658** (min/max 1440; 1,099 false-positives rejected) |
| redundancy-prune -> minimal core | (blocked tonight, see below) |
- The relaxation recovered the full cost-neutral closure (849k transfers vs the original
  quotiented 790); pre-dedup + Z3 cut it to 2,658 sound, alpha-distinct rules.
- Files: relaxed_d3_egg.txt (36,976), relaxed_d3_dedup.txt (3,757), relaxed_d3_verified.txt
  (2,658). prededup.py, generator toggles (taso 03825ff), tunable prune caps (tensat 4c7c77f).
- **The final prune of the 2,658 is validated (632->117 earlier today) but was blocked on
  tonight's cluster:** (1) the sbatch didn't reserve --mem, so the per-check saturation
  across ~2,657 simultaneous rules OOM-killed the job; fixed by resubmitting with
  --mem=200G. (2) Then Cuda failure 35 (cudaErrorInsufficientDriver) on every node reached
  -- rtx6k is genuinely container-incompatible, and the l40s node (g3120) was wedged from
  the session's earlier crash cycle. Driver 580 + container CUDA 12.4 are compatible, so
  this is a transient GPU-state issue, not the method. RE-RUN on a clean l40s+--mem
  allocation:  tensat -m redundancy -r relaxed_d3_verified.txt -o relaxed_d3_core.txt
  --redundancy_iters 4 --n_nodes 8000 --n_sec 4
