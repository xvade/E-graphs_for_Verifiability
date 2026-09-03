# A min/max-FREE, exact rewrite that tightens a full CROWN bound

**Goal (2026-09-02):** manually rewrite a plain-ReLU network — no min/max — so that its
**full CROWN** bound (not just IBP) gets tighter. Script:
`crown_redundancy_collapse.py` (run in the abcrown venv against real auto_LiRPA).

## The obstruction this had to break

The prior finding `plain-relu-rewrites-cant-move-crown-bound` argued (by induction from
layer 1) that any *exact* rewrite rearranging a plain-ReLU net's **linear skeleton**
preserves every ReLU's pre-activation function ⟹ same intermediate bounds ⟹ same
relaxations ⟹ **CROWN-neutral**, and that **min/max reassociation was the only door**.

**The hole:** that induction fixes the *neuron set* (up to nonneg-monomial relabeling). It
says nothing about equivalent nets with a **different number of ReLU nodes whose
pre-activations are linearly dependent** — i.e. **redundant ReLU structure**. Collapsing
that redundancy is an exact rewrite that changes the number of independently-relaxed
neurons, and CROWN is *not* invariant to it.

## The mechanism (why it moves a CROWN bound with no min/max)

CROWN relaxes each unstable ReLU **independently**, and each copy's slack is
`|coeff| · gap`. If two neurons carry the *same* pre-activation `z` (so the same gap) but
feed the output with coefficients `c1, c2`:

- **duplicated** form: slack `(|c1| + |c2|) · gap` (two independent relaxations)
- **merged** form (one neuron, coeff `c1+c2`): slack `|c1+c2| · gap`

`|c1+c2| < |c1|+|c2|` **strictly iff `sign(c1) ≠ sign(c2)`** — the merge realizes a
coefficient cancellation the duplicated form cannot see. It survives **CROWN-Optimized**
because the loosening lives in the ReLU chord's *constant offset* `−lu/(u−l)`, which α
does not control; α narrows the gap but cannot close it.

Two concrete exact, min/max-free rules:
1. **Merge proportional neurons.** Rows `w` and `βw` (β>0) feeding `c1, c2` → one neuron,
   coeff `c1+βc2`.
2. **Collapse complementary pairs** (the *natural* case — a two-sided feature, no
   artificial duplication). `relu(−z) = relu(z) − z`, so
   `c1·relu(z) + c2·relu(−z) = (c1+c2)·relu(z) − c2·z` — two unstable ReLUs become **one
   ReLU + a linear (skip) correction**.

## Result (real auto_LiRPA, 2-hidden-layer MLP 8→16→·→1, ε=1.0)

Planted unstable neurons (pre-activation gaps ~8–26), forms function-identical on 50
samples, duplicates are **distinct rows** (IBP's large gap confirms no node-sharing).
Downstream coefficients are **pinned to a moderate cancellation** (cA=1.0, cB=∓0.8 →
net 0.2, `|cA+cB|/(|cA|+|cB|)`=0.111) *identically across all pairs* — deliberately not
drawn, so no pair gets a near-zero net coeff ("merged neuron vanishes") that would
inflate the headline into a lucky seed.

| rule | form | IBP | CROWN | **CROWN-Optimized** |
|---|---|---|---|---|
| **1 merge**, opposite-sign | dup→merged width | 212.0→80.0 (−62.3%) | 132.8→53.9 (−59.4%) | **88.6→44.8 (−49.5%)** |
| **1 merge**, same-sign *(control)* | dup→merged | neutral | neutral | **neutral (0.0%)** |
| **2 collapse**, opposite-sign | base→collapsed | 249.9→183.6 (−26.5%) | 193.4→128.8 (−33.4%) | **124.9→98.9 (−20.8%)** |
| **2 collapse**, same-sign *(control)* | base→collapsed | looser* | neutral | **neutral (0.0%)** |

The **same-sign controls are CROWN-neutral** — that is what proves the mechanism is
coefficient cancellation, not a generic effect of shrinking the net. (*Rule 2's same-sign
IBP is *looser* because the skip's linear correction is re-boxed by IBP but
back-substituted exactly by CROWN — which further confirms the effect is a genuine
CROWN-relaxation phenomenon, not an IBP artifact.)

## Honest scope

- **The network is constructed to contain the structure, then rewritten** — this is a
  manual demonstration on a net *planted* with the redundancy compiled/exported nets
  exhibit, not a rewrite found on an off-the-shelf exported model. The claim is about the
  mechanism and its size, verified on the real verifier.
- The **baseline is an over-parameterized / "compiled" net** carrying redundant ReLU
  copies; the rewrite collapses them. This is not a strawman: the `tll` lift
  (`[[rewrite-verify-barrier-found-models]]`) showed real exported/compiled nets carry
  redundant relu gadgets, and its ~48% "lift" gain was exactly de-compiling such
  structure. The win is *relative to a non-canonical form*.
- **Rule 1 is what an e-graph does for free:** identical `relu(z)` subexpressions
  hashcons to one node. So tensat ingestion already *canonicalizes* this — meaning a net
  fed through the e-graph is automatically in the tighter form. Rule 2 (complementary
  collapse) is a genuine rewrite rule (not mere CSE) and is the more interesting target.
- This is a **manual** demonstration per the goal; automating rules 1–2 in tensat
  (a "merge-proportional" / "complementary-collapse" rewrite + a CROWN-gap extraction
  cost) is the obvious follow-on, **not done here**. Rule 2's collapse is
  **syntactically detectable** (rows `w` and `−w` in the same layer) and its identity
  `c1·relu(z)+c2·relu(−z) = (c1+c2)·relu(z) − c2·z` is Z3-verifiable — a clean bridge back
  to the project's tensat automation, if the goal is extended.

## What this refines

`plain-relu-rewrites-cant-move-crown-bound`'s "min/max is the only CROWN door" was true
only for **canonical / irreducible** nets (linearly-independent ReLU pre-activations).
**Redundancy-collapse is the second CROWN door**, min/max the first. Same correction
pattern as `dont-collapse-rewrite-to-fusion`: the earlier claim holds, with a named
exception.
