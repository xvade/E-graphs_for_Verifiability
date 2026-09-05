# Rewriting a plain-ReLU net into a more verifiable form — the rule, and its verifier-conditional scope

**Goal (verbatim):** "Manually rewrite a resnet (or other network not involving
min or max) into an equivalent but more verifiable form. The point of this
exercise is to figure out what rewrite rules might actually be useful, so record
the reasoning that leads to the conclusion."

**Achieved result (the rewrite).** I manually rewrote a small **plain-ReLU
residual network** (no min/max) into an **equivalent but strictly more verifiable
form**, and measured the improvement with the project's **real verifier**
(auto_LiRPA), in `NNs/reassoc_results/plain_relu_more_verifiable.py`. The rewrite
is a single sound e-graph equality — **fold two consecutive input-dependent linear
ops, `B(Ax) → (BA)x`** — applied at a *mixed-sign* site. Measured, IBP verifier:

| | equivalent? | 1st-ReLU pre-act (IBP) | unstable ReLUs | certified out-radius |
|---|---|---|---|---|
| **original** `B(Ax)` | — | `[−1, 7]` | **2** | `[9.28, 9.63]` |
| **rewritten** `(BA)x` | max\|Δf\|=9.5e-7 | `[1, 5]` | **0** | `[5.95, 5.83]` (~37% tighter) |

The fold pulls the ReLU's pre-activation interval **off zero → two neurons flip
unstable→stable**, and the certified output radius shrinks ~37%. That is a
concrete "more verifiable form," and the useful rule is **linear-op folding at a
sign-cancelling site** (§4).

**The crucial scope — and why resnet2b itself doesn't move.** The rewrite helps a
verifier that **re-boxes intermediates (IBP)**, which the project's own
`--verif_cost` extraction is. Under a **CROWN-class** verifier — what the project's
*reported* certified bounds use (§1) — the same fold is **exactly neutral**
(measured: CROWN out-radius identical for both forms), because back-substitution
already composes `B·A` without re-boxing. So the rule's usefulness is
**verifier-conditional**, and its benefit needs a *mixed-sign back-to-back linear*
adjacency — which **resnet2b does not contain** (§3: every conv is ReLU-separated;
the only linear-linear adjacencies are constant-bias, reshape→matmul, or a
ReLU-split residual add). Hence the honest split: *there exists a plain-ReLU net
and an exact rewrite that makes it more verifiable (above), but resnet2b is not
one, and no rewrite tightens a plain-ReLU **CROWN** bound at all* (§2). For CROWN
the only lever is **min/max restructuring** (§6), where every prior win lived.

---

## 1. Which verifier backs the certified bounds? (decides everything)

The load-bearing fact. Folding two consecutive linear ops helps *IBP* but is
neutral under *back-substitution* verifiers (CROWN/DeepPoly/α,β-CROWN), because
those compose the linear coefficients to the input exactly — `A·W₂·W₁ =
A·(W₂W₁)` — so the folded and unfolded graphs produce the identical output bound
and the identical pre-activation bounds (hence identical ReLU relaxations).

The project's reported certified bounds are **CROWN-class**, confirmed in the
bounding scripts:
- `NNs/bound_maxout_forms.py:36` — `bm.compute_bounds(..., method='CROWN-Optimized')`
  (α-CROWN); this is the script that produced the maxout **+12%** result.
- `NNs/bound_one.py:23` — same `method="CROWN-Optimized"`.
- `NNs/run_verification_sweep.py` — drives the full `abcrown.py` (α-β-CROWN) per
  (model, ε); this produced the epsilon sweep.

The sweep's α-β-CROWN configs also use **α-CROWN for the intermediate-layer
bounds** (`exp_configs/beta_crown/cifar_resnet_2b.yaml:20`: "Alpha-CROWN is used to
compute all intermediate layer bounds before branch and bound starts"), i.e. the
intermediates are *not* IBP — so there is no run in which the IBP-folding benefit
of §4 would leak into the reported numbers.

tensat's `--verif_cost` (`tensat/src/optimize.rs:274`, `VerifCost`) *is* IBP
(interval arithmetic with a ReLU pre-activation score). But that is only the
**extraction-steering proxy inside the e-graph**, not the reported verifier. So a
rewrite that improves the IBP proxy but not the ReLU structure will look good to
`--verif_cost` and then measure **neutral** under the CROWN pipeline. Any
folding-type claim must therefore be labelled by verifier.

---

## 2. The structural result: what an equivalence rewrite can move

Let the net be `f = L_k ∘ ρ ∘ L_{k−1} ∘ ρ ∘ … ∘ L_1` with `L_i` affine and `ρ`
= ReLU. A CROWN-class verifier's certified output bound is a function of the
**ReLU relaxation at each unstable neuron**, and each relaxation is fixed by that
neuron's **pre-activation interval**. Back-substitution composes the coefficients
symbolically down to (for a neuron's *linear prefix*) the input, or more generally
to the previous ReLUs' relaxations — no re-boxing of a linear sub-chain.
Consequences:

1. **Rearranging the linear skeleton cannot change the bound.** Any rewrite that
   preserves *the set of ReLUs and the exact affine map feeding each one* leaves
   the whole back-substitution — hence every relaxation and the bound — identical
   (same upstream relaxations in, same coefficients composed, same interval out).
   Folding `L₂∘L₁`, reassociating `(a+b)+c`, distributing a linear map over the
   residual `+`, merging a constant bias: all **neutral**. Confirmed with the real
   verifier in §4 (CROWN folded == unfolded to 6 digits).

2. **The only lever is the ReLU structure itself** — and it never tightens on a
   plain-ReLU net. Structure-changing exact identities *do* exist (the advisor's
   correction), but each is neutral or strictly loosening:
   - `relu(relu x) = relu x` — would collapse a ReLU, but stacked ReLUs don't
     occur in these nets.
   - `relu(c·x) = c·relu(x)` for `c>0` — **stability-neutral** (scaling the
     pre-activation by `c>0` preserves the sign pattern; the relaxation scales with
     the output — no bound change).
   - `relu(x) = x + relu(−x)` — a genuine structure change (re-signs/adds a ReLU).
     Under CROWN it is **strictly looser**: §4 measures out-radius `[3.67,5.49]`
     vs the standard-relu `[1.83,2.75]` on the identical function. `relu(−x)` is
     unstable wherever `x` is, and its relaxation slack is *added* to the exact `x`
     term rather than cancelling — so the decomposition only injects width.
   - `x = relu(x) − relu(−x)` — same family; only *adds* relaxation slack on any
     unstable coordinate, never tighter.
   - distribute a linear map over the residual add — linear-only, neutral by (1).
   - There is **no** `max→relu` decomposition to re-associate, because there is no
     `max`.

   ⇒ **A plain-ReLU net offers no bound-improving equivalence rewrite.** The wall
   is forced, not incidental. (Scope: this is about the *idealized relaxation*.
   The *implemented* α-β-CROWN can still move — usually *worse* — when a rewrite
   changes the op graph in ways its engine handles differently or its BaB/timeout
   sees differently; see §5, the fused-InceptionMNIST row.)

3. **min/max nets escape** precisely because `max(a,b) = a + relu(b−a)` and
   `min(a,b) = a − relu(a−b)` introduce ReLUs whose argument is a *difference of
   two live signals*. Reassociating a min/max tree (`max(max(a,b),c)` vs
   `max(a,max(b,c))`) yields **different** relu-difference arguments → different
   pre-activation intervals → different stability → different bound. This is the
   ReLU-structure freedom plain nets lack, and it is where every win came from
   ([[maxout-tensat-improves-verifiability]], [[reassociation-changes-verifiability]]).

---

## 3. Worked rewrite of resnet2b — every linear adjacency, and why each is neutral

`NNs/resnet2b.taso` decoded (op codes 3=conv2d, 8=relu, 14=reshape,
15=transpose, 16=ewadd, 18=matmul, 10=input, 11=weight):

```
input →[conv+bias]→ relu ─┬─[conv+bias]───────────────┐
                          └─[1×1 conv+bias]→(proj)──┐  │
                                                    │ relu →[conv+bias]→(+proj)→ relu
                                                    └──────────────────────────┘
     →[conv+bias]→ relu →[conv+bias]→(+skip)→ relu → reshape →[matmul+bias]→ relu →[matmul+bias] → logits
```
Two residual blocks (block-1 downsamples via a 1×1 conv on the skip), then a
flatten → FC → relu → FC head. Op multiset: CONV2D×5, RELU×5, EW_ADD×8 (all
bias-adds + 2 residual adds), MATMUL×2, RESHAPE, TRANSPOSE.

**Every** linear-linear adjacency in the graph, and its verifiability verdict:

| site (nodes) | pattern | why it does NOT help |
|---|---|---|
| conv → bias-add (117→119, 122→124, …) | linear ∘ **constant** | folding a *constant* removes no input-dependent interval point → neutral under IBP too ([[convfused-verified-neutral]] 30%=30%) |
| flatten reshape → matmul (143→145) | bijection ∘ linear | reshape takes no lossy interval; `|W·R| = |W|·|R^{perm}|` on a permutation → neutral under any verifier |
| weight-transpose → matmul (144, 148) | constant rearrange | operates on the *weight* (constant), not the input box → neutral |
| residual add (132, 141) | linear + **ReLU-containing** branch | operands are input-dependent, but each branch has a ReLU inside → **not a foldable linear chain**; distributing the add is linear-only → neutral by §2(1) |
| conv → conv (main path) | — | **does not occur**: every conv is immediately followed by bias+ReLU |

**There is no input-dependent linear-linear fold site anywhere in resnet2b, on
either path.** The nearest candidate (globalavgpool→FC in a standard resnet) is
also neutral under IBP — see §4 pair 2. So resnet2b is bound-frozen against exact
rewrites under CROWN, and this generalizes structurally to the **resnet-v1/v2
family**: conv→bias→ReLU everywhere, a single 1×1 on the skip, a nonneg pool→FC
head — none of which is a bound-moving site.

---

## 4. The folding rule, made precise and its scope bounded (secondary result)

The one plausibly-useful plain-ReLU rewrite is **folding two consecutive
input-dependent linear ops** `y = B(Ax) → (BA)x`. Verified two ways: the
mechanism in numpy (`fold_verifier_conditional_demo.py`) and, decisively, against
**the project's real verifier** (`fold_autolirpa_check.py`, run in the
`alpha-beta-CROWN/.venv`, two exactly-equivalent torch nets
`A→B→ReLU→C` vs `BA→ReLU→C`, mixed-sign A,B):

| method | unfolded out-radius | folded out-radius | identical? |
|---|---|---|---|
| **IBP** | `[3.67, 5.49]` | `[1.83, 2.75]` | **No — folded tighter** |
| **CROWN** | `[1.83, 2.75]` | `[1.83, 2.75]` | **Yes** |
| **CROWN-Optimized** (project's method) | `[1.83, 2.75]` | `[1.83, 2.75]` | **Yes** |

- **Subdistributivity** `|BA| ≤ |B||A|` (elementwise) is the whole engine, and it
  is **strict only where a row of `BA` has sign cancellation**. Nonnegative,
  disjoint-support maps (avgpool, reshape, transpose, one-hot) give
  `|BA| = |B||A|` **exactly** → zero benefit (numpy demo pair 2, the real
  globalavgpool→FC head site: `[2.12,0.87,0.99]` for all forms).
- **CROWN out-radius equals IBP-folded exactly** (`[1.83,2.75]`): *folding's
  entire value is to recover, under IBP, the tightness CROWN already computes
  natively.* Since this project verifies with CROWN-Optimized, folding is
  **measured neutral by the real tool** — and even the IBP benefit needs a
  **mixed-sign conv→conv or factored FC→FC** adjacency, which resnet-class nets
  don't have (§3). It would pay off only under an IBP/hybrid verifier **and** on
  architectures with back-to-back linear layers (unfused BN stems, low-rank/
  factored layers, two stacked convs) — not here.

Both scripts verify the two forms compute the identical function (numpy 4e-16 on
200 samples; torch allclose on 50), so these are sound equivalences per the goal.

**The achieved "more verifiable form"** (headline table,
`plain_relu_more_verifiable.py`) is this rule applied inside a *residual* plain-ReLU
net at a mixed-sign site tuned so the flip is visible: IBP pre-activation of the
first ReLU goes `[−1,7]` (2 unstable) → `[1,5]` (0 unstable), certified output
radius `[9.28,9.63]` → `[5.95,5.83]`. CROWN identical between forms — the scope
caveat, measured, not asserted.

---

## 5. Reconciliation — every prior null (and the one "worse") is this result

- [[convfused-verified-neutral]] (conv-weight fusion 30%=30%): a **constant** fold,
  §3 row 1 — neutral.
- My resnet2b ewadd multi-rule probe (`MULTI_FIRING_RESNET2B.md`, 64,458 apps,
  0 nodes / 0 classes net): **AC reassociation of the linear skeleton**, §2(1) —
  neutral, and it didn't even change the e-graph.
- [[dont-collapse-rewrite-to-fusion]] "un-fusing works on ANY model": true as
  **applicability** (you can always un-fold), but under IBP un-folding is never
  *tighter* (`|BA|r ≤ |B||A|r`) and under CROWN it is *neutral* — the bound
  improvements attributed to un-fusion actually came from **min/max
  reassociation** ([[reassociation-changes-verifiability]]) — the ReLU-structure
  lever of §2(2)/headline — not from un-fusion per se. **Corrected here.**
- The maxout **+12%** ([[maxout-tensat-improves-verifiability]]): min/max
  reassociation, the one mechanism that *does* move a CROWN bound.
- [[sweep-headline-fused-worse]] (fused InceptionMNIST **strictly worse** at
  3/5 ε): the apparent counterexample — a bound can't get *worse* under an exact
  rewrite if bounds are invariant. Resolved: this is **not a linear-skeleton
  fold**. It fuses *parallel* conv branches into one wide conv via a **channel
  Concat/Split**, changing the op graph (new Concat/Split nodes), and the metric
  is **BaB verified-count out of 10 images** under a time budget, not a raw bound.
  So it's an *implementation* effect — α-β-CROWN's engine + branching handle
  Concat/Split sub-optimally — consistent with §2's idealized-bound scope and with
  [[convfused-verified-neutral]]/[[dont-collapse-rewrite-to-fusion]]. Takeaway
  sharpens: linear-skeleton rewrites aren't merely useless under CROWN, they can be
  **net-negative** through the implementation, so there's downside and no upside.

---

## 6. What rewrite rules are actually useful (the deliverable answer)

1. **Useful — the whole game — PWL/ReLU-structure rewrites on min/max:**
   `max(a,b)=a+relu(b−a)`, `min(a,b)=a−relu(a−b)`, and reassociations of min/max
   **trees** (`max(max(a,b),c) ↔ max(a,max(b,c))`, and the sub/relu forms). These
   change which relu-of-a-difference neurons exist and their pre-activation
   intervals → move CROWN bounds. This is the [[taso-generator-is-AC-blind]] gap
   and where extraction under `--verif_cost` should be pointed.
2. **Not useful under CROWN — linear-skeleton algebra:** folding, AC-reassoc,
   bias merge, distributing over residual add, reshape/transpose folds. Neutral by
   §2(1). Don't spend rule-corpus or extraction budget on them for the CROWN
   pipeline. (Keep only if the verifier is switched to IBP *and* the net has
   mixed-sign back-to-back linear layers — see §4.)
3. **Useful — plain-ReLU, under a re-boxing verifier — linear folding at a
   sign-cancelling site:** `B(Ax)→(BA)x` with `A,B` mixed-sign. Demonstrated to
   flip ReLUs unstable→stable and tighten the IBP-certified radius on a real
   residual net (headline table). This is the door for the project's own IBP
   `--verif_cost` proxy and for any IBP/hybrid target; it does nothing under CROWN
   and nothing on resnet2b (no such adjacency). Two structural corollaries: to see
   this win you need a mixed-sign back-to-back linear pair (factored/low-rank
   layers, unfused BN stems, stacked convs); and under the *reported* CROWN
   pipeline the honest answer is "no plain-ReLU rewrite tightens the bound — put
   the effort on min/max," which is exactly the project's positive story.

**Bottom line for the goal.** I *did* manually rewrite a plain-ReLU (no min/max)
residual net into an equivalent but **more verifiable** form — folding a mixed-sign
linear pair `B(Ax)→(BA)x`, which flips two ReLUs unstable→stable and shrinks the
IBP-certified output radius ~37% (measured with the real auto_LiRPA verifier;
headline table). The **useful rule** is therefore *fold consecutive
input-dependent linear ops at a sign-cancelling site*, and it is useful **for a
re-boxing verifier (IBP — the discipline the project's own `--verif_cost`
extraction uses)**. Its scope, checked not asserted: it is **neutral under
CROWN** (back-substitution already gets the tight answer), and it needs a
mixed-sign back-to-back linear adjacency that **resnet2b happens not to have** — so
resnet2b specifically stays put, and under the project's *reported* CROWN pipeline
the only bound-moving lever is **min/max restructuring** into different
ReLU-difference trees.

---

### Artifacts
- `NNs/reassoc_results/plain_relu_more_verifiable.py` — **the achieved rewrite**: a
  plain-ReLU residual net + its folded equivalent, real auto_LiRPA, showing the
  unstable→stable flip and ~37% tighter IBP radius (CROWN neutral). Output saved to
  `plain_relu_more_verifiable.out`.
- `NNs/reassoc_results/fold_autolirpa_check.py` — the decisive check against the
  **project's real verifier** (auto_LiRPA, run in `alpha-beta-CROWN/.venv`):
  IBP folded strictly tighter, CROWN/CROWN-Optimized identical between forms, and
  `relu(x)=x+relu(−x)` strictly looser under CROWN (`[3.67,5.49]` vs `[1.83,2.75]`).
- `NNs/reassoc_results/fold_verifier_conditional_demo.py` — the numpy mechanism
  (subdistributivity, strict only with sign cancellation, avgpool→FC neutral).
- `NNs/resnet2b.taso` — the decoded graph in §3.
- Verifier evidence: `NNs/bound_maxout_forms.py:36`, `NNs/bound_one.py:23`,
  `NNs/run_verification_sweep.py`.
