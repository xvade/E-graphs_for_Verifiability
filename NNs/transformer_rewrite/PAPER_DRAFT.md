# Exact Attention-Gauge Rewrites Tighten Bound-Propagation Verification of Transformers

**Anonymous (draft)**

*Draft of 2026-09-07. All experiments reported as "ours" were run in this project; every number attributed to
another paper is marked as quoted.*

---

## Abstract

Bound-propagation verifiers for transformers (CROWN and its descendants) lose most of their precision at the two
bilinear products inside attention, `QKᵀ` and `A·V`, where a McCormick-style relaxation concretises each operand
into a per-coordinate interval. Those intervals are not a property of the function; they are a property of the
*basis* in which the weights happen to be written. We exploit the per-head gauge symmetry of attention — for any
invertible `G` (query/key space) and `Ga` (value space), `W_q ← W_q G`, `W_k ← W_k G⁻ᵀ`, `W_v ← W_v Ga`,
`W_o ← Ga⁻¹ W_o` leaves the network function unchanged — and *learn* the gauge by gradient ascent on the
verifier's own certified lower bound over tuning boxes disjoint from the evaluation instances. The gauge is then
folded into the weights, so the verified artefact is an ordinary network, verification-time cost is zero, and any
verifier runs on it unmodified. On the VNN-COMP 2023 `vit` benchmark (`pgd_2_3_16`, 100 instances) the unmodified
official α,β-CROWN pipeline verifies 58/100 on the stock ONNX and 64–65/100 on the gauged one (6–7
unknown→verified, 0 reverse), with initial CROWN tighter on 100/100 instances. On seven separately trained
BERT-style transformers released with DeepT we measure certified-radius gains of +1.5 % to +26.6 % under
auto_LiRPA CROWN. A diagnostic — the share of the CROWN bound width attributable to the attention nonlinearities
— acts as a *screen*: every model we measured at ≥ 27 % share gained one-sidedly, every model at ≤ 14 % was
neutral or mixed. The rewrite composes with a verifier-side method: inside Huang et al.'s (AAAI-26) own verifier,
on their own retrained SST model and at their protocol, the gauge gives +8.4 % mean certified radius over their
Baseline (52/52 instances) where their parameterised relaxation gives +2.4 % in our reproduction. We report the
negatives with the same weight: an IBP-trained competition ViT and four GenBaB ViTs where the gauge is inert, a
low-share model where it is mixed with three reverse flips, numerical cliffs that cap two results, and one
memory-forced 13-box gauge that overfits and costs −18.8 % under a verifier it was not trained on.

---

## 1. Introduction

Neural-network verification by bound propagation answers a question about a *function*, but it computes with a
*program*. CROWN and its relatives walk the computation graph, replacing each nonlinearity with a sound linear
relaxation and concretising intermediate quantities into intervals. Two programs computing the same function can
therefore receive very different certificates: precision depends on the syntax, not only on the semantics.

This is a lever that is usually left unused. If we can rewrite a network into an exactly equivalent one on which
the same verifier is tighter, we get a better certificate for free — no new relaxation, no new solver, no extra
verification time, and no soundness argument beyond the equivalence of the two programs. The broader project this
paper belongs to studies such exact rewrites with e-graph machinery (TENSAT/TASO-style rewriting over tensor
programs), asking which rewrites move a certified bound on ReLU networks and which are provably neutral. This
paper is the answer for attention.

Attention has a large, exactly characterised symmetry group. Inside one head, the query/key projections enter the
score only through `W_q W_kᵀ` and the value/output projections only through `W_v W_o`. Any invertible `G` acting
on the head's query/key space and any invertible `Ga` on its value space can be inserted and cancelled:

```
(X W_q + b_q)(X W_k + b_k)ᵀ  =  (X W_q G + b_q G)(X W_k G⁻ᵀ + b_k G⁻ᵀ)ᵀ
A (X W_v + b_v) W_o          =  A (X W_v Ga + b_v Ga)(Ga⁻¹ W_o)
```

The network function is identical. But CROWN does not see the products `W_q W_kᵀ` and `W_v W_o`; it sees the
*factors*, and it interval-bounds them one coordinate at a time before multiplying. The box on `G⁻ᵀ`-rotated keys
is not derivable from the box on the keys, so the relaxation of the two bilinear products — the largest single
source of slack on the models where attention matters — changes with `G`. We call `(G, Ga)` a **gauge**, and the
substitution above an **attention-gauge rewrite**.

Which gauge? Not a closed-form one, in general. We tried the natural closed form (an SVD-balanced factorisation
of `W_q W_kᵀ` and `W_v W_o`); it helps on one model and hurts on another. Instead we *learn* the gauge by
gradient ascent on the verifier's own certified lower bound, differentiating the whole bound computation through
the exact gauge algebra, on tuning boxes drawn from data disjoint from the evaluation instances. Learning happens
once per model, offline, at GPU-hour scale. Afterwards the gauge is folded into the weights and thrown away.

**Contributions.**

- **An exact, learned reparametrisation that improves bound-propagation verification of transformers.** The
  rewrite is function-preserving (fp64 exactness gate ≤ 1e-7 on every model, ≤ 3e-15 on the text models), adds
  zero verification-time cost, and yields an ordinary network that any verifier consumes unchanged.
- **A full-pipeline result on a competition benchmark.** On VNN-COMP 2023 `vit/pgd_2_3_16` the *unmodified*
  official α,β-CROWN pipeline verifies 58/100 stock and 64–65/100 gauged (6–7 unknown→verified, 0 reverse), with
  its two deterministic levels improved monotonically (initial CROWN tighter on 100/100; α-CROWN more specs on
  27 instances, fewer on 0). Two independently learned gauges flip the same seven instances.
- **Replication across fourteen downloaded transformers** (two competition ViTs, four GenBaB ViTs, eight
  DeepT-release BERTs), plus one model retrained from a competing paper's own script, all with their authors'
  verification specs.
- **A predictive screen.** The share of the CROWN bound width attributable to the attention nonlinearities,
  measured on the stock model before any learning, separates the models that gain (≥ 27 % share: one-sided
  gains of +6 % to +13 % in certified radius, 0 verified→unverified flips on every such model) from those that do not
  (≤ 14 % share: neutral, sometimes mixed). We preregistered predictions from it on four models and report where
  they held (direction, one-sidedness: 4/4) and where they failed (magnitude: 2/4).
- **A composition study with a concurrent verifier-side method** (Huang et al., AAAI-26). On their model, in
  their verifier, at their protocol, the gauge gives +8.4 % over their Baseline (52/52) while their own
  parameterised relaxation gives +2.4 % in our reproduction; the two partly stack (+6.1 % of the gauge's gain
  survives under their optimised relaxation).
- **Negatives and limits, reported in full.** Models where attention carries no slack and the gauge is provably
  inert; a closed-form gauge that hurts; a low-share model where the learned gauge is mixed with three reverse
  flips; NaN cliffs in the log-sum-exp softmax relaxation that cap two headline numbers; and the failure mode of
  thin tuning sets, where a 13-box gauge helps the verifier it was trained on and costs −18.8 % under another.

---

## 2. Background

### 2.1 Bound propagation and CROWN

Given a network `f`, an input region `X` (always an ℓ∞ ball here) and a linear specification `c` (a margin
`y_label − y_i`), an incomplete verifier computes `lb ≤ min_{x∈X} cᵀf(x)` and reports *verified* when `lb > 0`.
CROWN [Zhang et al., NeurIPS 2018] computes such a bound by a backward pass maintaining a linear function of the
input, replacing each nonlinearity by a sound pair of linear bounds valid over that node's precomputed interval.
α-CROWN [Xu et al., ICLR 2021] makes the relaxation slopes free parameters and optimises them per query;
β-CROWN [Wang et al., NeurIPS 2021] adds branch-and-bound (BaB). auto_LiRPA, in the α,β-CROWN distribution, is
the implementation used throughout.

### 2.2 Transformers: products and softmax

Shi et al. (ICLR 2020) gave the first backward bounds for transformers. Two facts matter here. First, the
attention score `QKᵀ` and the context `A·V` are *bilinear in perturbed quantities*: both operands depend on the
input. A backward bound must therefore relax a product `x·y` where both `x` and `y` range over intervals
`[x_l, x_u]`, `[y_l, y_u]`. The standard sound relaxations are the McCormick planes; Shi et al. select the pair
with minimal mean gap over the box, and auto_LiRPA's `MulHelper.interpolated_relaxation` implements exactly this
family, with a per-product interpolation parameter that is fixed in plain mode and optimised under
`CROWN-Optimized`. The crucial point for this paper is that the relaxation error of a McCormick plane is governed
by the *widths of the per-coordinate operand boxes*, and the operand boxes are computed on the coordinates as the
network happens to write them.

Second, softmax. auto_LiRPA offers two decompositions: `lse` (log-sum-exp, jointly) and `complex` (explicit
`max`/`sub`/`exp`/`reciprocal`). They differ a lot, in opposite directions per tier: on the ViT below, vanilla
CROWN gives −0.70 in `lse` versus −2.28 in `complex`, while α-CROWN gives −0.174 in `complex` versus −0.70 in
`lse` (the `lse` relaxation has no free slopes, so α-CROWN degenerates to plain CROWN there). We use `lse` for
all standalone CROWN work and for gauge learning; the official competition pipeline uses `complex` and we leave
it untouched. The `lse` relaxation has a numerical failure mode disclosed repeatedly below: far outside the
certified radius it returns NaN, and on the deepest/widest-attention models the bisection radius is set by that
**NaN cliff** rather than by the bound crossing zero.

### 2.3 The gauge symmetry of attention

That `W_q W_kᵀ` and `W_v W_o` are the only gauge-invariant objects of a head is folklore in the
transformer-circuits literature; Wang & Wang (NeurReps 2025) characterise the full group,
`(GL(d_k))^h × (GL(d_v))^h ⋊ S_h`, validating output invariance on GPT-2 to ~1e-6, with an interpretability
motivation. We are not aware of prior work using the symmetry to change what a verifier computes; the closest
analogues in spirit are cross-layer equalisation (Nagel et al., 2019), a rescaling-symmetry rewrite that helps a
*quantiser*, and the rescaling-invariant Lipschitz bounds of Gonon et al. (ICML 2025), which attack the same
"non-invariant bounds are arbitrarily pessimistic" problem from the bound side.

---

## 3. Method

### 3.1 The rewrite

Per head `h` and layer `l`, choose invertible `G_{l,h} ∈ GL(d_k)` and `Ga_{l,h} ∈ GL(d_v)` and set

```
W_q ← W_q G,   b_q ← b_q G,     W_k ← W_k G⁻ᵀ,  b_k ← b_k G⁻ᵀ,
W_v ← W_v Ga,  b_v ← b_v Ga,    W_o ← Ga⁻¹ W_o
```

in the row-vector convention `X W` used above (in the `nn.Linear` convention, with weights stored as
(out, in), this reads `W_q ← Gᵀ W_q`, `W_k ← G⁻¹ W_k`, `W_v ← Gaᵀ W_v`, `W_o[:, h] ← W_o[:, h] Ga⁻ᵀ`). The
scaling by `1/√d_k` and the softmax are untouched, so the attention probabilities and the layer output are
identical as functions.

**Only mixing matters.** A diagonal gauge is bound-neutral. The reason is that the McCormick planes for `x·y`
over per-coordinate boxes transform covariantly under `x → s·x`, `y → y/s` for `s > 0`: the operand boxes scale
reciprocally and the concretised product bound is unchanged. We state this as a claim, checked empirically in our
ViT harness rather than proved here; the useful consequence is that the search must range over genuinely mixing
gauges, and that the per-head condition number is the right thing to regularise (a nearly singular `G` widens one
rotated coordinate while narrowing another, and the McCormick error is not symmetric under that trade).

### 3.2 Learning the gauge

Let `B` be a set of tuning boxes and `lb_θ(B)` the verifier's certified lower bound computed on the network
gauged by `θ = (G, Ga)`. The whole bound computation is differentiable in `θ`, because the gauge enters only
through the folded weights. We maximise

```
J(θ) = mean over boxes of the certified margin lower bound   (ViT: 0.5·mean over the 9 specs + 0.5·mean of the
                                                              worst spec)   −  λ (‖G‖² + ‖G⁻¹‖²)
```

with Adam (lr 0.01), gradient-norm clipping 1.0, `λ = 1e-4`, and a best-of-held-in checkpoint. Two engineering
points were necessary. (i) auto_LiRPA's `lse` softmax bounds compute chordal slopes as
`torch.where(diff > 1e-5, num/diff, fallback)`; the unselected branch is `0/0` when `diff == 0`, masked correctly
in the forward pass but producing NaN gradients. We divide by a safe denominator instead; forward values are
bit-identical and only gradients change. (ii) `sparse_intermediate_bounds` (default `True`) makes lse-CROWN on a
12-token BERT-style transformer peak at 18.7 GiB and OOM at 16 tokens; `False` gives the *identical* bound at
0.24 GiB.

The objective is the verifier's bound, not the IBP bound (vacuous here: min lb ≈ −7·10⁵ on the ViT) and not the
bound width. For the text models each tuning box is a real sentence with one word embedding widened to *that
box's own stock certified radius*, so the stock bound sits at ≈ 0 on every tuning box and the gradient signal is
where the decision is. An early ViT gauge trained on boxes that were 82 % already-verified at initialisation was
out-of-sample neutral for exactly this reason; the fix (score a pool with stock CROWN, keep the hardest) is part
of the harness.

### 3.3 Exactness gates and folding

After learning, the gauge is folded into a copy of the weights and the resulting network is checked against the
original in fp64 at random points inside the specification boxes. Gates: ≤ 1e-7 on the ViT (4.9e-8 / 3.5e-8 for
the two learned gauges), 8.9e-16 on the DeepT SST models, 1.8e-15 to 2.7e-15 on the Yelp models. The *stored*
network is fp32, so the deployed artefact differs from the stock one by a rounding-scale amount; we measure that
too and compare it against the margins of every newly verified instance (§5.3, §7.4).

Folding means the verified object is an ordinary ONNX/PyTorch network. For the ViT we additionally write the
gauged values *into the stock ONNX graph* (structure byte-identical, 14 of 16 attention initializers change), so
no part of the gain can come from a different export path; for DeepT we emit checkpoints in Shi et al.'s
directory layout, so third-party verifiers on that codebase load stock and gauged weights identically.

### 3.4 The attention-share diagnostic

Before learning anything we ask how much of the verifier's slack the gauge could possibly reach. The gauge can
only change how CROWN relaxes `QKᵀ`, the softmax and `A·V`; it is neutral for linear operations and cannot touch
ReLUs, LayerNorm or the pooler. The diagnostic freezes the attention probabilities at their box-centre values —
which removes the `QKᵀ`-bilinear, softmax and `A·V` slack at once — and reports the fraction of the CROWN bound
width that disappears, evaluated at `ε` = the instance's stock certified radius. This is a diagnostic, not a
sound bound; frozen attention is not a valid relaxation. We call the resulting number the **attention share**.
A finer version linearises one nonlinearity at a time (§7.1).

---

## 4. Experimental setup

**Models and specifications (all downloaded; none built or modified by us apart from the gauge).**

| family | models | spec | source |
|---|---|---|---|
| VNN-COMP 2023 `vit` | `pgd_2_3_16` (2 layers, 3 heads × 16, 5 tokens), `ibp_3_3_8` (3 layers, 17 tokens) | CIFAR-10, ℓ∞ ε = 1/255, 9 margin specs, 100 instances each, 100 s timeout | competition benchmark repo |
| GenBaB (TACAS'25) | `vit_1_3`, `vit_1_6`, `vit_2_3`, `vit_2_6` | CIFAR-10 ℓ∞ ε = 1/255, vnnlib, authors' abcrown config | HF `zhouxingshi/GenBaB` |
| DeepT release (PLDI'21) | `sst_bert_small_3/6/12`, `sst_bert_big_3` (hidden 256), `sst_bert_smaller_3` (hidden 64), `yelp_bert_small_3/6`, `sst_bert_standard_layer_norm_3` | ℓ∞ ball of radius ε around ONE word embedding of a sentence; property = true-label logit stays largest | `eth-sri/DeepT` |
| Huang et al. (AAAI-26) | `model_sst_3` (3 layers, 4 heads, hidden 256, bias-free attention) — **retrained by us with their script** (test acc 0.836; their paper reports 84.61 %) | same one-word ℓ∞ protocol | their code archive ships no checkpoints |

**Verifiers and tiers.** (i) *vanilla CROWN*: auto_LiRPA `CROWN`, `lse` softmax — this is Shi et al.'s relaxation
family and is the baseline tier everywhere. (ii) *α-CROWN*: auto_LiRPA `CROWN-Optimized`, 20 iterations, which
also optimises the per-product interpolation parameters (see §6.3). (iii) *full official pipeline*: the
unmodified `abcrown.py` with the competition `vit.yaml` (complex softmax + α-CROWN 50 it + β-CROWN BaB, 100 s
per instance), each run alone on the GPU. (iv) *Huang et al.'s verifier* (a fork of Shi et al. 2020) in three
modes: `origin` = their Baseline = Shi's relaxation, `originPlus` = PBverifierI, `bilinear` = PBverifierT.

**Baselines.** In every table the baseline is the **stock weights run through the same verifier at the same tier
on the same instances**, produced by us. We never use a published number as our baseline; where we quote a
published number it is labelled as such and never enters a paired comparison.

**Metrics and the two gain conventions.** For the ViTs the metric is verified-instance counts plus paired
per-instance bound comparisons. For the text models it is the maximum certified ℓ∞ radius per (sentence,
position) instance, found by bisection on the same grid for stock and gauged, plus verified counts at fixed ε.
Radius gains use two conventions: **ratio of means** (mean gauged / mean stock − 1), the **headline convention
throughout this paper unless stated**, and **per-instance** (mean of gauged/stock − 1), in brackets. The two
differ because the gain is not uniform across instance difficulty; all numbers here were recomputed from the
saved per-instance JSON files.

"0 reverse" always means zero verified→unverified *verdict* flips at fixed ε; per-instance radius counts
(larger / smaller / equal) are reported separately, and a model can have 0 reverse flips while a handful of radii
shrink by one bisection step.

**Gauge training data (stated for every learned gauge).** ViT: 512 ε-boxes around correctly classified CIFAR-10
**train** images, disjoint from the test-set benchmark instances; 400 Adam steps, batch 32. DeepT SST models:
3 random positions per **dev** sentence of ≤ 8–10 tokens, per-box ε = that box's stock certified radius,
objective = mean CROWN(`lse`) margin lower bound, 120 steps × 4-box gradient accumulation. Yelp models: the same,
but tuning boxes come from `train.csv` (Yelp has no dev split) and evaluation from `test.csv`. Box counts:
small_3 119 (all-dev control 305), small_6 68, small_12 5, big_3 70, smaller_3 73, yelp small_3 113, yelp
small_6 115, `model_sst_3` 120; verifier-trained gauges 5 and 13 boxes (memory-forced, §6.5). Evaluation always
uses held-out **test** sentences (40 sentences, ≤ 12 tokens, 277–294 positions), most of which are longer than
anything in the tuning set.

**Hardware.** All numbers in this draft were produced on NVIDIA L40S (48 GB) or, where stated, A100 80 GB cards on UW Hyak;
the per-claim record is `PROVENANCE.md`. Time-capped counts (§5.2) are card-dependent and will be rerun on the final card;
CROWN / α-CROWN bounds are card-independent up to fp32 noise (≈ 1e-4 on lower bounds); the α-tier instance sets (§6.3) were
chosen by the 80 GB ceiling and will be extended, not replaced, on a larger card.

**Memory limits that shaped the study.** α-CROWN on DeepT `small_6` peaks at 36 GiB (5 tokens) / 62 GiB
(6 tokens) / ≈ 84 GiB (7 tokens), so the α tier for that model requires an 80 GB A100 and ≤ 6-token sentences.
The 12-layer learner OOMs the 48 GB L40S (44.4 GiB usable) at ≥ 6 tokens. Differentiating Huang et al.'s bound OOMs 80 GB at
≤ 8 tokens on `small_6`. **BaB was not run on any DeepT model**: the abcrown loader was never wired for them, but
the binding reason is memory — BaB multiplies an α-CROWN call that already needs 62 GiB by the number of live
domains. The DeepT α-CROWN rows for `small_3` and `small_6` come from two versions of our `eval_alpha` (module
reuse vs a fresh module per call); rerunning 18 `small_3` positions through the current code reproduced the
earlier bounds bit-for-bit, so the rows are comparable.

---

## 5. Results: the ViT tier

### 5.1 Vanilla CROWN

**Table 1.** Model `pgd_2_3_16` (VNN-COMP 2023 `vit`). Verifier/tier: auto_LiRPA vanilla CROWN, `lse` softmax.
Metric: instances whose 9 margin specs are all verified by the incomplete bound, plus paired per-instance
minimum-spec lower bound and bound width. Baseline: stock ONNX weights, same verifier, same instances.
Instances: all 100 benchmark instances (n = 100, 900 specs). Gauges: SVD closed form (no learning); learned on
512 CIFAR-10 **train** ε-boxes, SVD init; learned, identity init, seed 1.

| | stock | SVD closed form | learned (SVD init) | learned (id init, seed 1) |
|---|---|---|---|---|
| verified (all 9 specs) | 24/100 | 26/100 | **36/100** | **37/100** |
| mean min-spec lower bound | −0.614 | −0.525 | −0.154 | — |
| mean bound width | 4.264 | 4.089 (−4.1 %) | 3.282 (−23.0 %) | (−22.2 %) |
| paired vs stock: instances tighter | — | 100/100 | 100/100 | 100/100 |
| paired vs stock: specs tighter | — | 893/900 | 900/900 | 900/900 |
| mean Δ (min-spec lb) | — | +0.088 | +0.459 | +0.442 |
| max cond(G) | — | — | 2.8 | — (not recorded) |

Random orthogonal gauges are slightly *worse* than stock, so the SVD choice is a real basis rather than an
artefact of moving at all; the learned gauge dominates it (36 vs 26 verified, tighter on 100/100 against it).

### 5.2 The full official pipeline

**Table 2.** Model `pgd_2_3_16`. Verifier/tier: the **unmodified** official α,β-CROWN pipeline with the
competition `vit.yaml` (complex softmax, α-CROWN 50 it, β-CROWN BaB, 100 s/instance), each run alone on an L40S.
Metric: verified count at three levels, plus paired per-instance comparisons at the two deterministic levels.
Baseline: stock competition ONNX under the same pipeline (58/100). Instances: all 100 (n = 100). Gauge: learned
on 512 CIFAR-10 train ε-boxes, written into the stock ONNX graph (`patched`) or re-exported.

| level | stock | learned gauge (SVD init, patched) | learned gauge (id init, seed 1, patched) |
|---|---|---|---|
| initial complex-mode CROWN, all-9 verified | 0/100 (mean min-lb −2.52) | **13/100** (−1.06) | 12/100 |
| paired initial CROWN | — | tighter **100/100**, Δ mean +1.46, worst +0.36 | tighter 100/100, Δ mean +1.42 |
| α-CROWN (50 it, never time-capped), all-9 verified | 41/100 | **52/100** | 50/100 |
| paired α-CROWN #specs verified | — | more on **27**, fewer on **0** (net +46/900) | more on 26, fewer on 0 (net +42) |
| final verdict (α + BaB, 100 s) | 58/100 | **64/100** | **65/100** |
| verdict flips | — | unknown→verified 6, verified→unknown **0** | unknown→verified 7, verified→unknown **0** |
| mean wall time / instance | 49.5 s | 43.1 s | 41.2 s |

The exported-graph run of the same SVD-init gauge gives 65/100 and agrees with the patched run exactly at both
deterministic levels (initial CROWN Δ = 0 on 100/100; identical α-CROWN spec counts); the two runs differ on one
BaB verdict, which is time-capped noise, so the honest statement is **58 → 64/65**. The two independently learned
gauges (different initialisation, seed and step count) flip the **same seven instances**
[60, 388, 4671, 5927, 7064, 9106, 9145]: the result is a property of the rewrite family and the objective, not of
a particular `G`.

This is a transfer result: the gauge was learned against vanilla `lse`-CROWN on train boxes and carries to the
`complex` softmax decomposition, to α-optimised bounds, and through BaB. The closed-form SVD gauge through the
same pipeline moves the deterministic levels modestly (initial CROWN tighter on 99/100, 0 → 3 verified; α-CROWN
more specs on 5, fewer on 1) and the verdicts **not at all** (58 → 58): learning is what turns tightening into
verified instances.

### 5.3 Controls

- **Export path.** A stock-weights round trip through our export path changes the initial CROWN bound by max
  |Δ| = 6.3e-5 over 100 instances × 9 specs (mean 3.9e-6), verifying 0/100 either way — four to five orders below
  the rewrite's Δ (+1.46 mean). The patched-into-stock-graph variant removes the confound entirely; an identity
  "gauge" through the same patcher changes 0 of 16 initializers and is bit-identical.
- **Time caps.** The initial α-CROWN pass never hit its 30 s cap (max 21.6 s stock, 25.4 s gauged), so both
  deterministic levels are cap-free; only the BaB verdict is time-capped.
- **fp32 storage.** Sampling each box (uniform points plus corners and centre, via onnxruntime), the sup
  stock-vs-gauged logit discrepancy is 2.98e-6 on the newly verified boxes and 3.34e-6 over all 100, against a
  smallest newly verified margin of 9.0e-5 (α-only, instance 1546; BaB flips ≥ 6.9e-4, other α-only ≥ 2.1e-4) —
  at least 27× everywhere, so every new certificate transfers to the stock model.

### 5.4 The negative on the second competition ViT

**Table 3.** Model `ibp_3_3_8` (VNN-COMP 2023 `vit`, IBP-trained, 3 layers, 17 tokens). Tier: auto_LiRPA vanilla
CROWN, `lse`. Metric: verified instances and paired min-spec lb. Baseline: stock ONNX. Instances: all 100.
Gauges: SVD closed form; learned on 128 "easy" CIFAR-train boxes; learned on 192 benchmark-like hard boxes.

| gauge | verified | paired vs stock |
|---|---|---|
| stock (baseline) | 15/100 | — |
| SVD closed form | **12/100** | looser on 100/100 instances, 900/900 specs |
| learned, easy boxes | 15/100 | Δ mean +0.0006, tighter on 75/100 |
| learned, hard boxes (192, benchmark-like) | 15/100 | Δ mean +0.0006, tighter on 75/100, width −0.0 % |

The mechanism is measurable: on `ibp_3_3_8` the attention share is **3 %** (bound width 1.186 → 1.152 when the
three attention nonlinearities are linearised), against **77 %** on `pgd_2_3_16` (3.949 → 0.904). IBP training
has made the attention nearly interval-friendly; the remaining slack sits in 3 × 17 × 96 MLP ReLUs, which the
gauge cannot touch. A sampling diagnostic shows the two models are *alike* at the function level (attention
probabilities move by ≤ 0.004 across a box on both; 0.5–0.6 % of MLP ReLUs change sign), so the difference is in
relaxation looseness, not input sensitivity. We ran the stock full-tier baseline for `ibp_3_3_8` (59/100
verified, 41 unknown) to quantify the headroom; we did **not** run the gauged full tier there, the 3 % ceiling
and vanilla-tier neutrality making it moot. The closed-form gauge actively hurts this model — the cleanest
evidence that the closed form is not a principled choice.

---

## 6. Results: the DeepT family and composition

### 6.1 The main text-model table

**Table 4.** Models: seven separately trained BERT-style transformers from the DeepT release plus Huang et al.'s
retrained `model_sst_3`. Verifier/tier: auto_LiRPA vanilla CROWN, `lse` softmax. Metric: mean maximum certified
ℓ∞ radius on a one-word embedding perturbation, by bisection on an identical grid for stock and gauged; gain as
ratio of means with per-instance in brackets. Baseline: stock released weights, same verifier, same instances.
Instances: held-out test sentences (Yelp: test.csv), n given per row. Gauge: learned against auto_LiRPA vanilla
CROWN on the dev/train boxes given in the last column (never on test data).

| model (hidden/layers/data) | attention share | n | stock → gauged mean radius | gain (ratio of means) [per-inst.] | larger / smaller / equal | verified at top ε | reverse flips | tuning boxes |
|---|---|---|---|---|---|---|---|---|
| `sst_bert_smaller_3` (64/3/SST) | 13.6 % | 277 | 0.0384 → 0.0389 | +1.5 % [+1.0 %] | 140 / **115** / 22 | 88 → **85** | **3** | 73 dev |
| `sst_bert_small_3` (128/3/SST), seed 0 | 9.2 % | 278 | 0.0331 → 0.0340 | +2.7 % [+1.7 %] | 211 / 13 / 54 | 147 → 149 | 0 | 119 dev |
| `sst_bert_small_3`, seed 1 | 9.2 % | 278 | 0.0331 → 0.0336 | +1.6 % [+1.2 %] | 155 / 9 / 114 | 147 → 150 | 0 | 119 dev |
| `sst_bert_small_3`, all-dev control | 9.2 % | 278 | 0.0331 → 0.0337 | +1.8 % [+1.3 %] | 126 / 81 / 71 | 147 → 152 | 0 | 305 dev |
| `model_sst_3` (256/3/SST, Huang et al., retrained by us) | 26.7 % | 276 | 0.0199 → 0.0218 | **+9.7 %** [+7.2 %] | 244 / **0** / 32 | 36 → 79 | 0 | 120 dev |
| `sst_bert_big_3` (256/3/SST) | 33.9 % | 288 | 0.0166 → 0.0186 | **+11.9 %** [+9.3 %] | 274 / **0** / 14 | 4 → **48** | 0 | 70 dev |
| `sst_bert_small_6` (128/6/SST) | 39.7 % | 294 | 0.0220 → 0.0249 | **+13.1 %** [+10.6 %] | 273 / **0** / 21 | 41 → **95** | 0 | 68 dev |
| `yelp_bert_small_3` (128/3/Yelp) | 59.2 % | 277 | 0.0223 → 0.0245 | **+10.1 %** [+9.5 %] | 263 / 1 / 13 | 116 → **152** | 0 | 113 train |
| `yelp_bert_small_6` (128/6/Yelp) | 80.6 % | 277 | 0.0061 → 0.0066 | +7.2 % [+4.9 %] — **cliff-dominated** | 176 / 1 / 100 | 79 → 90 | 0 | 115 train |
| `sst_bert_small_12` (128/12/SST) | 70.0 % | 120 | 0.0095 → 0.0121 | +26.6 % — **cliff-limited, indicative only** | 120 / 0 / 0 | 67 → 96 (ε 0.01) | 0 | **5** dev |

`sst_bert_smaller_3`'s three reverse flips are at the top ε; it also loses one verdict at the middle ε
(167 → 169, 3 up and 1 down), so a reader counting all grid points sees four. The ε grid per model is
`0.5 / 1 / 1.5 ×` the median stock radius from the attribution run; "top ε" is the
highest of the three (for the SST 128-hidden models this rule reproduces the 0.01/0.02/0.03 grid used earlier).
Learned gauges are mild: max per-head condition numbers 2.1 (smaller_3), 4.7 (yelp small_6), 4.9/4.5 (small_6),
6.0 (yelp small_3), 8.7 (big_3, `model_sst_3`).

**Two rows carry cliff disclosures.** On `yelp_bert_small_6`, 23–46 % of positions have a stock lse-CROWN bound
that is already NaN at the evaluation ε, and the gauge does *not* move that cliff (stock NaN 64/97/127 of 277 at
the three ε; gauged 63/96/128). Restricting to positions where the comparison is meaningful: +6.2 % on the 213
positions with a finite stock bound at the smallest ε, +6.4 % on the 71 positions whose radius is a *provable*
zero crossing. Where both bounds are finite the gauge is tighter on 212/213, 174/180 and 147/148. On
`sst_bert_small_12` the bisection "radius" is where lse-CROWN turns NaN, not where the bound crosses zero (the
stock lb at the found radius is +1.2 on the tuning boxes); of the 29 newly verified positions at ε = 0.01, 14 are
zero crossings and 15 are NaN rescues, and the **zero-crossing subgroup gains +16.2 %** against the headline
+26.6 %. That row also used a 5-box gauge, the most the 12-layer learner fits on the 48 GB L40S. Both numbers are
indicative; neither is a headline.

### 6.2 The attention share as a screen

**Table 5.** The diagnostic of §3.4 across all model families. Metric: fraction of the vanilla CROWN bound width
removed when the attention probabilities are frozen at their box-centre values, at ε = the instance's stock
certified radius (a diagnostic, not a sound bound). Instances: 8 (ViTs) or 24 (text, 12 sentences × 2 positions,
seed 3). Measured on stock weights, before any gauge exists.

| model | share | gauge outcome |
|---|---|---|
| GenBaB `vit_1_3/1_6/2_3/2_6` | ≈ 0 % (softmax interval width exactly 0) | inert (random-gauge Δ ≤ 5e-5) |
| `ibp_3_3_8` | 3 % | neutral (15 → 15) |
| `sst_bert_small_3` | 9.2 % | +2.7 %, 0 reverse |
| `sst_bert_smaller_3` | 13.6 % | +1.5 %, mixed, **3 reverse** |
| `model_sst_3` (Huang et al.) | 26.7 % | +9.7 %, 0 reverse |
| `sst_bert_big_3` | 33.9 % | +11.9 %, 0 reverse |
| `sst_bert_small_6` | 39.7 % | +13.1 %, 0 reverse |
| `yelp_bert_small_3` | 59.2 % | +10.1 %, 0 reverse |
| `sst_bert_small_12` | 70.0 % | +26.6 % (cliff-limited) |
| `yelp_bert_small_6` | 80.6 % | +7.2 % (cliff-dominated) |
| `pgd_2_3_16` | 77 % | +7/100 verdicts at the full tier |

The share **screens**, it does not **calibrate**. Every model we measured with share ≥ 27 % gained one-sidedly,
with zero verified→unverified flips on every such model (plus the ViT); both models at ≤ 14 % were neutral, and
the lower-share one of the two was the *better* of them, so their ordering is noise. Above ≈ 60 % the lse NaN
cliff caps what the gauge can deliver, which is why `yelp_bert_small_6` (81 %) gains less than
`yelp_bert_small_3` (59 %). The practical rule is a threshold somewhere between 14 % and 27 %: below it, do not
gauge (nothing to gain, small risk of loss); above it, expect +6 % to +13 % certified radius.

We treated this as a prediction task rather than a post-hoc fit. For the four models added late in the study
(Yelp small_3/6, SST big_3/smaller_3) the attention share was measured and the prediction written down *before*
any learner ran. Direction and one-sidedness were predicted correctly on 4/4; magnitude was predicted correctly
for `big_3` (≈ +10 % predicted, +11.9 % measured) and `smaller_3` (the +2–3 % class, +1.5 % measured), and
over-predicted for both Yelp models. The stated falsifier for the Yelp small_3 prediction (a gain in the +2 %
class despite a 59 % share) did not occur.

### 6.3 α-CROWN tier

**Table 6.** Verifier/tier: auto_LiRPA `CROWN-Optimized` (α-CROWN, 20 iterations). Metric: verified count at
fixed ε and paired per-instance bound comparison. Baseline: stock weights, same verifier, same instances.
Instances: memory-restricted subsets of the test sets (short sentences only), n given. Gauge: the same
CROWN-trained gauge as Table 4 in each row. **BaB was not run on any of these models** (§4).

| model | instances | ε | stock → gauged verified | paired bound |
|---|---|---|---|---|
| `sst_bert_small_3` | 205 positions, ≤ 8 tokens | 0.03 | 104 → 104 | tighter 202/205, mean Δ +0.015 |
| `sst_bert_small_3` | same | 0.04 | 54 → 54 | tighter 162/205, mean Δ +0.025 |
| `sst_bert_small_6` (A100 80 GB) | 49 positions, ≤ 6 tokens | 0.02 | 24 → **27** (3 up, 0 down) | tighter **47/47** finite, mean Δ +0.33; NaN 2 → 0 |
| `model_sst_3` (A100) | 29 positions, ≤ 5 tokens | 0.01 / 0.015 / 0.02 / 0.03 | 29 → 29 / 13 → 13 / 3 → 3 / 0 → 0 | tighter **29/29** at all four ε |

The α tier is where the gauge's claim to compose with verifier-side optimisation is first tested, because
auto_LiRPA's `CROWN-Optimized` already optimises the per-product interpolation parameters — the same family
Huang et al. call PBverifierI (§6.4). The gauge tightens on top of it on every model tested, and on `small_6` it
converts that into +3 verdicts of a verified set of 24 (0 reverse) on a small sample.

### 6.4 Composition with a verifier-side method (Huang et al., AAAI-26)

Huang et al. (AAAI-26) keep the network fixed and parameterise the affine relaxation of every scalar product
inside `QKᵀ` and `V·softmax` — tangent planes to a mean-gap-optimal quadratic bound (PBverifierT) or a convex
combination of Shi et al.'s two McCormick planes (PBverifierI) — optimising the parameters per query. Reading
their code against auto_LiRPA's, their Baseline is Shi's relaxation = auto_LiRPA plain mode, and PBverifierI is
the family `CROWN-Optimized` already optimises. The two approaches drain the same slack from different sides:
theirs moves the plane inside the McCormick hull of the *original* per-coordinate boxes; the gauge changes
*which quantities get box-concretised* (the box on `Gᵀq` is not derivable from the box on `q`). Neither subsumes
the other. Because a gauged network is an ordinary network, their verifier runs on it unchanged: we folded our
gauges into Shi-layout checkpoints and ran their code, unmodified, on stock and gauged weights.

**Table 7 (grid 1).** Gauges learned against auto_LiRPA vanilla CROWN on 68–120 dev boxes (§4), evaluated under
four verifiers. Metric: ratio of mean certified radii, gauged vs stock, with (instances larger / smaller).
Baseline in every cell: **stock weights under that same verifier on those same instances**, run by us. DeepT
rows: the 35 instances of *their* protocol (12 distinct sentences, positions 1–3, ≤ 16 tokens; 3 duplicates from
sampling with replacement); the bracketed auto_LiRPA figure is our own 40-sentence protocol (n = 278/294).
`model_sst_3` row: 276 test positions under auto_LiRPA, 52 instances at their paper protocol for the Baseline, 20
instances (8 sentences) for the two optimised variants, which cost ≈ 5× the Baseline query time by their own
measurement.

| model (attention share) | auto_LiRPA CROWN | their Baseline (`origin`) | PBverifierI (`originPlus`) | PBverifierT (`bilinear`) |
|---|---|---|---|---|
| DeepT `small_3` (9 %) | +1.6 % (30 / 3) [+2.7 %, n = 278] | **−1.3 %** (10 / 18) | −1.0 % (21 / 12) | +0.9 % (32 / 0) |
| DeepT `small_6` (40 %) | **+17.5 %** (35 / 0) [+13.1 %, n = 294] | **+13.4 %** (29 / 6) | +2.9 % (18 / 17) | +3.3 % (35 / 0) |
| `model_sst_3`, retrained (27 %) | **+9.7 %** (244 / 0) | **+8.4 %** (52 / 0) | +6.1 % (20 / 0) | +6.0 % (20 / 0) |

The +17.5 % cell and the bracketed +13.1 % are the same gauge on different instance sets: the 35-instance subset
of their protocol is short-sentence-heavy and its stock radii differ, so the headline for `small_6` remains
+13.1 % on 294 positions.

**The like-for-like claim.** On *their* model, in *their* verifier, at *their* protocol (ℓ∞, 20 sentences,
positions 1–3, `--adv`, 52 instances):

| comparison, `model_sst_3`, their verifier, their protocol | mean certified radius | larger / smaller |
|---|---|---|
| their Baseline on stock weights (our re-run; **their paper reports 0.0204**) | 0.02455 | — |
| + CROWN-trained gauge, still their Baseline | **0.02661 (+8.4 %)** | **52 / 0** |
| their PBverifierI over their Baseline, stock weights, 20-instance subset (**our reproduction**; their paper reports +2.9 %, 44/50) | +2.4 % | 20 / 0 |
| the gauge under their Baseline, same 20-instance subset | +7.5 % | 20 / 0 |
| the gauge under their PBverifierI, same 20 instances | +6.1 % | 20 / 0 |
| the gauge under their PBverifierT, same 20 instances | +6.0 % | 20 / 0 |

So on the configuration where their method reports +2.9 % (and reproduces at +2.4 % for us), the exact rewrite
gives +8.4 % under the same Baseline with wins on every instance, and about 6 points of that survive when their
per-query optimisation is turned on (their optimisation absorbs ≈ 1.5 of the gauge's 7.5 points on the shared
20-instance subset). The rewrite costs nothing at verification time; their optimisation costs ≈ 5× the query
time by their own measurement.

**Composition is empirical and conditional, not guaranteed.** The `small_3` gauge does **not** transfer to Shi's
Baseline (−1.3 %, 10/18), although the same weights give +2.7 % under auto_LiRPA. A cross-check on the *same 35
instances* under auto_LiRPA gives +1.6 % (30/3), so it is the relaxation and not the instance set: a gauge is a
rewrite tuned to one relaxation, and the small `small_3` effect is relaxation-specific. The large `small_6`
effect is not (+13.4 % in their independently implemented Baseline vs +13.1 % under auto_LiRPA).

**A fairness note on their numbers.** Run as published with code defaults, their optimised variants come out
*below* their own Baseline on stock DeepT checkpoints (−2.7 % to −15.6 %). We attribute this to their optimiser's
initialisation and best-iterate tracking (auto_LiRPA initialises the product parameters exactly at Shi's plane
and keeps the best iterate, so it can never fall below plain CROWN; their code initialises at α ≈ −0.96 and
tracks the best iterate only at the last layer), not to their relaxation family — on their own hidden-256 model
it does reproduce (+2.4 %). We report these numbers only to make the stock-vs-gauged pairs in Table 7
interpretable; they are not a claim about their method.

### 6.5 Gauges trained through their verifier — and the overfitting lesson

The natural objection to Table 7 is that our gauges were trained against auto_LiRPA and only *transfer*. We
therefore differentiated *their* bound (a patched copy of their package with the detach removed, plus a fix for
their pooler-tanh slope `1/cosh²`, which overflows to ∞ and NaNs the backward pass; the gradient copy uses
`1 − tanh²`) and learned gauges directly against it. Differentiating their bounds is far more memory-hungry than
auto_LiRPA's: the learner OOMs an 80 GB A100 at ≤ 8-token boxes on `small_6` and at ≤ 6 on their hidden-256
model, so these gauges use 13 and 5 tuning boxes — 5–24× thinner than the CROWN-trained ones.

**Table 8 (grid 2).** Gauges learned *against* Huang et al.'s bounds. Metric and baseline as in Table 7 (stock
weights under the same verifier, same instances). Instance sets as in Table 7. `cond` = max over heads of the
condition number of the saved `G` / `Ga`. "Tangent-trained" approximates PBverifierT training by its
*unoptimised* midpoint-tangent centre (their 30-step inner loop is not unrolled).

| gauge (training bound, tuning boxes) | cond G / Ga | auto_LiRPA CROWN | their Baseline | PBverifierI | PBverifierT |
|---|---|---|---|---|---|
| `model_sst_3`, Baseline-trained, **5 boxes** ≤ 5 tokens | 4.4 / 2.9 | +7.8 % (240 / 0) | +6.0 % (52 / 0) | — | — |
| `small_6`, Baseline-trained, **13 boxes** ≤ 6 tokens | **28.3 / 15.5** | **−18.8 %** (0 / 289) | +6.9 % (18 / 17) | −1.8 % (18 / 17) | −6.1 % (9 / 26) |
| `small_6`, tangent-trained, **13 boxes** ≤ 6 tokens | 2.9 / 4.5 | +9.4 % (266 / 0) | **+13.2 %** (35 / 0) | — | +3.3 % (35 / 0) |
| *reference:* `small_6` CROWN-trained, 68 boxes ≤ 8 tokens | 4.9 / 4.5 | +13.1 % (273 / 0) | +13.4 % (29 / 6) | +2.9 % (18 / 17) | +3.3 % (35 / 0) |
| *reference:* `model_sst_3` CROWN-trained, 120 boxes ≤ 10 tokens | 8.7 / 2.9 | +9.7 % (244 / 0) | +8.4 % (52 / 0) | +6.1 % (20 / 0) | +6.0 % (20 / 0) |

Three readings. (i) Training against the target verifier is not by itself what makes a gauge good: the two
13-box `small_6` gauges were trained on *identical boxes* and differ only in the bound; one generalised
(tangent-trained: 35/35 in their Baseline, 266/0 under auto_LiRPA) and the other overfit into an ill-conditioned
gauge (cond 28.3) that is positive only in the relaxation it was trained on, and only on average — it costs
−18.8 % under auto_LiRPA, smaller on 289 of 294 instances. This is the strongest warning in the study against
thin tuning sets. (ii) Five boxes already produce a one-sided gain in their verifier (+6.0 %, 52/52), but not
enough to beat the 120-box CROWN-trained gauge there (+8.4 %). (iii) The transferred 68–120-box CROWN-trained
gauges were best or tied in every verifier tested. Tuning-set size and conditioning, not the training bound,
separate the good gauges from the bad one.

Caveat on (i): the tangent-trained run had 8 192 of 49 152 gauge-gradient entries non-finite at every logged
step, which the learner zeroed; that gauge was therefore trained with part of its gradient masked, and we did not
chase the source.

### 6.6 Mechanistically null models

On the four GenBaB ViTs the gauge is inert by construction of the models: over the ε = 1/255 boxes the softmax
interval width is exactly 0 on every layer of `vit_1_3`, `vit_1_6` and `vit_2_6` and on layer 0 of `vit_2_3` —
PGD training collapsed attention either to an exactly uniform token mean or to a saturated one-hot pattern the
ε-box cannot move. If attention is a constant linear map on the specification region, every gauge is
bound-neutral; measured random-gauge Δ ≤ 5e-5, and a 3-step learner run gives gradient norm ≈ 7e-7 on `G`. This
is a closed negative, not a missing experiment.

One further model from the DeepT release, `sst_bert_standard_layer_norm_3`, is **not boundable by auto_LiRPA as
built**: the full LayerNorm's `sqrt(var + ε)` trips an assertion because CROWN's linear relaxation of the squared
deviations yields a negative lower bound on the variance. This affects the *stock* model before any gauge, so it
is a verifier limitation, not a result about the rewrite; DeepT ships the variant because their zonotope verifier
handles the full norm.

---

## 7. Analysis and discussion

### 7.1 What the gauge can and cannot reach

The frozen-attention diagnostic bounds the gauge's ceiling from above but does not decompose it. A finer
diagnostic linearises *one* nonlinearity at a time at the box centre (`QKᵀ → q₀kᵀ + qk₀ᵀ − q₀k₀ᵀ`; softmax → its
centre Jacobian; `P·V` analogously) and reports the width-weighted share of the CROWN width removed.

**Table 9.** Models: three DeepT-release text transformers spanning the gain range. Verifier/tier: auto_LiRPA
vanilla CROWN, `lse`. Metric: width-weighted share of the CROWN bound width removed by each linearisation at
ε = the instance's stock certified radius (a diagnostic, not a sound bound). Baseline: the full CROWN width on
the *stock* weights — no gauge is involved. Instances: 12 test sentences × 2 positions, seed 3, restricted to
instances with finite bounds (n in the last column).

| model (gauge gain from Table 4) | frozen attention | QK only | softmax only | AV only | all three | n |
|---|---|---|---|---|---|---|
| `sst_bert_small_6` (+13.1 %) | 39.7 % | 32.6 % | 31.5 % | 28.8 % | 40.0 % | 24 |
| `yelp_bert_small_3` (+10.1 %) | 59.2 % | 35.8 % | 34.0 % | 44.3 % | 61.1 % | 24 |
| `yelp_bert_small_6` (+7.2 %, cliff-limited) | 80.6 % | 56.2 % | 71.5 % | 66.8 % | 82.3 % | 23 |

Two readings. First, the three single-nonlinearity shares overlap heavily on every model: their sum is 2.1–2.4×
the joint share, so the slack is *interactive* across layers rather than owned by any one operation — the product
boxes feed the softmax inputs and outputs, and linearising any one of them collapses much of the same width.

Second, and this was the question the split was run to answer: the decomposition does **not** explain
`yelp_bert_small_6`'s shortfall. The hypothesis was that its +7 % against an 81 % share reflects
softmax-*owned* slack the gauge cannot reach. But the model that gains most (`sst_bert_small_6`) has almost the
same softmax fraction of its joint share (0.79) as `yelp_bert_small_6` (0.87), and `yelp_bert_small_3` gains less
than `sst_bert_small_6` while having the *smallest* softmax fraction (0.56); the QK and AV fractions order the
three gains no better. No single-nonlinearity share, absolute or relative, predicts the gains. The remaining
explanation for `yelp_bert_small_6` is the one directly visible in its data — the lse NaN cliff (stock bounds
already NaN on 23 % of positions at the stock radius, 100 of 277 positions with an unmoved radius) and the very
small radius scale — not a slack the gauge structurally cannot reach.

Caveat on this diagnostic: at 1.5× the radius the softmax-only share collapses (`small_6` 4.8 %,
`yelp_bert_small_3` −2.5 %), because the centre-Jacobian linearisation of the softmax over wide score boxes is
itself a loose linear map. Single-mode shares are meaningful only near the certified radius; the coarse
frozen-attention share of Table 5 does not have this problem, as it replaces the probabilities by constants.

Two further questions are still running and deliberately not answered here: whether a gauge tuned on
out-of-distribution boxes (Yelp text through an SST model; random vocabulary sequences with the model's own
prediction as label) recovers the in-distribution gain, which would indicate the gauge is a property of the
weights rather than of the data; and whether a gauge trained on two-word perturbations beats the one-word gauge
on two-word boxes, which would indicate that gauges are specification-specific.

### 7.2 Conditioning and tuning-set size

Table 8 is the clearest evidence that the gauge's usefulness is a generalisation problem, not an optimisation
problem. Every gauge that generalised has max condition number between 2 and 9; the one that did not has 28.3.
The opposite control holds on `sst_bert_small_3`: a 305-box gauge trained on *all* dev positions achieves three
times the held-in gain of the 119-box gauge (+0.64 vs +0.20 mean tuning lb) and no extra test gain (+1.8 % vs
+2.7 % / +1.6 %). So on a low-share model the held-in/held-out gap is *not* overfitting but the attention-share
ceiling, while on a thin tuning set overfitting is real and can be catastrophic across verifiers. The
condition-number penalty is doing useful work and should probably be stronger than the 1e-4 used here.

### 7.3 Transfer across verifiers, relaxations and lengths

Tables 2, 6, 7 and 8 establish the transfer story and are not repeated here; the one clear transfer *failure* is
`small_3` into Shi's Baseline (−1.3 %). Transfer across sentence length is separate: the text gauges are tuned on
≤ 8–10-token sentences and evaluated on ≤ 12-token ones (267 of `small_6`'s 294 test positions are longer than
anything tuned on), with gains flat or growing across length bands, and a long-sentence eval on `small_3`
(251 positions, 10–32 tokens) shows its small gain fading to +0.7–0.8 % but never reversing (bounds tighter on
251/251, 0 flips).

### 7.4 Limitations

- **No out-of-sample guarantee.** The rewrite is exact, so the certificates are sound whatever gauge is chosen;
  but that a learned gauge *improves* the bound on unseen instances is an empirical finding, not a theorem. We
  observed 0 verified→unverified flips on every ≥ 27 %-share model, and 3 reverse flips on the 13.6 %-share
  model. A user who applies a gauge should verify with it and compare against the stock run — which is cheap,
  since both are ordinary networks.
- **Memory.** BaB was not run on any DeepT model. α-CROWN on `small_6` needs 62 GiB for 6-token sentences and
  ≈ 84 GiB for 7; BaB multiplies this by the number of live domains. The model where the effect is large cannot
  reach BaB on our hardware, and on `small_3`, where BaB would fit, the effect is below what a time-capped BaB
  verdict count can resolve. The ViT result does have the full official BaB pipeline.
- **Thin tuning sets** for the verifier-trained gauges (5 and 13 boxes), forced by the memory cost of
  differentiating a third-party bound. Those rows should be read as existence results, not as measurements of
  what verifier-matched training can do.
- **Numerical cliffs.** Two rows of Table 4 are limited by lse-CROWN NaNs rather than by the bound; both are
  reported with the zero-crossing subgroup alongside the headline. A NaN-free `complex`-mode evaluation would
  settle them and was not run on the text models.
- **fp32 storage.** The gauged weights are stored in fp32, so the deployed artefact differs from the stock one by
  ≈ 3e-6 (ViT) / ≤ 4.8e-7 (DeepT) in logit space over the specification boxes. Every newly verified margin we
  report is at least 27× (ViT) and typically 10⁴× (text) larger than that discrepancy, so the certificates
  transfer; but this is a check to repeat, not a property of the method.
- **Scope.** Fifteen models across four families, two text datasets, one perturbation type (one-word ℓ∞ for
  text, full-image ℓ∞ for vision), and one bound-propagation family. We did not run DeepT's own zonotope verifier on gauged weights,
  nor the gauged full tier on `ibp_3_3_8`.
- **Baseline attribution.** Our re-runs of other systems (their Baseline radius 0.02455 vs the 0.0204 in their
  paper; our retrained model's 0.836 accuracy vs their 84.61 %) differ from published values because the model is
  retrained and the sampling differs. All paired comparisons use our own re-runs on both sides.

---

## 8. Related work

**Verifier-side tightening for transformers.** Shi, Zhang, Chang, Huang & Hsieh (ICLR 2020) gave the first
backward bounds for transformers, with mean-gap-optimal affine bounds for products and tangent bounds for
`exp`/reciprocal; this relaxation is our Baseline everywhere, in auto_LiRPA and in their own code. Bonaert,
Dimitrov, Baader & Vechev (DeepT, PLDI 2021) introduce multi-norm zonotopes for attention — faster rather than
tighter — and release the checkpoints and one-word ℓ∞ protocol we use; a gauged network can be fed to their
verifier unchanged, which we did not do. Wei, Wu, Wu, Chen, Barrett & Farchi (AISTATS 2023) give convex bounds
for the softmax; Zhang, Shen, Guo & Ji (GaLileo, AAAI 2024) give the first n-dimensional linear softmax
relaxation, reporting up to 3.24× larger radii over CROWN-BaF on SST/Yelp with multi-word perturbations. Shi et
al. (GenBaB, 2024) extend branch-and-bound to general nonlinearities including products, with the ViT benchmarks
we tested. Rezazadeh et al. (Vertex-Softmax, 2026) compute the exact optimum of softmax over a score box; Liu,
Zhang & Zhao (CAV 2026) represent dot-product bounds through ReLUs and refine. α-CROWN / auto_LiRPA (Xu et al.,
ICLR 2021) provides per-query optimisable slopes, including the product interpolation that coincides with one of
Huang et al.'s families. All of these tighten the *abstraction of a fixed network*; our rewrite changes the
network, so it is orthogonal in kind and, as measured in §6.4, partly additive in practice.

**Closest prior art.** Huang, Wei, Isac, Wu, Wu & Barrett (AAAI-26) parameterise the affine relaxation of every
scalar product inside attention and optimise the parameters per query. That is the same slack, approached from
the verifier side; §6.4 compares and composes with it directly, on their model, in their verifier, at their
protocol.

**Symmetry and reparametrisation.** Wang & Wang (NeurReps 2025) characterise the complete gauge group of
transformer architectures, `(GL(d_k))^h × (GL(d_v))^h ⋊ S_h`, validating output preservation on GPT-2 to ~1e-6;
they do not consider verification. Gonon, Brisebarre, Riccietti & Gribonval (ICML 2025) make a Lipschitz bound
invariant to ReLU rescaling symmetry, addressing the same underlying observation — a bound that is not invariant
under a symmetry of the function is arbitrarily pessimistic — from the bound side rather than by rewriting.
Cross-layer equalisation (Nagel et al., 2019) uses rescaling symmetry as a weight rewrite to help a downstream
quantiser, which is the closest analogue in form to what we do, with a different symmetry and a different
downstream tool. To our knowledge, learning a gauge to move the boxes a bound-propagation verifier concretises is
new; a bounded literature search (the AAAI-26 paper's related work, five web searches, and the abstracts and
repositories cited above) found nothing of the kind, on transformers or otherwise.

**Exact rewrites for verifiability.** This paper is the attention instance of a wider line in the host project:
exact, semantics-preserving rewrites — min/max reassociation, redundancy collapse of duplicated or complementary
hidden units, stability-conditioned folds — that move certified bounds on ReLU networks, discovered and validated
with e-graph rewriting infrastructure (TENSAT/TASO). The attention gauge is by some distance the largest effect
we have found in that line, and the only one that reaches a competition model at the full verification tier.

---

## 9. Conclusion

The precision of a bound-propagation verifier on a transformer depends on a choice the network makes
arbitrarily: the basis in which each head writes its query/key and value/output projections. That choice is a
gauge freedom, it can be learned against the verifier's own bound, and the result folds back into the weights so
that the verified object is an ordinary network and verification costs nothing extra. On a competition ViT the
unmodified official α,β-CROWN pipeline goes from 58/100 to 64–65/100 verified with zero reverse flips; on seven
downloaded BERT-style transformers certified radii grow by 1.5 % to 13 % (up to 27 % where the measurement is
cliff-limited); and inside a concurrent verifier-side method's own code, on its own model and protocol, the
rewrite delivers about three times that method's reported gain while remaining partly additive with it. The
effect is not universal, and its absence is predictable in advance: measure how much of the bound width the
attention nonlinearities own, and gauge only when that share is large. Where it is small — an IBP-trained ViT,
four ViTs whose attention is constant over the specification region, a narrow 3-layer BERT — the rewrite has
nothing to move, and we say so.

---

## References

- Bonaert, G., Dimitrov, D. I., Baader, M., Vechev, M. *Fast and Precise Certification of Transformers.* PLDI
  2021. (DeepT; code and checkpoints: https://github.com/eth-sri/DeepT)
- Gonon, A., Brisebarre, N., Riccietti, E., Gribonval, R. *A rescaling-invariant Lipschitz bound based on path
  metrics.* ICML 2025. https://arxiv.org/abs/2405.15006
- Huang, D., Wei, F., Isac, O., Wu, H., Wu, M., Barrett, C. *Parameterized Abstract Interpretation for
  Transformer Verification.* AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/40860 · code:
  https://github.com/huangdiudiu/PBVerification-for-Transformers
- Liu, J., Zhang, M., Zhao, Y. *ReLU-catalysed abstraction refinement for transformer verification.* CAV 2026.
  https://arxiv.org/abs/2605.14294
- Nagel, M., van Baalen, M., Blankevoort, T., Welling, M. *Data-Free Quantization Through Weight Equalization and
  Bias Correction.* ICCV 2019. (cited from background knowledge)
- Rezazadeh, A., et al. *Vertex-Softmax: exact softmax bounds over score boxes.* 2026.
  https://arxiv.org/abs/2605.10974
- Shi, Z., Zhang, H., Chang, K.-W., Huang, M., Hsieh, C.-J. *Robustness Verification for Transformers.* ICLR
  2020. https://openreview.net/pdf?id=BJxwPJHFwS
- Shi, Z., et al. *Neural Network Verification with Branch-and-Bound for General Nonlinearities.* 2024 (GenBaB;
  TACAS 2025). https://arxiv.org/abs/2405.21063
- Wang & Wang. *A complete characterisation of gauge symmetries in transformer architectures.* NeurReps
  2025. https://github.com/kellywang2030/transformer-gauge-symmetry
- Wang, S., Zhang, H., Xu, K., Lin, X., Jana, S., Hsieh, C.-J., Kolter, J. Z. *Beta-CROWN: Efficient Bound
  Propagation with Per-neuron Split Constraints for Neural Network Robustness Verification.* NeurIPS 2021.
  (cited from general knowledge)
- Wei, F., Wu, H., Wu, M., Chen, S., Barrett, C., Farchi, E. *Convex Bounds on the Softmax Function with
  Applications to Robustness Verification.* AISTATS 2023. https://arxiv.org/abs/2303.01713
- Xu, K., Zhang, H., Wang, S., Wang, Y., Jana, S., Lin, X., Hsieh, C.-J. *Fast and Complete: Enabling Complete
  Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers.* ICLR 2021. (α-CROWN /
  auto_LiRPA)
- Zhang, H., Weng, T.-W., Chen, P.-Y., Hsieh, C.-J., Daniel, L. *Efficient Neural Network Robustness
  Certification with General Activation Functions.* NeurIPS 2018. (CROWN; cited from general knowledge)
- Zhang, Y., Shen, H., Guo, S., Ji, S. *GaLileo: General Linear Relaxation Framework for Tightening Robustness
  Certification of Transformers.* AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/30180

---

## Appendix A. Reproduction pointers

All harnesses, learned gauges and per-instance result files are in the project repository:
`NNs/vit_rewrite/` (ViT and GenBaB: `vit_model.py`, `vit_bounds.py`, `vit_gauge_opt.py`, `vit_export.py`,
`vit_patch_onnx.py`, `genbab_gauge.py`) and `NNs/transformer_rewrite/` (text models: `deept_gauge.py` with
subcommands `probe / radii / learn / eval / eval_alpha / attrib`, `export_gauged_ckpt.py`, `pbv_learn.py`,
`pbv_compare.py`). Per-instance radii are stored as JSON with keys `inst`, `stock_rad`, `gauged_rad`, `fixed`.
The only modification to a third-party verifier used for *verification* runs is none: α,β-CROWN and Huang et
al.'s code run unmodified on gauged checkpoints. Two patches exist for *learning* only — a gradient-safe
denominator in auto_LiRPA's `softmax.py` (forward values bit-identical) and a detach removal plus a
`1 − tanh²` pooler slope in a copy of Huang et al.'s package.
