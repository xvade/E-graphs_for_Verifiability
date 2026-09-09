# Related work: is the attention-gauge rewrite novel?

Written 2026-09-06 after the user pointed at Huang et al. (AAAI-26). Scope: work that tightens bound-propagation / abstract-
interpretation verification of transformers, and work that exploits weight-space symmetries. "Ours" = the exact per-head
gauge rewrite `W_q ← GᵀW_q, W_k ← G⁻¹W_k, W_v ← GaᵀW_v, W_o ← W_o Ga⁻ᵀ` with G, Ga learned on tuning boxes
(`NNs/vit_rewrite/`, `NNs/transformer_rewrite/`; results in `PROGRESS.md` 2026-09-05/06 and the two READMEs).

Search depth: the AAAI-26 paper's related-work section, five web searches (2026-09-06), and the abstracts/READMEs fetched
below. Not a systematic review; claims of "not found" are bounded by that.

## One-line verdict

Every method found tightens the **verifier's abstraction** of a fixed network (better relaxations of the products, of the
softmax, branch-and-bound, refinement). Ours **rewrites the network** into an exactly equivalent one on which the *same*
verifier is tighter, learned once per model, zero verification-time overhead. The gauge symmetry itself is known (Wang &
Wang 2025); using it as a learned, verification-driven rewrite was not found. Composability with the verifier-side methods
is *empirical and conditional*: a gauge is a rewrite tuned to one relaxation, not a verifier-agnostic improvement. The
well-tuned CROWN-trained gauges (68–120 tuning boxes) transferred to every relaxation tested with no per-instance reversal
except on the neutral small_3 cases (grid below); thin verifier-trained gauges (5–13 boxes, forced by memory) were
unreliable — one of them cost −16 % under auto_LiRPA.

## Closest prior art: parameterized abstract interpretation (Huang, Wei, Isac, Wu, Wu, Barrett — AAAI-26)

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/40860 · code: https://github.com/huangdiudiu/PBVerification-for-Transformers
(a fork of Shi et al. 2020's `main.py` verifier — the same codebase DeepT's checkpoints use, cloned to
`deept_benchmarks/PBVerification/`).

* **Idea.** Parameterise the affine relaxation of every scalar product xy inside QKᵀ and V·softmax: (T) tangent planes to a
  mean-gap-optimal convex/concave quadratic bound, tangent point free; (I) a convex combination α ∈ [−1, 1] of Shi et al.'s two
  mean-gap-optimal McCormick planes (their eq. 20). Parameters are optimised per verification query by Adam (lr 0.2, ≤ 30
  steps, early stop once verified — `Verifiers/Layer.py:last_layer_optimize`) on the final margin (eq. 26); in the code's `originPlus` / `bilinear` paths
  each attention product layer is first optimised for its own width (eq. 25, `Layer.optimize`) and the last layer for the margin. Code names: `origin` = Baseline (Shi), `inner` = midpoint tangent (unoptimised), `originPlus` =
  PBverifierI, `bilinear` = PBverifierT, `hybrid`.
* **Setup.** Shi-style SST/Yelp BERTs, N ≤ 3 layers, 4 heads, hidden 256; one perturbed word embedding, ℓ1/ℓ2/ℓ∞; radius by
  bisection on 50 (SST) / 60 (Yelp) sentence-position instances (positions 1–3 of 20 random correctly classified sentences).
* **Gains** (ratio of mean radii vs Baseline): SST ℓ∞ 1 layer 0.0346 → 0.0348 (tie), 2 layers +2.0 %, 3 layers 0.0204 →
  0.0210 (+2.9 %, wins 44/50); Yelp ℓ∞ 3 layers 0.0103 → 0.0112 (+8.7 %); interval widths "advantage more pronounced for
  deeper layers". ≈ 5× slower than Baseline (100 s vs 21 s per query at 3 layers).
* **Relation to ours, verified in code.** auto_LiRPA's `MulHelper.interpolated_relaxation`
  (`alpha-beta-CROWN/complete_verifier/auto_LiRPA/operators/bivariate.py`, used by `BoundMul` and by
  `BoundLinear.bound_backward_with_weight` for the QKᵀ / V·P matmuls) is Shi's (18a,b) in plain mode and exactly the
  PBverifierI family (r_l, r_u ∈ [0,1] per product, learned in `opt_stage`) under `CROWN-Optimized`. Hence our "vanilla CROWN"
  tier *is* their Baseline and our alpha-CROWN tier already runs their interpolated per-query optimisation (final-bound
  objective; intermediate boxes fixed at plain CROWN). The alpha-tier pairings are therefore a first composition test: on top
  of the PBverifierI family the gauge still tightens 47/47 (DeepT small_6), 202/205 (small_3), 100/100 initial + 41 → 52
  alpha-verified (VNN-COMP'23 ViT pgd_2_3_16).
* **Mechanism.** Both drain the same slack (the bilinear products, and through them the softmax inputs). Theirs moves the plane
  inside the McCormick hull of the *original* per-coordinate boxes; the gauge changes *which quantities get box-concretised*
  (the CROWN box on Gᵀq is not derivable from the box on q). Neither subsumes the other: their tangent family reaches
  off-centre planes the gauge never picks; the gauge reaches boxes their method never sees. Because the gauged net is an
  ordinary net, their verifier runs on it unchanged — which is the experiment below.
* **Like-for-like numbers (same model, same verifier, same protocol).** On their SST 3-layer configuration, retrained with
  their script and verified with their code at their protocol (ℓ∞, 20 sentences, positions 1–3, `--adv`), their PBverifierI
  gives **+2.4 %** over their Baseline (20/20 wins; paper: +2.9 %, 44/50), while the gauge under that *same* Baseline gives
  **+8.4 %** (52/52 wins). Under auto_LiRPA CROWN on a different instance set the gauge gives +9.7 % (244/0/32); on DeepT
  small_3 it gives +2.7 % (ratio of means) and on small_6 +13.1 % (273/294 larger, 0 smaller) — they stop at 3 layers, so
  small_6 has no counterpart. Their depth trend is independent evidence for our leverage rule (gain ≈ attention share of the
  CROWN width: 9 % / 27 % / 40 % / 70 % for small_3 / their model / small_6 / small_12).
* **Honest deltas.** Theirs: no tuning data, never worse than Baseline in principle (superset, modulo non-convex opt), ~5×
  time per query. Ours: one-off learning (GPU-hour scale) on dev boxes, zero verification-time overhead, no out-of-sample
  guarantee (empirically 0 reverse on small_6; neutral on the ibp-trained ViT; the SVD closed form even hurt there).

### Composition: summary grids (details in the sections that follow)

**Grid 1 — CROWN-trained gauges (learned against auto_LiRPA plain CROWN on 68–120 dev boxes), gauge gain stock → gauged
under each verifier.** Cells: ratio of mean certified radii, then instances larger / smaller. DeepT rows: the 35 instances of
their protocol (12 distinct sentences, seed 0, ≤ 16 tokens; 3 duplicates from sampling with replacement); the bracketed
auto_LiRPA figure is our own 40-sentence test protocol. Their-model row: 276 test positions under auto_LiRPA, 52 instances at
their paper protocol for the Baseline, and 20 instances from 8 sentences for the two optimised variants (cost-limited).

| model (attention share of CROWN width) | auto_LiRPA CROWN | their Baseline (`origin`) | PBverifierI (`originPlus`) | PBverifierT (`bilinear`) |
|---|---|---|---|---|
| DeepT small_3 (9 %) | +1.6 % (30 / 3) [+2.7 %, 278 inst.] | −1.3 % (10 / 18) | −1.0 % (21 / 12) | +0.9 % (32 / 0) |
| DeepT small_6 (40 %) | **+17.5 %** (35 / 0) [+13.1 %, 294 inst.] | **+13.4 %** (29 / 6) | +2.9 % (18 / 17) | +3.3 % (35 / 0) |
| their model_sst_3, retrained (27 %) | **+9.7 %** (244 / 0) | **+8.4 %** (52 / 0) | +6.1 % (20 / 0) | +6.0 % (20 / 0) |

Reference, their method vs their Baseline on stock weights (what the paper claims): their model PBverifierI +2.4 % (20 / 0),
PBverifierT −1.0 % (3 / 17); DeepT small_3 −2.7 % / −7.4 %, small_6 −15.6 % / −11.8 % (see the fairness note below the
DeepT table). On the 20 their-model instances used for the optimised rows, the gauge's gain under their Baseline is +7.5 %
(20 / 0), so their per-query optimisation absorbs about 1.5 points of the gauge's gain and the rest (+6 %) survives it.

**Grid 2 — gauges learned AGAINST their verifier** (all auto_LiRPA cells are ratio of means, recomputed from the eval JSONs on 2026-09-07; the diary quotes per-instance figures for some of these rows) (`pbv_learn.py`; "PBverifierT-trained" is approximated by training
through its unoptimised midpoint-tangent centre `inner`, not through their 30-step inner Adam loop). Same cell format; cond =
max over heads of the condition number of the saved Gq / Ga; the CROWN-trained small_6 gauge is repeated for reference.

| gauge (training bound, tuning boxes) | cond Gq / Ga | auto_LiRPA CROWN | their Baseline | PBverifierI | PBverifierT |
|---|---|---|---|---|---|
| their model, Baseline-trained, 5 boxes ≤ 5 tokens | 4.4 / 2.9 | +7.8 % (240 / 0) | +6.0 % (52 / 0) | — | — |
| small_6, Baseline-trained, 13 boxes ≤ 6 tokens | **28.3 / 15.5** | **−18.8 %** (0 / 289) | +6.9 % (18 / 17) | −1.8 % (18 / 17) | −6.1 % (9 / 26) |
| small_6, tangent-trained (`inner`), 13 boxes ≤ 6 tokens | 2.9 / 4.5 | +9.4 % (266 / 0) | **+13.2 %** (35 / 0) | — | +3.3 % (35 / 0) |
| *reference:* small_6 CROWN-trained, 68 boxes ≤ 8 tokens | 4.9 / 4.5 | +13.1 % (273 / 0) | +13.4 % (29 / 6) | +2.9 % (18 / 17) | +3.3 % (35 / 0) |
| *reference:* their model CROWN-trained, 120 boxes ≤ 10 tokens | 8.7 / 2.9 | +9.7 % (244 / 0) | +8.4 % (52 / 0) | +6.1 % (20 / 0) | +6.0 % (20 / 0) |

Reading of grid 2: training against the target verifier is not by itself what makes a gauge good — the two 13-box small_6
gauges were trained on identical boxes and one generalised (tangent-trained: 35/0 in their Baseline, 266/0 under
auto_LiRPA) while the other overfit into an ill-conditioned gauge that is harmful under auto_LiRPA. The 68–120-box
CROWN-trained gauges remained best or within noise of best in every verifier. Tuning-set size and conditioning, not the
training bound, separate the good gauges from the bad one.

### Composition experiment (their verifier on stock vs gauged DeepT checkpoints)

Setup (2026-09-06 11:17): `NNs/transformer_rewrite/export_gauged_ckpt.py` folds the learned gauges into DeepT's
`sst_bert_small_3` / `small_6` and writes Shi-layout checkpoints (`deept_benchmarks/gauged_ckpts/*_{stock,gauged}`; fp32 logit
change ≤ 5.8e-7 on random inputs). `deept_benchmarks/PBVerification/run_pbv_chain.sh <3|6> 15 8 16 <versions>` runs their
`main.py --verify --p 10 (ℓ∞) --perturbed_words 1 --samples 15 --num_verify_iters 8 --max_verify_length 16 --adv --seed 0`
on stock then gauged (same seed ⇒ same 15 sentences, positions 1–3 ⇒ ≤ 45 instances). Jobs: 39672839 (small_6 `origin` +
`originPlus`, L40S), 39672840 (small_3 same, L40S, queued), 39672841 / 39672842 (small_6 / small_3 `bilinear`, A100 ckpt,
preemptible). Logs `NNs/vit_rewrite/_scratch/pbv_s{3,6}_<version>_<stock|gauged>.{log,out}`, per-instance radii
`NNs/transformer_rewrite/results/pbv_*.json`.

Results (filled in as the jobs land):

| model | their version | n | mean radius stock → gauged | larger / smaller / equal | note |
|---|---|---|---|---|---|
| small_3 | `origin` (Baseline = Shi) | 35 | 0.03667 → 0.03617 (−1.3 %) | 10 / 18 / 7 | 12 distinct sentences (3 duplicates from sampling with replacement); per-instance changes within ±9 %, median −0.2 % |
| small_3 | `originPlus` vs `origin`, **stock weights** (their method vs their Baseline on our checkpoint) | 35 | 0.03667 → 0.03569 (−2.7 %) | 12 / 11 / 12 | their optimised bounds do not beat their Baseline on DeepT small_3 with the code defaults (Adam lr 0.2, ≤ 30 steps; init α ≈ −0.96, not exactly Baseline); one instance −15 % |
| small_3 | `bilinear` vs `origin`, **stock weights** | 35 | 0.03667 → 0.03396 (−7.4 %) | 0 / 34 / 1 | their tangent-plane variant is *worse* than their Baseline on every instance but one here (code defaults; tangent init at the box midpoint, 30 Adam steps) |
| small_3 | `originPlus` (PBverifierI), stock vs gauged | 35 | 0.03569 → 0.03534 (−1.0 %) | 21 / 12 / 2 | median +0.3 %, range −6.7 … +1.4 %: neutral, like the small_3 Baseline pair |
| small_3 | `bilinear` (PBverifierT), stock vs gauged | 35 | 0.03396 → 0.03426 (+0.9 %) | 32 / 0 / 3 | small but one-sided: larger on 32, smaller on none (range 0 … +1.8 %) |
| small_6 | `bilinear` vs `origin`, **stock weights** | 35 | 0.01686 → 0.01486 (−11.8 %) | 14 / 21 / 0 | their tangent-plane variant is again below their Baseline (median −2.3 %, one instance −40 %) |
| small_6 | `bilinear` (PBverifierT), stock vs gauged | 35 | 0.01486 → 0.01535 (**+3.3 %**) | **35 / 0 / 0** | uniformly positive (range +1.2 … +6.9 %), smaller than under their Baseline (+13.4 %) |
| small_6 | `originPlus` vs `origin`, **stock weights** | 35 | 0.01686 → 0.01422 (−15.6 %) | 9 / 26 / 0 | their interpolated variant below their Baseline on small_6 too (median −6.8 %, one instance −42 %) |
| small_6 | `originPlus` (PBverifierI), stock vs gauged | 35 | 0.01422 → 0.01463 (**+2.9 %**) | 18 / 17 / 0 | positive on average, mixed per instance (median +4.5 %, range −2.9 … +13.7 %); their per-query optimisation is non-convex (30 Adam steps), so per-instance pairs are noisier than under the deterministic Baseline |
| small_6 | `origin` (Baseline = Shi) | 35 | 0.01686 → 0.01912 (**+13.4 %**) | 29 / 6 / 0 | median per-instance +14.6 %, range −4.7 % … +30.4 %; **same 35 instances under auto_LiRPA CROWN: 0.02835 → 0.03333 (+17.5 %), larger 35 / 0 / 0** (`results/pbv_s6_origin_crosscheck.json`); their Baseline's stock radii are about half of auto_LiRPA's on this 6-layer model (median ratio 0.51) |

Note on their own methods on our checkpoints (12:45): run as published (defaults, `--adv`), `originPlus` (PBverifierI) is −2.7 %
and `bilinear` (PBverifierT) −7.4 % against their *own* Baseline on stock DeepT small_3 — the paper's +2.9 % (SST, 3 layers)
does not reproduce on this hidden-128 DeepT checkpoint with default optimiser settings. Not tuned by us; the code-name →
paper-method mapping is inferred from `Edge.py`. Fairness note: this is an optimiser artefact, not a weakness of their
relaxation family — auto_LiRPA initialises the product parameters at r = 1 (exactly Shi's plane) and keeps the best iterate,
so CROWN-Optimized can never be below plain CROWN; their code initialises at α ≈ −0.96 (not the Baseline plane), optimises
each attention layer's own width before the margin (`Layer.optimize`), and tracks the best iterate only at the last layer
(`last_layer_optimize`). Same family, more fragile optimiser; on their own hidden-256 model it does reproduce (+2.4 %). **Tested 2026-09-09:** starting PBverifierI at the Baseline plane (`--init_v 0`, our flag in their `Parser.py`; X0 = 0) does not repair it on small_3 — stock 0.03544 vs Baseline 0.03667 (−3.3 %, 9 / 19), marginally worse than the published start (−2.7 %) — so the cause is the per-layer width objective with per-layer freezing and no margin-tracking against the Baseline, not the initial plane. Gauge under PBverifierI(v = 0) on small_3: +1.6 % (32 / 0 / 3). small_6 rerun pending. The stock-vs-gauged pairs below still measure whether the gauge helps *under*
each relaxation, which is the composition question.

**All DeepT pairs in (18:41).** Summary of the gauge's effect inside their verifier: small_6 Baseline +13.4 % (29/6/0),
PBverifierT +3.3 % (35/0/0), PBverifierI +2.9 % (18/17/0); small_3 Baseline −1.3 %, PBverifierI −1.0 %, PBverifierT +0.9 %
(32/0/3). So the large small_6 effect survives their Baseline nearly intact and remains positive but smaller under both of
their optimised relaxations; the small small_3 effect does not transfer. Their optimised relaxations themselves come out below
their Baseline on every stock DeepT checkpoint tested (−2.7 … −15.6 %), with code defaults.

**Headline (12:35): the small_6 gauge composes across verifiers** — in their Baseline it gives +13.4 % mean radius (larger on
29/35, smaller on 6, none equal; median +14.6 %), the same size as the +13.1 % it gives under auto_LiRPA. The large effect
transfers; only the small small_3 effect was verifier-specific (next paragraph).

First reading (11:50): in *their* Baseline the small_3 gauge is neutral-to-slightly-negative, whereas the same weights give
+2.7 % (ratio of means, 278 instances) under auto_LiRPA's CROWN. The gauge was learned against auto_LiRPA's relaxation (lse
softmax, its exp/reciprocal bounds, its intermediate-bound schedule); Shi's verifier relaxes the softmax differently, so the
boxes the gauge was tuned to shrink are not the boxes that bind there. Cross-check (`pbv_crosscheck.py`, job 39673360, A100, 12:00): the *same 35 instances* through auto_LiRPA CROWN give
0.03605 → 0.03663 (**+1.6 %**, larger on 30, smaller on 3, equal 2; `results/pbv_s3_origin_crosscheck.json`), so it is not the
instance set — the small_3 gauge's (small) gain is specific to the relaxation it was learned against. The two verifiers also
disagree on the stock radii themselves per instance (their/ours ratio ranges ~0.6–1.8, median 0.97): Shi's verifier and
auto_LiRPA are different relaxations of the same network (softmax handled as exp + reciprocal vs log-sum-exp, different
intermediate-bound choices), and a gauge tuned to one need not help the other. Implication for a fair composition test: learn the
gauge against *their* bound (their bound computation is torch autograd-differentiable — they already differentiate it w.r.t.
their relaxation parameters), which is a larger change than this session's runs; decided after the small_6 pair.

### How the two verifiers were compared (protocol)

Same instances: sentences matched by their token strings, positions 1–3 as in their code (`pbv_crosscheck.py`). Both
"Baselines" are one CROWN-style backward pass per query (no branch-and-bound, no per-query optimisation, deterministic, no
time limit), so the only protocol differences are the radius search: theirs warms up by doubling/halving from `--max_eps`,
then 8 bisection steps on [l, 2l] (resolution ≈ 0.4 % of the radius) with the `--adv` PGD-attack shortcut (an attack success
counts as unsafe without a verifier call); ours bisects [0, 0.1] for 10 steps (absolute resolution 1e-4 ≈ 0.3–0.6 % of the
radius), no attack. Both give the largest ε at which one bound call certifies the margin, so the protocol adds ≈ 0.5 % noise —
far below the per-instance disagreements (factor 0.3–2 on small_3, ≈ 0.5 on small_6), which are relaxation differences.
Cost per bound call is comparable: their Baseline ≈ 0.7 s (small_3) / 1.3 s (small_6) per call on an L40S; auto_LiRPA CROWN
≈ 0.5 s / 1.8 s per call on an A100 (`sparse_intermediate_bounds=False`); their optimised variants ≈ 3.7 s (`originPlus`) and
≈ 6.3 s (`bilinear`) per call.

### Their own model (retrained) — reproduction of their table and a gauge learned on it

Their archive has no trained checkpoints, but their training code (Shi et al.'s `--train`) and its inputs (BERT-base
folder, SST data) are in it. 13:41–13:47: trained `model_sst_3` with their script (3 layers, 4 heads, hidden 256, 3 epochs;
test accuracy 0.836 vs 84.61 % in their Table 1; note their attention has **no Q/K/V biases**, a change from Shi's code —
our harness loads such a checkpoint with zero biases, `pbv_harness_check.py` verifies the logits match before anything runs).
Launched:
* reproduction of their Table 1 row (SST, 3 layers, ℓ∞, 20 sentences, positions 1–3, 10 bisection steps, ≤ 32 tokens, `--adv`):
  `origin` vs `originPlus` vs `bilinear` on the stock model — jobs 39676401 (origin + originPlus) and 39676402 (bilinear), A100 ckpt;
  logs `pbv_rep_sst3_linf_<version>.{log,out}`, results `results/pbv_rep_sst3_linf_<version>.json`.
* gauge pipeline on that model (`pbv_model_chain.sh model_sst_3 pbv_sst3`, job 39680163): attribution → learner (dev
  sentences ≤ 10 tokens, 40 sentences × 3 positions, 120 steps × 4) → paired auto_LiRPA eval (40 test sentences ≤ 12 tokens)
  → export stock/gauged Shi-layout checkpoints → their verifier (`origin`, `originPlus`, `bilinear`) on stock and gauged with
  their protocol; logs `pbvg_pbv_sst3_*.log`, `pbv_pbv_sst3_<version>_<w>.out`, results `results/pbv_sst3_*`, `results/pbv_pbv_sst3_*`.

Results so far:

* attribution (`results/pbv_sst3_attrib.json`): attention nonlinearities = **26.7 %** of the CROWN width at the certified radius
  (39.6 % at 1.5×) — three times DeepT small_3's 9 %, between small_3 and small_6.
* learner (`gauges/pbv_sst3_seed0.pt`, 120 dev boxes ≤ 10 tokens, 17 min on an A100): mean CROWN margin at the stock radius on the
  48 scored tuning boxes +0.107 → +1.002; fp64 exactness gate 8.9e-16; max cond(G) ≈ 9.
* **out-of-sample paired auto_LiRPA eval** (40 test sentences ≤ 12 tokens, 276 positions, `results/pbv_sst3_eval_short_seed0.json`):

  | metric | stock → gauged |
  |---|---|
  | mean certified radius (ℓ∞, one word) | 0.0199 → 0.0218 (**+9.7 %** ratio of means; per-instance +7.2 %); larger on 244, smaller on 0, equal 32 |
  | verified at eps 0.02 | 131 → 153 of 276 (22 up, 0 down) |
  | verified at eps 0.03 | 36 → 79 of 276 (43 up, 0 down) |
  | bound tighter at every eps | 276/276 |

  So on *their* model family (retrained with their code), the gauge's vanilla-CROWN gain (+9.5 %) is three times the gain their
  parameterised relaxation reports on the same configuration (+2.9 %, their Table 1, SST 3 layers ℓ∞) — with the leverage rule
  again placing it correctly (27 % attention share vs 9 % for DeepT small_3 → +2.7 %, 40 % for small_6 → +13 %).
* alpha-CROWN tier (`CROWN-Optimized`, 20 it; `results/pbv_sst3_eval_alpha5_seed0.json`): OOM at 80 GB for ≤ 8 and ≤ 6-token
  sentences (hidden 256 retains far more alpha state than DeepT's 128), so only 10 sentences ≤ 5 tokens = 29 instances: gauge
  tighter on **29/29** at eps 0.02 (mean Δ +0.35) and 29/29 at 0.03 (+1.08), verified 3 → 3 / 0 → 0 (both eps exceed these short sentences' radii;
  plain CROWN already flips 0 → 3 at 0.02). Rerun at eps 0.01 / 0.015 (`results/pbv_sst3_eval_alpha5b_seed0.json`): tighter on
  29/29 at both (mean Δ +0.014 / +0.075), verified 29 → 29 / 13 → 13 — so at the alpha tier on their model the gauge is
  tighter on every one of 29 × 4 pairings and flips no verdict in this small sample.
* **their verifier, their Baseline, their protocol, their (retrained) model** (20 sentences, positions 1–3, 10 bisection
  steps, ≤ 32 tokens, `--adv`, ℓ∞; 52 instances; `results/pbv_pbv_sst3_origin_stock.json` vs `results/pbv_pbv_sst3g_origin.json`):

  | | mean certified radius |
  |---|---|
  | stock (= their Table 1 Baseline row, re-run; paper: 0.0204) | 0.02455 |
  | CROWN-trained gauge folded in | 0.02661 (**+8.4 %**; larger on **52/52**, smaller on 0; median +5.9 %, range +0.2 … +15.1 %) |

  On the configuration of their SST 3-layer row, where their parameterised relaxation reports +2.9 % over this Baseline
  (44/50 wins), the exact rewrite gives +8.4 % under the *same* Baseline with wins on every instance — and the rewrite costs
  nothing at verification time.

* **PBverifierI on their model (8 sentences, 20 instances; `results/pbv_pbv_sst3{s8,g8}_originPlus.json`):**

  | comparison (same 20 instances) | mean radius | larger / smaller |
  |---|---|---|
  | their Baseline → their PBverifierI, stock weights (= their paper's claim, reproduced here) | 0.02566 → 0.02627 (**+2.4 %**) | 20 / 0 |
  | PBverifierI: stock → CROWN-gauged weights | 0.02627 → 0.02787 (**+6.1 %**) | 20 / 0 |
  | their Baseline on stock → their PBverifierI on gauged (both contributions) | 0.02566 → 0.02787 (**+8.6 %**) | 20 / 0 |

  So on their own model family their method does reproduce (+2.4 % here vs +2.9 % in the paper; it did *not* on DeepT's
  hidden-128 checkpoints) and the gauge helps under their optimised relaxation too (+6.1 %, every instance). The third
  row is just the product of the first two on the same instances; the non-trivial composition statement is that the
  gauge's gain under their *Baseline* on these same 20 instances is +7.5 % (20/0; `pbv_compare.py` on the two `origin`
  files restricted to these instances), so their optimisation absorbs ≈ 1.5 points of the gauge's gain and ≈ 6 % survives.
* PBverifierT on their model, stock weights (same 20 instances): their Baseline → PBverifierT 0.02566 → 0.02540 (−1.0 %; larger
  on 3, smaller on 17) — consistent with their own Table 1, where PBverifierT ties the Baseline on the SST 3-layer ℓ∞ row
  (0.0205 vs 0.0204, 0 wins). Gauge under PBverifierT: 0.02540 → 0.02692 (**+6.0 %**, larger on 20/20); Baseline stock →
  PBverifierT gauged +4.9 % (19/0/1).

  Summary on their model (CROWN-trained gauge, their verifier): Baseline +8.4 % (52/52), PBverifierI +6.1 % (20/20),
  PBverifierT +6.0 % (20/20). All their-model runs with the CROWN-trained gauge are complete.

### Gauge learned AGAINST their verifier (the fair composition test) — launched 19:30

The table above uses one gauge, learned against auto_LiRPA's plain CROWN, and only asks whether it *transfers* to their
relaxations. `NNs/transformer_rewrite/pbv_learn.py` instead differentiates **their** bound: a copy of their package
(`deept_benchmarks/PBVerification_grad/`, patch in `diagnostics/pbv_grad.patch`: the `final_lb/ub` detach removed so the graph
reaches the folded weights) is driven without the `no_grad`/attack wrapper; the gauge is folded into their bias-free attention
modules as plain tensors; the objective is their margin lower bound at each box's stock certified radius *under their bound*
(same protocol as ours: dev sentences ≤ 8 tokens, 40 × 3 boxes, 120 Adam steps × 4, cond penalty, best-of-eval). Smoke test on
small_3 (6 boxes, 6 steps): margin +1.62 → +1.88, their forward's logits unchanged. Two training targets: `origin` (their
Baseline, also the α ≈ −1 starting point of PBverifierI) and `inner` (midpoint tangent planes = the unoptimised centre of the
PBverifierT family; PBverifierT's own inner Adam loop is not unrolled). Jobs (A100 ckpt, dependency-chained): small_6 ×
origin-trained → their `origin`/`originPlus`/`bilinear` + auto_LiRPA (39699642–46); small_6 × inner-trained → `bilinear`/`origin`
+ auto_LiRPA (39699647–50); their `model_sst_3` × origin-trained → their `origin` (paper protocol) + auto_LiRPA (39699651–53).
Practicalities found on the way (20:00–21:10): differentiating their backward bounds is far more memory-hungry than
auto_LiRPA's — the small_6 learner OOMs an 80 GB A100 at ≤ 8-token boxes and their hidden-256 model at ≤ 6, so the
verifier-trained gauges use ≤ 6-token (small_6: 13 boxes from 5 dev sentences) and ≤ 5-token (their model: 5 boxes) tuning
sets, far thinner than the CROWN-trained gauges' 68–120 boxes. Their pooler-tanh relaxation computes the tangent slope as
1/cosh(x)², which overflows to ∞ for wide pooler bounds and turns the whole backward pass NaN (found with
`torch.autograd.detect_anomaly`, `pbv_learn.py --anomaly 1`); the gradient copy uses the identical 1 − tanh(x)² instead
(`diagnostics/pbv_grad.patch`), and the learner additionally zeroes any remaining non-finite gradient entries — after the
fix none remained when training against `origin`, but training against `inner` still had 8 192 of the 49 152 gauge-gradient
entries non-finite at every logged step, which were zeroed (source not chased down; so the `inner`-trained gauge was
effectively trained with part of its gradient masked). Their stock radii under `origin` on these
short boxes are ≈ ⅓ of auto_LiRPA's (mean 0.0060 vs 0.0184 on ≤ 8-token dev boxes), and under `inner` smaller still (0.0047).

Results (verifier-trained gauges):

* **their model, gauge learned against their Baseline on 5 boxes (≤ 5 tokens)** — `gauges/pbvtrained_sst3_origin_seed0.pt`,
  in-sample margin +0.20 → +1.07. Under **auto_LiRPA** CROWN (276 test positions, `results/pbvtrained_sst3_origin_eval_short_seed0.json`):
  certified radius 0.0199 → 0.0214 (**+7.8 %** ratio of means, +5.7 % per-instance; larger on 240, smaller on 0, equal 36); eps 0.02 verified 131 → 151, eps 0.03
  36 → 63, 0 reverse; tighter 274–276/276. That is nearly the CROWN-trained gauge's +9.7 % from a 24× smaller tuning set and
  a different verifier — the transfer works in this direction too. Under **their Baseline at the paper protocol** (52 instances,
  `results/pbv_sst3_origin_pbvorigin.json`): 0.02455 → 0.02603 (**+6.0 %; larger on 52/52**; median +3.5 %), vs +8.4 % for the
  CROWN-trained gauge under the same verifier (head-to-head 25/27, −2.2 %). Five boxes are enough for a one-sided gain in
  their verifier, but not enough to beat the 120-box CROWN-trained gauge there.
* **small_6, gauge learned against their Baseline on 13 boxes (≤ 6 tokens)** — `gauges/pbvtrained_small6_origin_seed0.pt`,
  in-sample margin +0.66 → +2.44; the saved (best-of-eval, step 119) gauge has max cond(Gq) 28.3 and cond(Ga) 15.5, vs
  4.9 / 4.5 for the CROWN-trained small_6 gauge (a mid-training reading was ≈ 35). Under **their Baseline** (same 35 test instances as the
  earlier pairs, `results/pbv_small6_origin_pbvorigin.json`): stock 0.01686 → 0.01802 (**+6.9 %**; larger on 18, smaller on 17;
  median +7.8 %, range −10 … +42 %). That is *worse* than the CROWN-trained gauge under the same verifier (+13.4 %, 29/6/0):
  0.01912 → 0.01802 (−5.8 %, 15/20). Reading: with the memory-forced 13-box, ≤ 6-token tuning set the verifier-trained gauge
  overfits (large per-instance swings both ways), while the 68-box CROWN-trained gauge generalises; training against the
  target verifier does not, by itself, beat a well-tuned gauge transferred from auto_LiRPA. Under **auto_LiRPA** it is *harmful*: 0.0220 → 0.0179 (**−18.8 %** ratio of means, −16.3 % per-instance; smaller on 289/294,
  larger on 0; `results/pbvtrained_small6_origin_eval_short_seed0.json`) — an ill-conditioned gauge (cond up to 28) tuned on 13
  boxes to one relaxation can wreck another; this is the strongest warning in the study against thin tuning sets. Under their
  `bilinear` (PBverifierT) it is also negative: 0.01486 → 0.01395 (−6.1 %; larger 9, smaller 26; `results/pbv_small6_bilinear_pbvorigin.json`),
  where the CROWN-trained gauge was +3.3 % (35/0). Under `originPlus` (PBverifierI; `results/pbv_small6_originPlus_pbvorigin.json`): 0.01422 → 0.01397 (−1.8 %; larger 18, smaller 17;
  median +2.4 %, range −11.7 … +18.5 %), where the CROWN-trained gauge was +2.9 % (18/17); head-to-head against it −4.5 % (6/29).
  So the overfit gauge is positive only in the relaxation it was trained on, and only on average.
* **small_6, gauge learned against their midpoint-tangent bound (`inner`) on the same 13 boxes** —
  `gauges/pbvtrained_small6_inner_seed0.pt`, in-sample margin +0.66 → +2.00, saved gauge max cond 2.9 / 4.5. Under **their Baseline**
  (`results/pbv_small6_origin_pbvinner.json`): stock 0.01686 → 0.01908 (**+13.2 %; larger on 35/35, smaller on 0**; median +13.3 %,
  range +0.6 … +25.2 %) — as large as the CROWN-trained gauge's +13.4 % and cleaner per instance (35/0 vs 29/6). So from the same
  13 boxes one training bound overfit and the other generalised; with tuning sets this thin the outcome is seed/objective
  sensitive. Under **auto_LiRPA** (294 test positions, `results/pbvtrained_small6_inner_eval_short_seed0.json`): 0.0220 → 0.0241
  (**+9.4 %** ratio of means, +7.6 % per-instance; larger on 266, smaller on 0, equal 28) — vs +13.1 % for the CROWN-trained gauge on the same instances. Under
  `bilinear` (PBverifierT, the family it was trained toward; `results/pbv_small6_bilinear_pbvinner.json`): 0.01486 → 0.01535
  (**+3.3 %; larger on 35/35**), identical to the CROWN-trained gauge's +3.3 % (35/0); head-to-head 0.0 % (14/21, range
  −0.9 … +2.6 %). Training toward their tangent family bought nothing over the transferred CROWN gauge under that family.

## Other verifier-side work on transformers

| work | what it tightens | models / gains | relation to ours |
|---|---|---|---|
| Shi, Zhang, Chang, Huang, Hsieh, *Robustness Verification for Transformers*, ICLR 2020 (https://openreview.net/pdf?id=BJxwPJHFwS) | first backward (CROWN-style) bounds for transformers; mean-gap-optimal affine bounds for products, tangent bounds for exp/reciprocal | SST/Yelp BERTs ≤ 3 layers | the Baseline everywhere; our vanilla tier is this relaxation (auto_LiRPA) |
| Bonaert, Dimitrov, Baader, Vechev, *DeepT* (PLDI 2021) | multi-norm zonotopes for attention; faster than backward bounds, not tighter | `sst_bert_small_3/6/12` checkpoints we use | provides the models + spec; a gauged net can be fed to DeepT unchanged (not run) |
| Wei, Wu, Wu, Chen, Barrett, Farchi, *Convex bounds on the softmax* (AISTATS 2023, https://arxiv.org/abs/2303.01713) | softmax relaxation | complementary to AAAI-26 per its authors | orthogonal slack (softmax vs products); composable |
| Zhang, Shen, Guo, Ji, *GaLileo* (AAAI 2024, https://ojs.aaai.org/index.php/AAAI/article/view/30180) | first n-dimensional linear relaxation of softmax (keeps input dependencies) | SST/Yelp; up to 3.24× larger radii vs CROWN-BaF; multi-word perturbations | verifier-side; composable |
| Shi et al., *GenBaB* (branch-and-bound for general nonlinearities, 2024, https://arxiv.org/abs/2405.21063) | BaB over sigmoid/tanh/sin/products; ViT benchmarks | GenBaB ViTs | we tested those ViTs: attention constant over the boxes ⇒ zero gauge leverage (`NNs/vit_rewrite/README.md`) |
| Rezazadeh et al., *Vertex-Softmax* (arXiv 2605.10974, 2026) | exact optimum of softmax over a score box (vertex theorem), tightest bound from score intervals alone | MNIST/F-MNIST/CIFAR attention models; beats α-CROWN/BaB at lower cost | softmax slack only; would stack with a gauge that shrinks the score boxes |
| Liu, Zhang, Zhao, *ReLU-catalysed abstraction refinement* (CAV 2026, https://arxiv.org/abs/2605.14294) | represents dot-product bounds through ReLUs, then refines | "significant precision improvement for most tasks" | verifier-side refinement; does not modify the network |
| Huang et al., AAAI-26 (above) | parameterised product relaxations | SST/Yelp ≤ 3 layers, +2–9 % radius | closest; see above |
| α-CROWN / auto_LiRPA (Xu et al. 2021) | per-query optimisable slopes incl. `BoundMul` interpolation | — | what our alpha tier uses; already contains the AAAI-26 (I) family |

## Symmetry / reparametrisation work

| work | content | relation |
|---|---|---|
| Wang & Wang, *Complete characterisation of gauge symmetries in transformer architectures* (NeurReps 2025; https://github.com/kellywang2030/transformer-gauge-symmetry) | full gauge group ((GL(d_k))^h × (GL(d_v))^h) ⋊ S_h; validated on GPT-2 (outputs preserved to ~1e-6) | the *symmetry* we use, stated in full generality; no verification or bound-propagation use |
| Gonon, Brisebarre, Riccietti, Gribonval, *Rescaling-invariant Lipschitz bound via path metrics* (ICML 2025, https://arxiv.org/abs/2405.15006) | makes a Lipschitz bound invariant to ReLU rescaling symmetry (bound-side, not a rewrite) | same insight — non-invariant bounds are arbitrarily pessimistic — attacked by changing the bound, not the parametrisation |
| cross-layer equalisation for quantisation (Nagel et al. 2019; from background knowledge, not re-checked) | rescaling symmetry used as a *weight rewrite* to help a downstream tool (quantiser) | closest analogue of our move (rewrite for a downstream analysis), different tool and symmetry |
| QK / OV circuit invariance (transformer-circuits folklore) | W_Q W_Kᵀ and W_V W_O are the only gauge-invariant objects | why a per-head G is free |

Not found in this search: any work that learns a gauge / reparametrisation to tighten bound propagation, on transformers or
otherwise. The nearest thing in our own project is the ReLU-side "exact rewrite that moves the verifier" line
(min/max reassociation, redundancy collapse, stability fold — `PROGRESS.md` 2026-09-03/04), of which the gauge is the
attention instance.

## Sources

AAAI-26 paper https://ojs.aaai.org/index.php/AAAI/article/view/40860 · PBVerification code
https://github.com/huangdiudiu/PBVerification-for-Transformers · Shi et al. 2020 https://openreview.net/pdf?id=BJxwPJHFwS ·
GaLileo https://ojs.aaai.org/index.php/AAAI/article/view/30180 · Vertex-Softmax https://arxiv.org/abs/2605.10974 ·
ReLU-catalysed refinement https://arxiv.org/abs/2605.14294 · Gonon et al. https://arxiv.org/abs/2405.15006 · Wang & Wang
https://github.com/kellywang2030/transformer-gauge-symmetry · Wei et al. 2023 https://arxiv.org/abs/2303.01713 · GenBaB
https://arxiv.org/abs/2405.21063
