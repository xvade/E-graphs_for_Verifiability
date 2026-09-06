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
verifier is tighter, learned once per model, zero verification-time overhead, composable with all of the below. The gauge
symmetry itself is known (Wang & Wang 2025); using it as a learned, verification-driven rewrite was not found.

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
* **Like-for-like numbers.** Their SST 3-layer ℓ∞ +2.9 % vs our DeepT small_3 seed 0 +2.7 % (ratio of means; the diary's
  +1.7 % is the per-instance mean ratio) on the same model family and threat model; our small_6 +13.1 % (larger on 273/294,
  smaller on 0) has no counterpart — they stop at 3 layers. Their depth trend is independent evidence for our leverage rule
  (gain ≈ attention share of the CROWN width: 9 % / 40 % / 70 % at 3 / 6 / 12 layers).
* **Honest deltas.** Theirs: no tuning data, never worse than Baseline in principle (superset, modulo non-convex opt), ~5×
  time per query. Ours: one-off learning (GPU-hour scale) on dev boxes, zero verification-time overhead, no out-of-sample
  guarantee (empirically 0 reverse on small_6; neutral on the ibp-trained ViT; the SVD closed form even hurt there).

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
| small_6 | `origin` (Baseline = Shi) | 35 | 0.01686 → 0.01912 (**+13.4 %**) | 29 / 6 / 0 | median per-instance +14.6 %, range −4.7 % … +30.4 %; **same 35 instances under auto_LiRPA CROWN: 0.02835 → 0.03333 (+17.5 %), larger 35 / 0 / 0** (`results/pbv_s6_origin_crosscheck.json`); their Baseline's stock radii are about half of auto_LiRPA's on this 6-layer model (median ratio 0.51) |

Note on their own methods on our checkpoints (12:45): run as published (defaults, `--adv`), `originPlus` (PBverifierI) is −2.7 %
and `bilinear` (PBverifierT) −7.4 % against their *own* Baseline on stock DeepT small_3 — the paper's +2.9 % (SST, 3 layers)
does not reproduce on this hidden-128 DeepT checkpoint with default optimiser settings. Not tuned by us; the code-name →
paper-method mapping is inferred from `Edge.py`. The stock-vs-gauged pairs below still measure whether the gauge helps *under*
each relaxation, which is the composition question.

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
