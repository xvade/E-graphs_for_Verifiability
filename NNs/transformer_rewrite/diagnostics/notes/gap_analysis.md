# Residual gap: manual vs learned gauges on the paired protocol (2026-09-08 00:05)

Scripts: `gap_bucket.py` (full output `gap_bucket.out`), `gauge_cmp.py` (`gauge_cmp.out`) in this directory. CPU only, no repo edits.
Paired protocol = 40 **test** sentences ≤12 tokens (cmd_eval hardcodes `load_data(a, "test")`), all positions, radius by bisection + fixed-eps CROWN lbs.
Learned gauges: big_3 / small_6 trained on the same 60 SST-dev sentences ≤8 tokens (seed 0, 3 positions each); Yelp on Yelp dev.

## Part 1 — where is the learned gauge's edge?

Headline: **the gap is a label split, not a length, position, radius or eps effect.** Manual vs learned larger/smaller counts:

| model / manual gauge | overall share | label 0: share, m>l / m<l | label 1: share, m>l / m<l |
|---|---|---|---|
| big_3 unified (l1N_qk+av0) | 0.91 (100/97, eq 91) | **1.09, 100 / 0** (eq 79) | **0.69, 0 / 97** (eq 12) |
| big_3 QK-only (l1N_qk) | 0.79 (11/161) | 0.98, 11 / 55 (eq 113) | 0.56, 0 / 106 |
| big_3 closed form (svd_jacN_all) | 0.61 (0/220) | 0.78, 0 / 113 (eq 66) | 0.40, 0 / 107 |
| Yelp l1N | 0.84 (62/174) | 0.64, 3 / 105 | 0.97, 59 / 69 (eq 32) |
| Yelp closed form | 0.64 (18/225) | 0.27, 2 / 112 | 0.86, 16 / 113 |
| small_6 closed form (24-box) | 0.92 (142/133) | **0.57, 0 / 133** (eq 13) | **1.42, 142 / 0** (eq 6) |

Per-label gains vs stock (radius, ratio of means):
- big_3: label 0 manual +12.0 % / learned +11.0 %; label 1 manual +9.2 % / learned +13.2 %.
- small_6: label 0 manual +9.1 % / learned +16.1 %; label 1 manual +14.8 % / learned +10.4 %.
- Yelp: label 0 manual +4.6 % / learned +7.2 %; label 1 manual +12.9 % / learned +13.3 %.

Same at the largest eps (lb tighter, manual vs learned): big_3 unified label 0 → manual tighter on 175/179, label 1 → learned tighter on 89/109;
small_6 label 1 → manual tighter on 148/148, label 0 → learned tighter on 118/146. Verified counts at the largest eps follow the same split
(big_3 eps 0.0287: label 0 manual 30 = learned 30; label 1 manual 4 vs learned 18 — the whole 34-vs-48 deficit is label 1).

Everything else is flat:
- Token count T: big_3 unified share 0.87–1.02 across T=5…12 (T=11: 0.71 on n=58 is the one low bucket, but it is again label-driven);
  ≤8 vs >8 tokens: 0.89 vs 0.92 (big_3), 0.92 vs 0.83 (Yelp), 0.88 vs 0.93 (small_6). No sign that the learner's ≤8-token tuning set
  gives it an edge on short sentences, nor that the manual gauge is hurt by long ones.
- Position: shares 0.87–1.01 across positions 1–10 on big_3, 0.77–0.90 on Yelp, 0.78–0.99 on small_6; first/last/middle word identical.
- Stock-radius quartile: big_3 1.00 / 0.97 / 0.84 / 0.94; Yelp 1.03 / 0.94 / 0.86 / 0.72 (the Yelp deficit grows with the stock radius,
  i.e. it is the large-eps regime — but within each quartile it is still the label-0 instances that lose); small_6 0.96 / 0.83 / 0.85 / 0.99.
- eps: big_3 share of the lb gain 0.98 / 0.96 / 0.96 at the three eps; Yelp 1.01 / 0.82 / 0.84; small_6 1.09 / 1.08 / 1.11 (small_6 closed form
  is *ahead* of the learned gauge on the fixed-eps mean lb at every eps, behind on the radius mean 0.92 — the radius is dominated by the label-0 losses).

Why a label split can exist at all. The verification query bounds the signed margin logit[y] − logit[1−y]; the sign of every back-propagated
CROWN coefficient flips with y, and CROWN picks the lower or the upper relaxation of each nonlinearity (McCormick planes of QKᵀ and PV, exp /
normalisation of the softmax) by that sign. The manual construction is **sign-blind**: N_out is the margin gradient at the probe centres but it
enters the surrogate only through column norms (‖Nᵀ ob‖), and M only through row norms, so flipping the sign of any probe's margin changes
nothing. The learned gauge maximises the *signed* CROWN lb on its boxes, so it can trade one class against the other; the manual gauge cannot
see the trade and lands on a different point of it. This also explains why the two SST learners, trained on identical boxes, are skewed in
opposite directions (big_3 favours label 1, small_6 label 0): the skew is model-structural, not a class-imbalance artefact.

## Part 2 — the gauge matrices (D = G_a⁻¹ G_b per layer / head / side; `gauge_cmp.out`, `metric_cmp.out`)

Conventions: relF = ‖D − I‖_F/√d (a random rotation gives ≈1.46 at d=64); "to orthog" = ‖D − polar(D)‖_F/‖D‖_F (0 = pure rotation);
"metric distance" = ‖AAᵀ/‖AAᵀ‖ − BBᵀ/‖BBᵀ‖‖_F (0 = the two gauges differ by a rotation only; diagonal D would be CROWN-neutral — none is).

1. **Manual and learned gauges are unrelated as matrices, everywhere.** D = manual⁻¹ learned has relF 1.1–1.6, off-diagonal/diagonal energy 5–8,
   "to scalar" 1.00, "to diag" 0.98–0.99 on all three models, all layers, both sides. There is no (layer, side) where they agree; the matrix
   distance does not localise the gap (big_3 layer 0 is not special in matrix terms — but see 3).

2. **Learned ≈ closed form ∘ (large rotation) ∘ (mild stretch).** D = closed⁻¹ learned is close to orthogonal: to-orthog 0.13 / 0.14 / 0.31
   (big_3 Q, L0–L2), 0.18 / 0.18 / 0.25 (A); small_6 0.16–0.35 for L0–L4 (L5Q 0.68); Yelp 0.15–0.51. Its singular values sit in
   [0.79, 1.34] at big_3 L0 (cond 1.7). The rotation itself is large (relF ≈ 1.45 ≈ random). By contrast **the ℓ1 refinement moves the closed form
   by a stretch**, not a rotation: D = closed⁻¹ manual has to-orthog 0.4–0.55 and singular values up to 2.9 (big_3 L0), and the refined
   gauge's D against the learned one is the *least* orthogonal at layer 0 (0.91 Q / 0.89 A on big_3, 1.04 / 0.80 on small_6). AV layers ≥1 are
   the closed form itself under the unified rule (D = I there, as designed).

3. **The ℓ2 closed form does not determine the rotation at all, and the rotation is what the probe seed changes.** At the balanced optimum
   Gᵀ A = G⁻¹ B = √S Uᵀ, so G·Q has the same p=2 cost for every orthogonal Q: O(d_h) per head is a flat valley of the ℓ2 surrogate and the SVD
   picks a point in it by numerical accident. Empirically the Yelp closed forms built from probe seed 0 and seed 1 have **identical metrics**
   (distance 0.00 / 0.03 / 0.05 Q, 0.02 / 0.02 / 0.02 A) although their held-out shares were 0.66 vs 0.28 — the seed dependence lives entirely in
   the rotation. After ℓ1 refinement the seeds' metrics differ again (0.23–0.36): ℓ1 is not rotation-invariant, so it chooses a rotation, but it
   also bends the metric away from the (seed-robust, learned-like) ℓ2 one. Metric distances closed↔learned are 0.25–0.34 at big_3 L0/L1 (about as
   far as each is from the identity, so "same metric" is only approximate, cond of D ≈ 1.7–2.3) and grow with depth (big_3 L2Q 0.68, small_6
   L4–L5 Q 0.60–0.68), where CROWN's widths are known to be inflated relative to the Jacobian shapes.

4. Conditioning: learned gauges are mild (cond 1.3–3.6, big_3 L2Q 7.3); the ℓ1-refined manual gauges are 1.5–3× worse (small_6 L4/L5 Q 11.8 / 10.8,
   closed form 13.5 / 13.7 there); all far below the cond-28 overfit regime.

Agreement with the hybrid localisation: only indirect. On big_3 layer 0 the closed-form and learned *metrics* are closest (0.25) and D is most
nearly a rotation (0.13) — i.e. at layer 0 the whole gap was the rotation, which is exactly what the ℓ1 step could fix there (widths at layer 0
are the ℓ1 row norms, so ℓ1 sees the rotation correctly) → 0.61 → 0.91. Deeper, the metrics themselves drift apart and ℓ1 sees inflated widths
wrongly, consistent with ℓ1 being harmful on the AV side beyond layer 0 on small_6.

## Part 3 — hypothesis and the one candidate for a ≤2.5 h GPU round

Hypothesis. The manual procedure has the metric G Gᵀ about right (closed form, probe-seed-robust, close to the learned one) and gets the
**rotation** wrong or unstably: the ℓ2 surrogate cannot see it, the ℓ1 surrogate sees it through box widths but (i) also distorts the metric and
(ii) is sign-blind, so it cannot make the class trade the learned gauge makes — hence the clean label split of Part 1. The learned gauge's edge
is not in long sentences, positions, or large eps per se; it is a per-class rotation choice.

Candidate: **rotation-only ℓ1 refinement, `l1N_rot`** — G_l,h = G_closed,l,h · Q_l,h with Q ∈ O(d_h) (Cayley Q = (I−S)(I+S)⁻¹, S skew, S = 0 init;
Adam 400 steps as in candidate_l1, same ℓ1-with-N surrogate, no cond penalty needed since the metric — and hence cond — is the closed form's).
Test rot on both sides at every layer and the unified variant (QK all layers, AV layer 0), plus the probe-seed-1 build on Yelp, on the three
held-out radius screens (identity / learned / closed / l1N / l1N_qk+av0 / l1N_rot / l1N_rot_qk+av0). What it would show: if the share reaches
the learned gauge's on big_3/Yelp (the metric is already learned-like, so the rotation is the whole difference at layer 0) and the seed-1 Yelp
build no longer collapses, the procedure becomes "ℓ2 metric + ℓ1 rotation", one fewer knob and seed-robust; if it lands below l1N, the ℓ1
stretch was doing real work and the rotation is not separable — either outcome is informative and cheap (parameters d(d−1)/2 per head, same cost
code). Honest expectation: parity on the label the ℓ1 rotation happens to favour, not above the learned gauge overall — beating it needs a
sign-aware surrogate (per-class N_out with the McCormick plane signs, i.e. a linear-bound rather than interval cost), which is a post-maintenance
item. A cheaper post-maintenance diagnostic for that: train the learner on label-0 boxes only and on label-1 boxes only (2 × ~40 min) and
evaluate each on its own class; the per-class ceiling above the single learned gauge bounds what any sign-aware manual rule could gain.

Not done (would need the model on GPU/CPU): label bucketing of the held-out screens (the JSONs store per-box radii but not the boxes' labels)
and of the hybrid rows (hyb runs had no held-out screen).
