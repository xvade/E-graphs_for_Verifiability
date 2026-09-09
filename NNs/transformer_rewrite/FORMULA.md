# The formula for the attention gauge

Status 2026-09-09 08:45. Goal (second phase): a *manual* procedure — weights plus a handful of random probe sequences, no
verifier in the loop — whose gauge beats the learned one (`deept_gauge.py learn`, Adam on CROWN's lower bound over tuning boxes).
What is settled: the closed form below reaches 61–64 % of the learned radius gain on the paired protocol (big_3, Yelp small_3);
refining it on the ℓ1 version of the same cost model (step 2) lifts the paired share to 0.91 on big_3 (one-sided, per-instance
tie with the learned gauge) and 0.84 on Yelp; on small_6 the closed form and the refined rule score 1.06–1.08 on held-out text
(paired: closed form 0.92; **unified rule 1.00** — mean radius 0.02487 vs 0.02486, larger on 142 / smaller on 122 head to head, 98 vs 95 verified at eps 0.03). The CROWN learner warm-started from the closed form ties the learned gauge exactly, so the learned gauge
is the optimum of its own objective: parity is the realistic target for a manual procedure, and beating it needs a different
objective. The procedure of record is the **unified rule**: closed form, then ℓ1 refinement of the QK gauge at every layer and of
the value gauge at layer 0 only (the both-sided refinement is harmful on small_6).

## The construction

For every layer l and head h, with the stock projections W_q, W_k, W_v (rows h·d_h … (h+1)·d_h − 1) and the output projection W_o
(the corresponding columns), build

    A_qk = W_q,h M_l        B_qk = W_k,h M_l               (d_h × cols)
    A_av = W_v,h M_l        B_av = W_o,hᵀ N_l              (d_h × cols)

and for each pair (A, B) take the **SVD balancing**

    Aᵀ = Q_a R_a,  Bᵀ = Q_b R_b   (thin QR)
    R_a R_bᵀ = U S Vᵀ             (SVD, d_h × d_h)
    G = R_a⁻¹ U Λ,   Λ = sqrt(max(S, 10⁻³ · S_max))

G_q,h is the gauge of the (A_qk, B_qk) pair, G_a,h of the (A_av, B_av) pair. Then fold: W_q ← G_qᵀ W_q, W_k ← G_q⁻¹ W_k, W_v ← G_aᵀ W_v,
W_o ← W_o G_a⁻ᵀ (biases likewise). This is exact for any invertible G (the network function is unchanged).

**Step 2 (`l1N`, 2026-09-07 evening): refine on the ℓ1 cost.** Starting from the closed form, minimise

    Σ_l Σ_h Σ_c ‖(G_qᵀ W_q)_c M_l‖₁ ‖(G_q⁻¹ W_k)_c M_l‖₁ + ‖(G_aᵀ W_v)_c M_l‖₁ ‖(W_o G_a⁻ᵀ)_{:,c}ᵀ N_l‖₁

with Adam (400 steps, lr 0.02, cosine-annealed; keep the best step; penalty 1e-4 on ‖G‖² + ‖G⁻¹‖²). Same inputs as step 1,
still no verifier. The reason it matters: at layer 0 the box perturbs one token, so M_0 has a single block and the width of a
functional over the box is exactly its ℓ1 row norm; the ℓ2 closed form barely moves the layer-0 width product (0.97–1.00 of
identity on all three models) while the ℓ1 optimum reaches 0.78–0.87, below the learned gauge's 0.87–0.91. With many token blocks
at deeper layers the ℓ2 proxy is close. The earlier ℓ1 refinement *without* N (`l1_jac`) lost overall because its value gauge
ignored where downstream reads; with N both sides improve.

Why this G: with A' = Gᵀ A and B' = G⁻¹ B, the rows satisfy ‖A'_c‖₂ ‖B'_c‖₂ = s_c for every diagonal Λ, and
Σ_c ‖A'_c‖ ‖B'_c‖ = ‖Aᵀ B‖_* (the nuclear norm) is the minimum of that sum over all factorisations Aᵀ B = Σ_c a_c b_cᵀ. Λ is
CROWN-neutral (a diagonal gauge does not move any bound); it is set to balance the two factors and floored so that a near-zero
singular value cannot make G singular in fp32.

**M_l, the input-side box shape** (hid × cols): the Jacobian of the layer-l input (the residual stream entering the layer) with
respect to the perturbed embedding row, at the box centre, one block per (probe sequence, token); blocks are concatenated
column-wise. Probe sequences are random whole-word entries of the model's own vocabulary (no dataset); the model's own prediction
serves as the label where one is needed. With DeepT's no_var LayerNorm (linear), M_0 is exact and deeper M_l are first-order.
Scaling every block by its box's radius (the version called `svd_jac`) or by 1 (`svd_jac_u`) gives the same gauge on small_6
and big_3; a second probe seed gives the same gauge on small_6 and big_3 (Yelp small_3: see "open").

**N_l, the output-side functionals** (hid × cols): how downstream reads the attention block's output u_l = W_o c'. Three blocks
with equal Frobenius weight:

    N_ffn  = (W₁ Γ₁ P)ᵀ                               same layer's ReLU inputs (weight-only)
    N_next = ([W_q; W_k; W_v]_{l+1} Γ₂ P Γ₁ P)ᵀ        next layer's projections through the residual, FFN branch skipped
                                                       (weight-only; last layer: the pooler)
    N_out  = ∂(margin)/∂u_l per token, at the probe centres (one backward pass per probe)

where Γ are the LayerNorm gains and P = I − 11ᵀ/hid (the linear no_var LayerNorm is Γ P u + β). N_l = I (plain column norms of
W_o) is the variant `svd_jac`; with N_l it is `svd_jacN_all`.

## The cost model behind it

CROWN relaxes each product in QKᵀ and in PV with McCormick planes whose slack is proportional to the product of the operands'
interval widths. Over a box that enters layer l as centre + M_l z (z in the unit ℓ∞ ball), the width of a linear functional aᵀx is
2‖aᵀ M_l‖₁, so the slack of head h is modelled by

    S_qk = Σ_c ‖(G_qᵀ W_q)_c M_l‖ · ‖(G_q⁻¹ W_k)_c M_l‖
    S_av = Σ_c ‖(G_aᵀ W_v)_c M_l‖ · ‖(W_o G_a⁻ᵀ)_{:,c}ᵀ N_l‖

The second factor of S_av is the point that took longest to see: the slack of context coordinate c reaches the bound through
column c of W_o *as read by the downstream backward functionals*, so the value gauge can rotate the wide coordinates into
directions downstream barely reads. On all three models the learned gauge *raises* the plain width product at layer 0 and lowers
the N-weighted one; ablations put the larger half of the learned gain on the value side.

The model is a guide, not a predictor: the ℓ2 version has the closed form above; the ℓ1 version, minimised by Adam from the SVD
init, reaches a lower surrogate and a *worse* bound on all three models, and on Yelp small_3 the learned gauge scores ≈ 1.0 on
every width surrogate while gaining a lot. The surrogate earns its place by ordering identity / learned / overfit correctly on
all three models and by producing the construction. Coupling across layers (a tighter early layer shrinks the widths of every
later one) is not in the model; M_l and N_l are taken as gauge-independent.

## Evidence (share of the learned gauge's gain; `gauge_formula.py validate`, `deept_gauge.py eval`)

Three metrics, from cheapest to the headline: **held-in** = mean CROWN lower bound at each box's stock radius on 24–96 random-token
boxes / the learner's own 48 boxes (held-in for the learned gauge only); **held-out screen** = certified radius (bisection) on
~48 boxes from 24 sentences the learner never saw (Yelp: dev; SST: test, since SST dev has no short sentences beyond the
learner's), ratio of means, with larger / smaller counts vs stock; **paired** = the standard protocol (40 dev sentences ≤ 12
tokens, 277–294 instances, ratio of means, fixed-eps verified counts, fp64 gate).

| model | gauge | held-in random / learner | held-out screen (larger / smaller) | paired radius gain (larger / smaller) |
|---|---|---|---|---|
| SST big_3 | learned | 1.00 / 1.00 | +12.7 % (46 / 0) | +11.9 % (274 / 0) |
| | svd_jacN_all (step 1) | 0.75 / 0.77 | 0.56 (45 / 0) | **0.61** (+7.2 %, 273 / 0) |
| | **l1N (step 2)** | 0.96 / 0.95 | **0.84** (46 / 0; > learned on 4) | **0.91** (+10.8 %, 276 / 0; vs learned 101 / 97) |
| | l1N, layer 0 only | 0.94 / 0.93 | 0.82 (46 / 0) | 0.87 (+10.4 %, 276 / 0; vs learned 87 / 98) |
| | **unified rule** (l1 QK all layers, l1 AV layer 0) | — | 0.83 (46 / 0) | **0.91** (+10.8 %, 276 / 0; vs learned 100 / 97) |
| | l1N, QK side only | 0.85 / 0.86 | 0.71 (45 / 0) | 0.79 (+9.4 %, 275 / 0) |
| | l1N, 1500 steps | 0.96 / 0.95 | 0.86 (46 / 0) | — |
| Yelp small_3 | learned | 1.00 / 1.00 | +10.7 % (44 / 0) | +10.1 % (263 / 1) |
| | svd_jacN_all (step 1) | 0.64 / 0.83 | 0.66 (39 / 2) | **0.64** (+6.5 %, 234 / 25) |
| | **l1N (step 2)** | 0.85 / 0.94 | **0.88** (43 / 2; > learned on 5) | **0.84** (+8.5 %, 255 / 10) |
| | unified rule | — | 0.82 (42 / 2) | — |
| | second probe seed: closed form / l1N / unified | — | 0.28 / 0.59 / 0.56 (own screen) | — |
| | l1N, layer 0 only | 0.73 / 0.88 | 0.75 (41 / 2) | 0.75 (+7.6 %, 247 / 15) |
| SST small_6 | learned | 1.00 / 1.00 | — | +13.1 % (273 / 0) |
| | svd_jacN_all (step 1) | 0.90 / 0.92 | **1.06** (41 / 0; 18 / 16; second probe seed 1.06) | **0.92** (+12.1 %, 276 / 0; vs learned 142 / 133); 2-box svd_jac 0.90 (254 / 2) |
| | l1N (step 2, both sides) | 0.83 / 0.84 | 0.52–0.54 (36 / 0) — harmful beyond layer 0 | — |
| | l1N, QK side only | 0.96 / 0.96 | 1.06 (42 / 0; head to head 17 / 12) | 0.98 (+12.9 %, 277 / 0; vs learned 142 / 124) |
| | l1N, layer 0 only | 0.91 / 0.93 | **1.08** (41 / 0; 18 / 15) | — |
| | **unified rule** (l1 QK all layers, l1 AV layer 0) | — | **1.08** (42 / 0; 18 / 11) | **1.00** (+13.1 % vs +13.1 %, 277 / 0; vs learned 142 / 122; eps-0.03 verified 98 vs 95) |

The held-out screen predicts the paired number (Yelp 0.66 → 0.64 and 0.88 → 0.84, big_3 0.56 → 0.61); the learner-box column
does not (it said 0.82–0.83 for Yelp, and 0.86 for the both-sided l1N on small_6 that scores 0.54 held-out). Fixed-eps verified counts and flips move with the radius; reverse flips are 0 everywhere; fp64 gates 1e-15.

## Where the gap is (side / layer swaps between the learned gauge and step 1, held-in)

- **big_3:** the learned *layer 0* inside the candidate gives 0.95 / 0.93; learned layers 1 or 2 give 0.76 / 0.78 and 0.81 / 0.84.
  The candidate's deeper layers already match the learned ones; the gap was layer 0, where M_0 is exact — hence the norm, not the
  box shape. Step 2 closes it (layer-0 splice alone 0.94 / 0.93).
- **small_6:** learned QK + candidate AV = 1.01 / 1.00; candidate QK + learned AV = 0.88 / 0.91. The whole gap is the QK side,
  spread over the layers (no single-layer swap moves more than 0.05).
- **Yelp small_3:** diffuse — learned QK + candidate AV 0.88 (held-out screen), learned layer 0 / 1 / 2 in the candidate 0.76 / 0.80 /
  0.75, candidate layer 0 / 1 / 2 in the learned gauge 0.89 / 0.85 / 0.90. Step 2 gets 0.88 on the screen; the layer-0 splice 0.75.

## What the residual gap is (diagnostics, 2026-09-08 00:00–00:15, sub-agent reports in the job tmp dir)

1. **A label split, not a length / position / eps effect.** Bucketing the paired results: big_3 unified rule vs learned gauge
   is 100 / 0 on the label-0 instances it does not tie (share 1.09) and 0 / 97 on label 1 (share 0.69); the eps-0.0287 verified
   deficit (34 vs 48) is entirely label 1 (4 vs 18). small_6 closed form: label 0 share 0.57 (0 / 133), label 1 share 1.42
   (142 / 0). Yelp `l1N`: 0.64 / 0.97. Token-count, position, stock-radius and eps buckets are flat (0.87–1.02 on big_3). The
   manual construction is sign-blind (M and N enter through norms only); the CROWN learner maximises the *signed* margin bound
   and trades one class against the other — in opposite directions on the two SST models, so it is model-structural.
2. **Learned ≈ closed form ∘ rotation.** D = closed⁻¹ · learned is nearly orthogonal (distance to the polar factor 0.13–0.35;
   singular values 0.79–1.34 at big_3 layer 0), while the ℓ1 refinement moves the closed form by a large stretch (singular
   values up to 2.9; cond 30 on small_6 vs 4.9 learned). The ℓ2 closed form has a flat O(d_h) valley per head (Gᵀ A = G⁻¹ B =
   √S Uᵀ for every G · Q), so its rotation is a numerical accident — and the rotation is exactly what the probe seed changes:
   the Yelp seed-0 and seed-1 closed forms have identical metrics G Gᵀ (distance 0.00–0.05) with held-out shares 0.66 vs 0.28.
3. **Objective mismatch, not optimiser failure.** The ℓ1-with-N surrogate at the learned gauge is *higher* than at the manual
   optimum on every model, layer and side (relative to identity: big_3 0.679 vs 0.576, Yelp 0.971 vs 0.947, small_6 0.744 vs
   0.731; big_3 layer-0 QK: closed form 0.970 → `l1N` 0.795, learned 0.865). The surrogate orders gauged < identity correctly but
   ranks manual below learned; more optimisation of it cannot close the gap (consistent with the 1500-step null).
4. Box shape vs probe count (small_6 paired): `svd_jac` from 2 boxes 0.90, from 24 boxes 0.92; the isotropic closed form from
   the same 2 boxes 0.30 with 59 smaller radii. The Jacobian shaping carries the closed form; the probe count barely matters.

Round 6 (last GPU window before the maintenance): rotation-only ℓ1 refinement `l1N_rot` (closed-form metric kept, Cayley-
parametrised rotation optimised on the ℓ1 surrogate; unified variant), screened on all three models and Yelp seed 1; per-class
learners on big_3 (label-0-only / label-1-only boxes) to bound what a sign-aware manual rule could gain.

## What did not help (all three models unless noted)

- More probes (Yelp, 24 → 96): random-box share 0.58 → 0.65, learner-box share unchanged.
- Probe seed: a second random-token seed gives an unrelated layer-1/2 gauge on Yelp (identical layer 0); it scores 0.25 on the
  learner's boxes but 0.51 on the held-out screen with 0 smaller radii — the learner-box column exaggerated it. small_6 / big_3
  are seed-insensitive.
- Unlabeled dev-text probes (Yelp): 0.52 on the screen with 19 larger / 18 smaller (two-sided); random + text 0.68. Random
  tokens are the better probe distribution on this model.
- Sensitivity-weighted token blocks (query / key blocks scaled by the rank-1 factors of |∂margin/∂score|, value blocks by first-
  order softmax widths): held-in ±0.01 on all three models, cond 45–83. Dropped.
- CROWN-inflation rescaling of M (measured width / Jacobian width per token: 1.3× / 2.2× at layers 1 / 2 on big_3, up to 1.9×
  at layer 5 on small_6; fixed point in one round): held-in ±0.00 on all three models. Dropped — a near-uniform column scaling
  does not change the SVD balancing, and the layer-0 block is exact anyway.
- Longer ℓ1N optimisation (1500 steps, cosine): surrogate −1.5 %, held-out screen 0.84 → 0.86 on big_3. The surrogate's optimum is
  reached; the remaining gap is in the model, not the optimiser.
- Verifier-assisted variants (`svd_infl*`, `l1N_infl`) are labelled as such; the procedure of record uses none.
Other facts from the same tables: the SST-, Yelp-, random-token-, two-word- and verifier-trained small_6 gauges all sit at the
same surrogate value and the same held-in margin (one flat optimum); the candidate is an unrelated matrix to the learned gauge
(diag-ness of learned⁻¹·candidate ≈ 0.03, random-like) — the formula lands in the optimum, it does not reconstruct the learned
matrix; the cond-28 overfit gauge is ranked worst by every Jacobian-shaped surrogate; `svd_iso` is the ViT R45 construction
(right idea, wrong metric: it half-worked on pgd and hurt ibp).

## Pending

- Pre-registered bar for "beats the learned gauge": paired radius ratio-of-means above the learned gauge's with a smaller-radius
  count at or below it, on at least two models. Standing: paired shares 0.91 (big_3, unified rule), 0.84 (Yelp, `l1N`),
  1.00 (small_6, unified rule; mean 0.02487 vs 0.02486 — a tie at the bisection's 1e-4 resolution; ahead head to head 142 / 122
  and on eps-0.03 verified 98 vs 95; by label 0.79 / 1.30, the closed form's class trade evened out) — one-sided everywhere; the
  bar (mean above the learned gauge's on two models) is not met.
- Ceiling test done: the CROWN learner warm-started from the closed form (Yelp, 100 steps) ties the learned-from-identity gauge
  (+10.0 % vs +10.1 %, 94 / 94 / 89 head to head) — the learned gauge is the optimum of its objective.
- Round 6 screens and the per-class learners (above); the small_6 refined paired eval; the fp32-interval-weight run (cut at 8 h,
  resubmit after the maintenance).
- Verifier-free and probe-seed checks of step 1 (done): big_3 `svd_jacN_all_u` +0.384 / +0.947, second seed +0.382 / +0.941 vs
  +0.384 / +0.946; Yelp verifier-free +0.190 / +1.517 matches, seed dependence as described above.

## Files

`gauge_formula.py` (validate: surrogates, ablations, candidates, hybrids `--hybrids`, probe-seed swaps `--cross`, text probes
`--dev_probes`, ℓ1 refinement `--l1_steps --l1N --l1_lr`, sensitivity weights `--sens`, inflation `--infl`, held-out radius
screen `--radius_names auto --n_dev --dev_split`; saves `gauges/formula_<model>_<candidate><tag>.pt`), `gauge_formula_chain.sh
smoke|full|big3|yelp3|yelp3_big|hyb_*|r2_*|r3_*|r4_*`, `deept_formula_eval.sh <tag> <gauges> <model> <eps>` (paired eval),
`deept_init_chain.sh` (warm-start learner + paired eval), `validate_summary.py`, `paired_summary.py`,
`results/formula_<model>_{validate,hyb,r2,r3,r4}.json`, `results/deept_formula_<tag>_eval_short_seed0.json`.
