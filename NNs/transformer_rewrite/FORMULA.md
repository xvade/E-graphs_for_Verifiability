# The formula for the attention gauge

Status 2026-09-11 01:45 (round 7 CLOSED: the PER-CLASS manual rule — unified rule + sign-aware QK refinement on label-y probes, gauge chosen by the model's prediction — beats the single learned gauge on the paired protocol on both models, small_6 +18.0 % vs +13.1 % and big_3 +14.3 % vs +11.9 %, 246/0 and 208/0 head to head, and reaches the identity-initialised per-class learners; plain-CROWN tier; see "Round 7" below). Earlier status 2026-09-10 17:30, Earlier status 2026-09-09 08:45. Goal (second phase): a *manual* procedure — weights plus a handful of random probe sequences, no
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

5. **Per-class ceiling (big_3, 2026-09-09).** Learners trained on label-0-only / label-1-only boxes, held-out screen split by label:
   label 0 — single learned +9.5 %, label-0 learner +9.9 %, **manual unified rule +10.0 %** (at the ceiling); label 1 — single
   learned +15.1 %, **label-1 learner +18.5 %** (from 17 boxes), manual +11.1 %. Choosing the gauge by the label being verified
   gives +14.9 % vs +12.7 % for the single learned gauge (1.17×). The class trade is real; the sign-blind manual rule lands on
   the class the weights favour and is at 0.59 of the other class's ceiling — the target for a sign-aware surrogate.
   small_6 (24 + 24 boxes): label 0 — single +15.8 %, label-0 learner +17.2 %, manual +13.5 %; label 1 — single +12.8 %,
   label-1 learner +23.1 % (15 boxes), manual +16.5 %; per-class combined +20.6 % vs +14.1 % single (1.46×). Classes swapped
   relative to big_3; the sign-blind rule lands on label 1 here.
   Rotation-only ℓ1 refinement (round 6): big_3 0.87 vs unified 0.83, Yelp 0.83 vs 0.82 (full ℓ1 0.88) — same numbers with
   the closed form's conditioning; a cleaner rule, not a better one.

Round 6 (last GPU window before the maintenance): rotation-only ℓ1 refinement `l1N_rot` (closed-form metric kept, Cayley-
parametrised rotation optimised on the ℓ1 surrogate; unified variant), screened on all three models and Yelp seed 1; per-class
learners on big_3 (label-0-only / label-1-only boxes) to bound what a sign-aware manual rule could gain.

## Round 7 (2026-09-10): the sign-aware surrogate and the per-class manual rule

**What plain CROWN actually does with a product** (`auto_LiRPA/operators/bivariate.py`, `mul_middle` False = our setting): x·y is
replaced by the lower McCormick plane through the corner (x_l, y_l) when the backward coefficient on the product is positive
[error (x − x_l)(y − y_l)] and by the upper plane through (x_l, y_u) when it is negative [error (x − x_l)(y_u − y)]; x is input 0
(q in q·kᵀ, p in p·v). Both planes touch the box at x = x_l; which plane applies flips with the sign of the coefficient, i.e.
with the label; and the error is paid where the bound's minimiser sits, not uniformly. The ℓ1 width product of the earlier
rounds is the sign-blind envelope of this.

**The sign-aware cost** (`signed_probe_data`, `signed_cost`, `candidate_signed`; `gauge_formula.py --signed 1`): per probe
box, first-order intervals from the layer-input Jacobians, the margin's first-order worst corner z* = −sign ∂margin/∂δ, the
score gradients λ = ∂margin/∂(qkᵀ) (their sign picks the plane) and the dense-output gradients g; cost = Σ |coef| × plane
error at z*, with x − x_l = ε(‖r‖₁ + r·z*) and y_u − y = ε(‖r‖₁ − r·z*) for a functional with Jacobian row r. Positive
diagonal gauges leave it unchanged; sign flips do not (verified on CROWN: diag(±1) gauges tighten plain CROWN by ≈ +0.12 on
9 / 9 boxes and are α-neutral; positive diagonals neutral in both tiers).

**Ordering test (cost-only, label-y random-token probes, no verifier):** on small_6 the signed cost reproduces the held-out
screen's order on both labels — label-y learner best, single learned next, unified rule / closed form, wrong-label learner at
≈ 1.5 × identity — where ℓ1N has every gauge within 0.74–0.78. On big_3 it ranks the per-class learners right but over-rates
the unified rule on label 1 (its AV term, built on first-order softmax widths, dominates).

**Optimising it (400 Adam steps from the unified rule on label-y probes, held-out screen split by label, 24 + 24 boxes):**

| small_6 rule (label-y probes → gain on label y \| other label) | label 0 | label 1 |
|---|---|---|
| single learned gauge | +15.8 \| +12.8 | +12.8 \| +15.8 |
| unified rule (start point) | +13.7 \| +16.5 | +16.5 \| +13.7 |
| per-class learner (`--label`) | +17.2 \| +5.8 | +23.1 \| +5.4 |
| pure signed, both sides / QK only | −16.8 / +2.6 | −27.2 / +6.7 |
| cond penalty 1e-2, both / QK | −4.1 / +10.8 | −13.6 / +13.8 |
| rotation-only (Cayley), both / QK | +9.2 / **+17.2** \| +8.4 | +14.4 / **+20.9** \| +8.0 |
| ℓ1N mix (normalised, weight 1), both / QK | +5.9 / **+17.6** (8 / 0 vs single) \| +8.6 | +4.6 / **+22.0** (21 / 0 vs single) \| +7.4 |

The pure cost is exploitable: Adam drives it to 0.03 of identity (the QK part to ≈ 0 at deeper layers) by arranging every
product on its plane's zero-error edge at z*, and CROWN's actual minimiser moves elsewhere (−17 … −31 % radius). Constrained
to the QK side and either to rotations of the unified rule's gauge or to the ℓ1N envelope, it becomes a **per-class manual
rule** that reaches the per-class learners: +17.6 % on label 0 (learner +17.2 %) and +22.0 % on label 1 (learner +23.1 %),
with the same trade-away of the other class the learners show. The AV gauge must stay at the unified rule (every both-sided
variant loses). **big_3 confirms it:** label 0 +10.3 % (learner +9.9 %, single +9.5 %; the ceiling), label 1 **+19.6 %**
(learner +18.5 %, single +15.1 %, unified rule +11.2 %; 19 / 0 vs the single gauge), rotation-only +9.1 / +17.5 %.

| held-out screen, gain on label 0 \| label 1 | single learned | per-class learners | **per-class manual** | rotation-only |
|---|---|---|---|---|
| big_3 | +9.5 \| +15.1 | +9.9 \| +18.5 | **+10.3 \| +19.6** | +9.1 \| +17.5 |
| small_6 | +15.8 \| +12.8 | +17.2 \| +23.1 | **+17.6 \| +22.0** | +17.2 \| +20.9 |

Above the single learned gauge in every cell, at or above the per-class learners in three of four. Paired-protocol
confirmation (per-class manual vs per-class learners vs single learned, gauge chosen by the label being verified;
`perclass_paired.py`) in `PROGRESS.md` 2026-09-10 evening.

**Caveats.** The screen numbers are selected (eight variants per cell screened on the same boxes; the paired protocol with the
variant fixed in advance is the nearly unbiased read: 1 of 40 small_6 and 4 of 40 big_3 paired sentences were also in the
screen). Probes are the learner's dev boxes; the paired label is the model's own prediction (correctly classified sentences only). Ablation: the same label-y probes WITHOUT the signed term (round 6's
label-conditioned rule) move ≤ 1 point (big_3 label 1 +12.0 vs +11.1; small_6 label 0 +13.3 vs +13.5), so the signed term is
what carries the gain. The signed cost is a gradient direction, not a score: the collapsed pure gauge (0.033 of identity) and
the working mixed gauge (0.042) are indistinguishable by it and differ by 34 points of CROWN radius — never optimise it
unconstrained and never read its value as a predictor.

**Paired confirmation, small_6 (2026-09-10, 294 instances, `sgn_mix_qk` fixed in advance):** chosen by label +18.0 % vs stock
(281 / 0), vs the single learned gauge (+13.1 %) 246 / 0 head to head — bar MET; ε-0.03 verified 41 → 103 (single 95). Per-class
learners on the same protocol: +18.6 % (label 0 +18.0 = manual, label 1 +19.2 vs manual +18.0; head to head 31 / 128 / 135 equal).
The manual gauges never lose to stock on the other class (0 / 3 smaller) where the learners do (25 / 22), so a wrong label costs
nothing vs stock.
**Paired confirmation, big_3 (288 instances):** chosen by label +14.3 % vs stock (277 / 0), vs the single learned gauge (+11.9 %)
208 / 0 head to head — bar MET (label 0 +12.4 vs +11.0, label 1 +17.0 vs +13.2); ε-0.0287 verified 4 → 52 (single 48); other-class
0 / 0 smaller than stock. **The pre-registered bar (ratio of means above the single learned gauge's, smaller count ≤ it, two
models) is met.** Per-class learners on big_3: +13.3 % (276 / 0; vs single 167 / 15); manual vs learners head to head 113 / 19 / 156 equal, ahead
on both labels (+12.4 vs +11.2, +17.0 vs +16.5).

| paired, plain CROWN | single learned | per-class learners | per-class MANUAL | manual vs single | manual vs learners |
|---|---|---|---|---|---|
| small_6 (294) | +13.1 % (273/0) | +18.6 % (280/0) | **+18.0 % (281/0)** | 246 / 0 | 31 / 128 / 135 eq |
| big_3 (288) | +11.9 % (274/0) | +13.3 % (276/0) | **+14.3 % (277/0)** | 208 / 0 | 113 / 19 / 156 eq |

**Round-7 verdict:** the per-class manual rule beats the single learned gauge on both models (bar met, 0 smaller, strict
head-to-head sweeps) and reaches the identity-initialised per-class learners (ahead on big_3, tie / −1.2 pts on small_6; the
warm-started `initlab` learners, the stronger learned baseline, were not paired-evaluated); it never falls below stock under a
wrong label where the learners do (small_6 25 / 22). No verifier at construction time, but the design choices (variant, mix
weight, QK-only) were selected with CROWN radius screens. Plain-CROWN tier only — CROWN-Optimized absorbs per-class gains.

**Procedure of record for a per-class gauge (label y):** unified rule → `candidate_signed` on label-y random-token probes, QK
side only, ℓ1N mix weight 1 (or rotation-only), 400 Adam steps lr 0.02 → `gauges/formula_<name>_sgn_mix_qk_mix_lab<y>.pt`.
Note that the α tier absorbs the per-class refinement (CROWN-Optimized: label-1 learner +2.40 vs single +2.38 on label 1);
per-class gauges are a plain-CROWN-tier gain.

### Round-7 paired verdict (2026-09-11)

Both per-class constructions were run on the full paired protocol (small_6: 294 instances, ε 0.01 / 0.02 / 0.03; big_3: 288
instances, ε 0.00957 / 0.0191 / 0.0287), each label's gauge on every instance and then chosen by the instance's label
(`perclass_paired.py`; at verification time the choice uses the model's prediction, which equals the label on the correctly
classified instances the protocol uses, and each gauge is an exact rewrite so the choice costs nothing in soundness).

| model | single learned | per-class learners | **manual per-class rule** (`sgn_mix_qk`) |
|---|---|---|---|
| small_6 | +13.1 % | +18.6 % (vs single: 232 larger / 0 smaller) | **+18.0 %** (246 / 0); per label +18.0 / +18.0 |
| big_3 | +11.9 % | +13.3 % (167 / 15) | **+14.3 %** (208 / 0); per label +12.4 / +17.0 |

Verified counts at the largest ε: small_6 stock 41 → single 95 → learners 105 → manual 103; big_3 stock 4 → single 48 →
learners 51 → manual 52. Either manual per-class gauge used on *all* instances is within a point of the single learned gauge
(+12.3 % / +12.2 % on small_6, +11.0 % / +11.1 % on big_3), so the rule matches the learned gauge as a single gauge and beats it
by choosing per label; used on the wrong class it degrades less than the learners do (+7.1 / +5.8 % vs +3.8 / +3.8 % on
small_6). The screen overstated the small_6 label-1 cell (+22.0 % → +18.0 % paired) by variant selection; the other cells
held. Test-set reuse: the 24 screen sentences that chose the variant overlap the paired protocol on 1 sentence (3 of 294
instances) for small_6 and 4 sentences (17 of 288) for big_3. Conclusion: given the predicted label, the manual procedure beats the single learned gauge on both models with no instance
smaller; against the like-for-like comparator (per-class learners) it is 0.97× on small_6 and 1.07× on big_3, where the
label-1 learner had only 17 tuning boxes. As a single gauge (no label) it reaches 0.94 / 0.93 of the single learned gauge, so
the pre-registered one-gauge bar is still not met — the gain over the learned gauge comes from per-label selection, which the
procedure makes cheap (no verifier calls per class) and which the learners can also use. α-CROWN tier (CROWN-Optimized, 20 it; one manual gauge per
run, chosen by label offline, vs the once-trained single gauge): big_3 ≤ 5 tokens (29 instances) tighter on 29 / 29 at ε 0.015
and 0.02 with verified 14 / 3 unchanged; small_6 ≤ 6 tokens (49 instances, ε 0.02) verified 27 → 28, tighter on 37 / looser 12
(mean lb +0.412 vs +0.400) — the per-class lead is mostly absorbed by the α optimisation, as it was for the learners, but the
rule stays at or above the single gauge. Not measured for the manual rule: the per-class learners at the α tier, and the
weak-rule models (Yelp, smaller_3).

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

- Pre-registered bar for "beats the learned gauge" (paired radius ratio-of-means above the learned gauge's with a
  smaller-radius count at or below it, on at least two models) — two readings, both stated: **as pre-registered, one gauge
  against one gauge, NOT met**: either manual per-class gauge used on all instances reaches 0.94 / 0.93 of the single learned
  gauge's gain (+12.3 % vs +13.1 % small_6, +11.0 % vs +11.9 % big_3), and the unified rule ties only on small_6 (1.00; big_3
  0.91, Yelp 0.84). **Under per-label selection (the gauge chosen by the model's predicted label), MET against the single
  learned gauge** on both models (small_6 +18.0 % vs +13.1 %, 246 larger / 0 smaller; big_3 +14.3 % vs +11.9 %, 208 / 0) — but
  the single gauge does not get to use the label; the like-for-like learned comparator is the per-class learners, against
  which the rule is 0.97× on small_6 and 1.07× on big_3 (whose label-1 learner was tuned on 17 boxes).
- Ceiling test done: the CROWN learner warm-started from the closed form (Yelp, 100 steps) ties the learned-from-identity gauge
  (+10.0 % vs +10.1 %, 94 / 94 / 89 head to head) — the learned gauge is the optimum of its objective.
- Round 6 screens and the per-class learners: done (round-7 paired verdict above). Rigorous certificate transfer: done with the
  gauge UNFOLDED (`deept_unfolded.py`, G⁻¹/A⁻¹ as verified 2-ulp intervals): the interval network certifies the plain gauged
  radius on 294/294 small_6 instances, confirmed per instance (PROGRESS.md 2026-09-10 20:00–20:36); the four-sided folded
  interval tier is superseded.
- α-CROWN tier for the manual per-class rule: done 2026-09-11 (see the round-7 paired verdict): at or above the once-trained
  single gauge on both models (big_3 29 / 29 tighter, small_6 27 → 28 verified, 37 / 12).
- Verifier-free and probe-seed checks of step 1 (done): big_3 `svd_jacN_all_u` +0.384 / +0.947, second seed +0.382 / +0.941 vs
  +0.384 / +0.946; Yelp verifier-free +0.190 / +1.517 matches, seed dependence as described above.

## Files

`gauge_formula.py` (validate: surrogates, ablations, candidates, hybrids `--hybrids`, probe-seed swaps `--cross`, text probes
`--dev_probes`, ℓ1 refinement `--l1_steps --l1N --l1_lr`, sensitivity weights `--sens`, inflation `--infl`, held-out radius
screen `--radius_names auto --n_dev --dev_split`; saves `gauges/formula_<model>_<candidate><tag>.pt`), `gauge_formula_chain.sh
smoke|full|big3|yelp3|yelp3_big|hyb_*|r2_*|r3_*|r4_*`, `deept_formula_eval.sh <tag> <gauges> <model> <eps>` (paired eval),
`deept_init_chain.sh` (warm-start learner + paired eval), `validate_summary.py`, `paired_summary.py`,
`results/formula_<model>_{validate,hyb,r2,r3,r4}.json`, `results/deept_formula_<tag>_eval_short_seed0.json`.
