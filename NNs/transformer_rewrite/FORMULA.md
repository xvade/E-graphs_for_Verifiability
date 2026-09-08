# The formula for the attention gauge

Status 2026-09-07 19:20. Goal: derive the per-head gauge (G_q, G_a) that the learner (`deept_gauge.py learn`, Adam on CROWN's lower
bound over tuning boxes) finds, from the weights alone. What is settled: a closed-form construction that reaches 90 % of the
learned gain on the learner's own boxes for small_6 and 75–86 % on big_3 / Yelp small_3, from the weights plus a handful of random
probe sequences. What is pending: the paired 294-position evals (the headline numbers) and the verifier-free confirmation on the
two smaller models. Numbers in the tables below are the *held-in screen* (mean CROWN lower bound at each box's stock radius), not
the headline metric.

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

## Evidence so far (held-in screen; `gauge_formula.py validate`, 24 random-token boxes for M/N, 48 learner boxes)

Mean CROWN lower bound at the stock radius, random boxes / learner's own boxes. The learner's boxes are held-in for the learned
gauge and held-out for every candidate.

| model | identity | learned | svd_iso (M=I) | svd_jac | svd_jacN_all | share of learned gain (jacN_all) |
|---|---|---|---|---|---|---|
| SST small_6 | +0.092 / +0.156 | +0.468 / +1.184 | +0.272 / +0.716 | +0.426 / +1.087 | **+0.430 / +1.097** | 90 % / 92 % |
| SST big_3 | +0.111 / +0.146 | +0.473 / +1.186 | +0.262 / +0.609 | +0.322 / +0.805 | **+0.384 / +0.946** | 75 % / 77 % |
| Yelp small_3 | +0.076 / +0.246 | +0.275 / +1.785 | +0.065 / +1.157 | +0.144 / +1.373 | **+0.192 / +1.510** | 58 % / 82 % |

Other facts from the same tables: the SST-, Yelp-, random-token-, two-word- and verifier-trained small_6 gauges all sit at the
same surrogate value and the same held-in margin (one flat optimum); the candidate is an unrelated matrix to the learned gauge
(diag-ness of learned⁻¹·candidate ≈ 0.03, random-like) — the formula lands in the optimum, it does not reconstruct the learned
matrix; the cond-28 overfit gauge is ranked worst by every Jacobian-shaped surrogate; `svd_iso` is the ViT R45 construction
(right idea, wrong metric: it half-worked on pgd and hurt ibp).

## Pending

- Paired evals on the standard protocols (radius gain = ratio of means, eps verified counts, reverse flips, fp64 gate):
  small_6 294 positions — 2-box `svd_jac` (job 39773109), 24-box `svd_jacN_all` + `svd_jac` (39787014); Yelp small_3 277 instances
  (39778637); big_3 288 instances (39782433). Bar: ≥ 80 % of the learned radius gain with reverse flips at or near zero, on two models.
- Verifier-free confirmation of the N construction (`svd_jacN_all_u`: eps := 1 everywhere) and its probe-seed sensitivity
  (`svd_jacN_all_u2`) on big_3 and Yelp small_3 (jobs 39787038, 39787037). On Yelp small_3 the plain `svd_jac` collapsed under a
  second probe seed (+0.40 vs +1.37 on the learner boxes at identical surrogate value); whether N removes that is open.

## Files

`gauge_formula.py` (validate: surrogates, ablations, candidates, held-in screen; saves `gauges/formula_<model>_<candidate>.pt`),
`gauge_formula_chain.sh smoke|full|big3|yelp3`, `deept_formula_eval.sh <tag> <gauges> <model> <eps>` (paired eval),
`results/formula_<model>_validate*.json`.
