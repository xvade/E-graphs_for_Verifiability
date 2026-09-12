# Objective-vs-optimiser diagnostic for the manual gauge procedure (2026-09-08, fork report)

**Question.** At the learned gauge, is the ℓ1 width-product surrogate with N (the objective the manual step 2 minimises)
LOWER than at the manual optimum (→ the optimiser is stuck) or HIGHER (→ the surrogate is the wrong objective)?

**Answer: HIGHER on all three models, at every layer, on both sides.** The manual gauges already sit *below* the learned gauge
on the surrogate everywhere, including the places where the learned gauge wins on CROWN (big_3 layer 0; small_6 QK side; Yelp
diffuse). The residual CROWN gap is therefore an objective mismatch, not an optimisation failure: pushing the surrogate further
down cannot recover it, and the surrogate does not know what the learned gauge knows at those layers.

## 1. Exact numbers already in the r5 runs (M with eps = each box's stock CROWN radius, same probes the manual gauges were built from)

`surr.l1_jacN` from results/formula_{big3,yelp3,small6}_r5.json, all values / identity. "learned" = the CROWN learner's gauge
(deept_*_seed0.pt). Per-layer QK = `per_layer_l1_jac` (N does not enter the QK side, so this is the exact per-layer ℓ1N QK cost).

| model | gauge | ℓ1N total | QK | AV·N | per-layer QK ℓ1N | ℓ2 jac | max cond |
|---|---|---|---|---|---|---|---|
| big_3 | learned | 0.679 | 0.647 | 0.800 | 0.865 · 0.744 · 0.612 | 0.749 | 11.7 |
| big_3 | closed form svd_jacN_all | 0.606 | 0.560 | 0.779 | 0.970 · 0.716 · 0.501 | 0.665 | 10.6 |
| big_3 | l1N (both sides) | 0.576 | 0.530 | 0.748 | 0.795 · 0.665 · 0.485 | 0.677 | 10.8 |
| big_3 | unified l1N_qk+av0 (procedure of record) | 0.579 | 0.530 | 0.763 | 0.795 · 0.665 · 0.485 | 0.672 | 10.8 |
| Yelp small_3 | learned | 0.971 | 0.972 | 0.959 | 0.910 · 0.954 · 0.990 | 1.055 | 6.1 |
| Yelp small_3 | closed form | 0.959 | 0.959 | 0.942 | 1.005 · 0.950 · 0.967 | 0.891 | 6.5 |
| Yelp small_3 | l1N | 0.947 | 0.949 | 0.908 | 0.852 · 0.944 · 0.956 | 0.967 | 10.9 |
| Yelp small_3 | unified | 0.948 | 0.949 | 0.930 | 0.852 · 0.944 · 0.956 | 0.919 | 10.9 |
| small_6 | learned | 0.744 | 0.739 | 0.816 | 0.887 · 0.790 · 0.960 · 0.813 · 0.455 · 0.791 | 0.664 | 4.9 |
| small_6 | closed form | 0.736 | 0.732 | 0.791 | 0.979 · 0.750 · 0.956 · 0.805 · 0.436 · 0.788 | 0.601 | 34.6 |
| small_6 | l1N | 0.731 | 0.727 | 0.787 | 0.781 · 0.733 · 0.950 · 0.801 · 0.432 · 0.783 | 0.616 | 30.5 |
| small_6 | unified | 0.731 | 0.727 | 0.788 | 0.781 · 0.733 · 0.950 · 0.801 · 0.432 · 0.783 | 0.607 | 30.5 |

Reading per model:
- **big_3** (CROWN gap = layer 0). Layer-0 QK surrogate: closed form 0.970 → l1N 0.795, learned 0.865. The ℓ1 step took the
  layer-0 cost from *above* the learned gauge to *below* it and CROWN moved most of the way (screen share 0.56 → 0.84, paired 0.61 → 0.91) —
  but the learned gauge is still better on CROWN while being 9 % worse on the surrogate. Layers 1–2: manual below learned on QK
  (0.665/0.485 vs 0.744/0.612) and on AV·N. Total: learned 0.679 vs manual 0.576–0.579.
- **Yelp** (diffuse gap). Manual below learned at every layer and side; the whole model's surrogate only moves ~5 % under any gauge
  (attention is a small share of the width here), so the surrogate has little resolution and the CROWN gain is decided by things it
  does not model.
- **small_6** (CROWN gap = QK side across layers, screens at parity). Manual below learned on QK at every one of the six layers and
  on AV·N; the ℓ2 closed form is the worst on ℓ1 at layer 0 (0.979) yet ties learned on the paired protocol (share 0.92).

So the surrogate ordering is: manual < learned < identity on every model. It orders "gauged vs identity" correctly (that is why the
closed form works at all) but does NOT order "manual vs learned" the way CROWN does. The ℓ1 refinement lowered the surrogate by
5 % (big_3), 1.2 % (Yelp), 0.6 % (small_6) relative to the closed form; the learned gauge is 18 %, 2.5 %, 1.8 % ABOVE the manual optimum.

## 2. Cost-only run (job 39832397/39832715, A100, 8 min total): per layer AND per side, with a warm start from the learned gauge

Mode `--cost_only 1` (new; no CROWN anywhere). Probe boxes = the same random-token draw as r5 (same seed/args), but every box gets the
r5 run's MEAN stock radius as eps (`--probe_eps` 0.0088 / 0.0127 / 0.0106) instead of its own CROWN radius, so the token blocks of M
are weighted equally instead of by radius. The gauge ordering agrees with section 1 to three decimals, so the approximation is immaterial.
Values / identity; "AV·N" = value side read through the downstream functionals N_all. `l1N_from_<learned>` = `candidate_l1`
warm-started at the learned gauge (400 steps, lr 0.02, same cond penalty). Logs: cost_{big3,yelp3,yelp3s1,small6}.log in this directory;
JSON: results/formula_{big3,yelp3,small6}_cost.json, results/formula_yelp3_cost_s1.json.

| model | gauge | ℓ1N total | QK | AV·N | ℓ2N total | cond | per-layer ℓ1N QK | per-layer ℓ1N AV·N |
|---|---|---|---|---|---|---|---|---|
| big_3 | learned | 0.674 | 0.642 | 0.799 | 0.913 | 11.7 | 0.865 0.743 0.608 | 0.758 0.803 0.805 |
| big_3 | closed form (r5 file) | 0.604 | 0.560 | 0.777 | 0.874 | 10.6 | 0.970 0.711 0.505 | 0.819 0.781 0.761 |
| big_3 | l1N (r5 file) | 0.577 | 0.533 | 0.748 | 0.886 | 10.8 | 0.795 0.665 0.490 | 0.685 0.766 0.747 |
| big_3 | unified (r5 file) | 0.580 | 0.533 | 0.761 | 0.880 | 10.8 | 0.795 0.665 0.490 | 0.685 0.781 0.761 |
| big_3 | l1N warm-started AT learned | 0.578 | 0.535 | 0.747 | 0.889 | 9.3 | 0.790 0.667 0.492 | 0.679 0.766 0.746 |
| Yelp | learned | 0.972 | 0.973 | 0.963 | 0.973 | 6.1 | 0.910 0.956 0.990 | 0.850 0.942 1.000 |
| Yelp | closed form (r5) | 0.961 | 0.962 | 0.944 | 0.956 | 6.5 | 1.005 0.952 0.970 | 0.821 0.929 0.980 |
| Yelp | l1N (r5) | 0.951 | 0.953 | 0.914 | 0.963 | 10.9 | 0.852 0.947 0.961 | 0.741 0.872 0.976 |
| Yelp | l1N warm-started AT learned | 0.951 | 0.953 | 0.914 | 0.963 | 10.8 | 0.852 0.947 0.960 | 0.741 0.872 0.975 |
| small_6 | learned | 0.745 | 0.740 | 0.816 | 0.851 | 4.9 | 0.887 0.789 0.959 0.812 0.457 0.792 | 0.838 0.977 0.870 0.756 0.728 0.809 |
| small_6 | closed form (r5) | 0.737 | 0.733 | 0.793 | 0.820 | 34.6 | 0.979 0.750 0.955 0.804 0.437 0.788 | 0.881 0.966 0.852 0.727 0.691 0.778 |
| small_6 | l1N (r5) | 0.732 | 0.728 | 0.788 | 0.845 | 30.5 | 0.781 0.731 0.948 0.799 0.434 0.784 | 0.796 0.958 0.858 0.733 0.689 0.757 |
| small_6 | unified r4 (paired-eval file) | 0.731 | 0.727 | 0.790 | 0.829 | 28.7 | 0.773 0.729 0.947 0.798 0.432 0.784 | 0.787 0.966 0.852 0.727 0.691 0.778 |
| small_6 | l1N warm-started AT learned | 0.732 | 0.728 | 0.789 | 0.846 | 26.9 | 0.779 0.731 0.948 0.799 0.433 0.784 | 0.788 0.956 0.859 0.734 0.691 0.756 |

Per (layer, side) the learned gauge is above the manual gauge in **every cell** of every model, with two ties (small_6 layer 2/4 QK
within 0.01; Yelp layer 2 AV within 0.005) — including the cells where CROWN says the learned gauge is better (big_3 layer 0, QK 0.865
vs 0.795 and AV·N 0.758 vs 0.685; small_6 QK at all six layers).

**Warm start from the learned gauge.** The optimiser walks the learned gauge down to the manual optimum's surrogate value
(big_3 3.17e4 → 2.73e4 vs 2.72e4 from the closed form; Yelp 9.67e5 → 9.48e5 vs 9.46e5; small_6 2.38e5 → 2.34e5 vs 2.34e5) and in
doing so moves it by a drift ‖G_learned⁻¹ G′ − I‖_F/√d_h of 0.85–1.23 per layer (QK: big_3 1.03/1.15/0.93, Yelp 0.87/1.11/0.89,
small_6 1.13/1.05/1.07/1.04/0.91/0.85; AV similar) — for scale, two unrelated rotations score ≈1.41 on this metric and the learned
gauge vs the closed form scores 1.4. So the surrogate does not merely fail to prefer the learned gauge: descending it from the learned
gauge actively leaves it. The learned gauge is not a stationary point of the surrogate. Saved as
gauges/formula_<model>_l1N_from_<learned>_cost.pt (big3 / yelp3 / small6) — screenable, but the diagnostic says they will land near
the plain l1N gauge on CROWN, not near the learned one.

## 3. Yelp cross-draw table (the coordinator's addition): can the surrogate rank the two probe draws?

Seed-0 probe draw gave a closed form with screen share 0.66 (l1N 0.88); seed-1's gave 0.28 (l1N 0.59) [note: the two r5 screens
also use different held-out sentence sets — `rngd = Random(seed + 7)` — so those shares are not on identical sentences]. ℓ1N total / identity:

| gauge | under seed-0 probes | under seed-1 probes |
|---|---|---|
| closed form from seed 0 | 0.9612 | 0.9601 |
| closed form from seed 1 | 0.9611 | 0.9598 |
| l1N from seed 0 | 0.9514 | 0.9505 |
| l1N from seed 1 | 0.9518 | 0.9505 |
| unified from seed 0 | 0.9521 | 0.9513 |
| unified from seed 1 | 0.9525 | 0.9513 |
| learned | 0.9723 | 0.9726 |

**No.** The two draws' gauges are identical on the surrogate to 3–4 decimals under either probe set (differences ≤ 0.0004, and the sign
flips between draws), while their CROWN shares differ by a factor 2. Per layer they are equal to 3 decimals at every layer and side.
Yet the gauges themselves are far apart: at layers 1–2 the drift between the seed-0 and seed-1 closed forms is 1.31/1.36 (QK) and
1.2–1.3 (AV) — as far apart as unrelated rotations — and 0.00 at layer 0 (the layer-0 box shape is data-free: the no_var LayerNorm
Jacobian is the same matrix for every probe, so only eps and the token count change). There is no verifier-free selection rule in this
surrogate: it is flat along the direction the probe draw moves the gauge.

## 4. Answers in one sentence each

- **big_3:** the learned gauge is 17 % ABOVE the manual optimum on the surrogate (0.674 vs 0.577), above at every layer and side including
  layer 0 where it wins on CROWN — objective mismatch; the ℓ1 step lowered the cost 4.5 % below the closed form.
- **Yelp small_3:** learned 2.2 % above manual (0.972 vs 0.951), above everywhere; the whole surrogate only spans 5 % under any gauge,
  and it cannot tell the good probe draw from the bad one — objective mismatch, and at layers ≥1 the surrogate is flat where CROWN is not.
- **small_6:** learned 1.8 % above manual (0.745 vs 0.732), above at every one of 6 layers on QK — objective mismatch; the ℓ1 step moved
  the cost only 0.7 % below the closed form.

So in every case the fix is a different objective (or a constraint on what the optimiser may change), not more optimisation.

## 5. Recommendation for the ≤2.5 h GPU round

The surrogate's minimiser is not where CROWN's is, and the direction the ℓ1 step moves the closed form (drift 0.7–1.5, mostly a
stretch — cond 30 on small_6) is not the direction the learned gauge sits in (the other fork: learned ≈ closed form ∘ near-orthogonal
rotation ∘ mild stretch). The candidate to test is therefore a **rotation-only refinement of the closed form**: G = G_closed · Q with
Q ∈ O(d_h) (Cayley-parametrised, start at Q = I), Adam on the same ℓ1-with-N surrogate, no cond penalty (conditioning is the closed
form's). It keeps the closed form's conditioning and its singular values, moves only inside the ℓ2 problem's flat O(d_h) valley (which is
exactly what the probe seed randomises on Yelp), and lets the ℓ1 surrogate choose the point in that valley instead of a numerical accident.
Screen `cand:l1N_rot` and `cand:l1N_rot_qk+av0` against the unified rule on all three models (r6). If it does not close the gap, the
next objective must come from CROWN itself (the intermediate-bound widths at layers ≥1), which the surrogate does not see.

## 6. Code changes (gauge_formula.py, +~55 lines, backward compatible)

- `gauge_drift(G0, G1, dh)`: per layer, mean over heads of ‖G0⁻¹G1 − I‖_F/√d_h.
- `cost_report(...)`: the `--cost_only 1` path (builds boxes, M_l with `--probe_eps`, N_all, closed form, l1N from the closed form and
  optionally from `--l1_init <gauge.pt>`; prints the per-layer/per-side ℓ1N and ℓ2N tables, cond, pairwise drifts; writes `--out` JSON and
  `gauges/formula_<model>_l1N*<tag>.pt`). Called from main() right before the lirpas are built (`if a.cost_only: return cost_report(...)`).
- `--l1_init <gauge.pt>` also works in the normal pipeline: adds `cand:l1N_init` (l1N warm-started from that gauge) next to the unified rule.
- The chain modes are untouched; the r5 command lines run exactly as before.
