# Certified neuron-merging by row-proportionality snapping (the 7-step procedure)

**Procedure asked for:** (1) take an MLP, (2) verify with CROWN, (3) find near-proportional
weight-row pairs `row_j ≈ β·row_i`, (4) **snap** them to exact proportionality, (5) **merge**
the two now-proportional ReLU neurons in the next layer, (6) reverify with CROWN, (7) **certify**
the snap changed nothing significant in the relevant region.

Script: `snap_merge_pipeline.py` (+ probes `snap_merge_probe.py`, `snap_merge_probe2.py`), run
in the abcrown venv against real auto_LiRPA, `method="CROWN-Optimized"`, margin-lb spec.

## The three nets (all exact algebra) and the step-7 certificate

```
orig  --(lossy snap: A1[j]:=β·A1[i], b1[j]:=β·b1[i])-->  snapped  --(EXACT merge)-->  merged
```
For β>0, `relu(β·z)=β·relu(z)`, so once snapped, neuron j ≡ β·neuron i: drop j and fold
`β·A2[:,j]` into `A2[:,i]`. **float64 gate: snapped ≡ merged (≤1e-8).**

**Step-7 = a *composed certificate for the ORIGINAL net*.** The snap perturbs only pre-activation
`z_j`; over the ε-box `d_j = max|Δz_j| = |r·c+r_b| + Σ ρ_k|r_k|` (exact; `r` = residual row,
`c,ρ` = box center/radius after [0,1] clamp). ReLU is 1-Lipschitz, so the induced margin change is
bounded analytically by `δ_m ≤ d_j·(|C||A3||A2[:,j]|)_m`. Since `margin_orig(x) ≥ margin_snap(x) − δ`
pointwise, **`lb_merged − δ` is a sound lower bound on the original net's true margin.** The headline
metric is `(lb_merged − δ)` vs direct `CROWN-Opt(orig)` on the same images — verifying the original
network *better* by routing through the merged surrogate. This composed certificate is the novel
piece (compression/merging literature accepts *uncertified* error; here the error is certified and
folded back into a valid bound for the original model).

## Finding 1 (real nets, NEGATIVE): standard training produces NO snappable pairs

For every same-layer row pair (augmented `[w|b]`), least-squares β and relative residual
`‖row_j − β·row_i‖/‖row_j‖` (β any sign; near-zero-norm dead rows excluded). Minimum residual =
how close the *closest* pair is to proportional:

| training regime | H | min residual (layer A1 / A2) |
|---|---|---|
| vanilla (Adam, 3 ep) | 64 | 0.656 / 0.785 |
| vanilla | 128 | 0.692 / 0.797 |
| vanilla | 256 | 0.697 / 0.821 |
| dropout 0.5 (8 ep) | 64 | **0.551 / 0.450** |
| dropout 0.5 | 128 | 0.571 / 0.590 |
| weight-decay 1e-3 | 64 | 0.610 / 0.635 |
| dropout 0.3 + wd 3e-3 | 64 | **0.448 / 0.405** |
| long train, small data | 64 | 0.618 / 0.735 |

**No regime gets below ~0.40.** Width does not help (features are near-orthogonal in high dim);
dropout/weight-decay (which are known to encourage redundant units) help only marginally. Snapping
a 0.4-residual pair moves that neuron's pre-activation by ~40% of its scale — far too large to
certify as insignificant.

Running the full pipeline on the vanilla net at ε∈{0.02,0.03,0.05}: **0 snap-candidates**
(res<0.20) at every ε. The "best" pair by net score merely *prunes* a low-impact neuron (β≈0.009,
res≈1.0) — not a merge. **Crossover:** δ grows ~linearly in residual while the recoverable CROWN
slack requires proportionality to actually hold, so a pair pays only around **res ≲ 0.02** — an
order of magnitude below the ~0.4 real-training floor. *On standard MLPs the technique does not fire:
the required structure is absent.*

## Finding 2 (POSITIVE, machinery): where proportional pairs exist, the merge certifiably tightens full CROWN

To test the pipeline itself (constraint: not achievable on a stock net, so structure is **planted**
— a soft proportionality penalty on 8 disjoint row-pairs during training; labelled as a
*pipeline* existence proof, **not** a claim about real nets), 4 pairs converged to res=0.001,
β≈1.0. Full pipeline, CROWN-Optimized, 60 images (test acc 0.919):

| ε | pairs | merge tightens CROWN (mean / min) | δ (mean) | **cert − lb_orig** (mean) | images cert>lb_orig |
|---|---|---|---|---|---|
| 0.03 | 1 | +0.0024 / −0.0000 | 0.001 | +0.0017 | 36/60 |
| 0.03 | 4 | +0.0068 / +0.0001 | 0.007 | −0.0002 | 19/60 |
| 0.05 | 1 | +0.0057 / +0.0001 | 0.004 | +0.0013 | 27/60 |
| 0.05 | 4 | +0.0175 / +0.0006 | 0.008 | +0.0098 | 45/60 |
| 0.08 | 1 | +0.0104 / +0.0013 | 0.005 | +0.0056 | 39/60 |
| 0.08 | 4 | **+0.0363 / +0.0058** | 0.009 | **+0.0275** | **57/60** |

- **δ-soundness validated empirically.** On every row, 1000 random points per image sampled in the
  ε-box give `max|Δmargin_m| − δ_m ≤ 0` (worst = +0.000, OK) — the analytic δ genuinely upper-bounds
  the true margin change, so `lb_merged − δ` is a sound bound for the original net. (Guards against a
  sign/indexing bug that the exact float64 gate alone would not catch.)
- **The merge tightens CROWN on average at every ε** (step-5 isolation mean > 0 throughout). The
  per-image *min* is > 0 for ε ≥ 0.05; at ε=0.03 the single-pair min is −0.0000 — CROWN-Optimized
  α-iteration noise at the rounding floor, not a real loosening. Mechanism = the same door as the
  exact CReLU-collapse (independent relaxation of linearly-dependent unstable ReLUs collapses to one),
  reached here via an *approximate* snap.
- **δ is negligible** (~0.009 for 4 pairs; empirical function change ~0.002) because the snapped
  pairs are genuinely proportional (res=0.001).
- **The composed certificate beats direct CROWN on the ORIGINAL net** on a growing majority of images
  as ε rises (57/60 at ε=0.08), because the merge's CROWN tightening outgrows δ once enough neurons
  are unstable. Two caveats, both honest: (i) at small ε the gain is tiny and δ *adds linearly per
  pair*, so 4 pairs at ε=0.03 over-spends δ and goes net-negative (19/60) while a *single* pair still
  pays (36/60) — pair count must be tuned to ε; (ii) images where cert < lb_orig stay sound — deploy
  `max(lb_orig, cert)` and the certificate never regresses.

## Honest scope

- **The real-net evidence is Finding 1, and it is negative.** Finding 2's net has planted structure;
  it demonstrates only that the snap→merge→**certify** pipeline is *sound and can improve the
  original net's certificate* where proportional pairs exist — it is not evidence that real nets have
  them (they don't, per Finding 1).
- δ uses the sound but loose 1-Lipschitz `|W|` propagation. A tighter δ (CROWN on the difference, or
  per-neuron stability in the box) would widen the crossover, but not by the ~20× needed to reach the
  0.4 real-training floor.
- This is the *approximate / certified-surrogate* cousin of the exact CReLU-collapse
  (`CRELU_CROWN_RESULT.md`): same CROWN door (collapsing linearly-dependent unstable ReLUs), but here
  the proportionality is inexact and the resulting function change is **certified** and composed back
  into a valid bound for the unmodified original model.
