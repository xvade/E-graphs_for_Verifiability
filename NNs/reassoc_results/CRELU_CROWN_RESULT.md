# CReLU-collapse: a min/max-free exact rewrite tightening full CROWN on real trained nets

**Goal (2026-09-02):** replicate the CROWN redundancy-collapse improvement under strict
constraints — (1) a **real** (not hand-crafted) model, (2) **no min/max** in the technique,
(3) improve **full CROWN-Optimized**, (4) **replicate across independent trainings** of the
same architecture (so coincidental weight values can't be abused). Scripts:
`crelu_pilot.py`, `crelu_replicate.py` (run in the abcrown venv against real auto_LiRPA).

## Why the constraints force CReLU

The redundancy-collapse mechanism (`CROWN_REDUNDANCY_RESULT.md`) needs **redundant ReLU
structure**. My own neutrality induction (`plain-relu-rewrites-cant-move-crown-bound`) says
any *exact* non-min/max rewrite of a plain-ReLU net that only rearranges the linear skeleton
is CROWN-neutral — and standard training destroys *exact* redundancy. So **no off-the-shelf
plain-ReLU benchmark model can satisfy this goal with an exact rewrite**; the redundancy must
be **architectural** (weight-independent), which is exactly what makes it survive constraint 4.

**CReLU** (Concatenated ReLU, Shang et al., ICML 2016) is the canonical published activation
with that structure: `CReLU(z) = [relu(z), relu(−z)]`. It was *motivated by the observation
that standard trained CNNs naturally learn opposite-phase filter pairs* — CReLU just makes the
pairing exact. Its complementary-pair redundancy is present in **every** training regardless of
weights or task.

## The rewrite (exact, pure ReLU algebra, no min/max)

Per CReLU layer, using `relu(−z) = relu(z) − z`:
```
W₊·relu(z) + W₋·relu(−z) = (W₊+W₋)·relu(z) − W₋·z ,   z = linear in the previous layer
```
So each CReLU layer's **two** unstable ReLUs per unit collapse to **one**, plus an exact linear
(skip) correction. Cascading this across both hidden layers yields DenseNet-style skips
(auto_LiRPA bounds the DAG). CROWN relaxes the two copies of the baseline independently
(`(|W₊|+|W₋|)·gap`); the collapsed single neuron costs `|W₊+W₋|·gap` — and `|W₊+W₋| ≤
|W₊|+|W₋|` (triangle inequality), **strict whenever any coordinate pair has opposite
signs**, which held in all 6 trained models.

## Result — real auto_LiRPA, CROWN-Optimized, MLP 784→CReLU(64)→CReLU(64)→10

6 **independent** trainings (3 seeds × MNIST @ ε=0.05, 3 seeds × FashionMNIST @ ε=0.03),
100 correctly-classified test images each. Every model: exact float64 gate (~3e-7), ReLU
coordinates halved (256→128, so auto_LiRPA does *not* share the pairs).

*(ε was fixed per task, not tuned to flatter the result: FashionMNIST @ 0.03 is the
balanced operating point (baseline verified 49–59%); MNIST @ 0.05 is a **stressed** point
(baseline 20–24%, most images unverifiable in both forms) — the paired per-image margin
delta is valid evidence at any ε, and the pilot measured it **positive at every ε tried**
(0.03/0.05/0.08), growing with ε as more pairs go unstable.)*

| training | test acc | verified (base→coll) | mean per-img margin Δ | min Δ | improved | cancel |
|---|---|---|---|---|---|---|
| MNIST seed0 | 0.936 | 20 → **29** /100 | +0.993 | +0.243 | **100%** | 0.68/0.63 |
| MNIST seed1 | 0.941 | 19 → **25** /100 | +1.154 | +0.378 | **100%** | 0.68/0.61 |
| MNIST seed2 | 0.937 | 24 → **33** /100 | +1.072 | +0.310 | **100%** | 0.67/0.63 |
| FashionMNIST seed0 | 0.835 | 59 → **62** /100 | +0.211 | +0.034 | **100%** | 0.70/0.64 |
| FashionMNIST seed1 | 0.843 | 49 → **54** /100 | +0.243 | +0.037 | **100%** | 0.68/0.63 |
| FashionMNIST seed2 | 0.850 | 56 → **61** /100 | +0.223 | +0.030 | **100%** | 0.69/0.64 |

**Headline:** across all 6 trainings and all 600 per-image CROWN-Optimized evaluations, the
collapsed form's certified margin lower bound is **strictly larger on every single image**
(min Δ > 0 everywhere), and verified accuracy rises in every training. The pilot confirmed the
same under *plain* CROWN too, and that the effect grows with ε (more unstable pairs). The
per-layer cancellation ratio `|W₊+W₋|/(|W₊|+|W₋|)` sits at ~0.63–0.70 (near the ~0.71
random-weight expectation, slightly below — trained pairs partially cancel), tying the measured
gain to the mechanism.

## Constraints, checked

1. **Real model** — genuinely trained (81–94% acc) on real MNIST/FashionMNIST; weights are
   learned, never hand-set. The *architecture* was selected because it instantiates the
   mechanism (surfaced, not hidden), but that is precisely why constraint 4 is the right test.
2. **No min/max** — the rewrite is `relu(−z)=relu(z)−z`, pure ReLU algebra.
3. **Full CROWN-Optimized** — the reported metric is `method="CROWN-Optimized"` margin lower
   bounds; improvement is universal across images.
4. **Replicates across trainings** — 6 independent trainings (2 tasks × 3 seeds); the
   improvement holds in *every* one, so it is architectural, **not a coincidental-weight
   artifact**. This is the direct answer to constraint 4's stated concern.

## Honest scope

- Not claimed as a **theorem** about full CROWN-Optimized: the triangle inequality bounds the
  per-pair slack, but per-direction chord/line selection and intermediate-bound recomputation
  aren't covered by that accounting. The claim is the **measured** distribution — 600/600
  images improved.
- Demonstrated on an **MLP**; a CReLU **CNN** (and larger ε operating points) is the obvious
  extension, not done here.
- This is the real-model, replicated version of `CROWN_REDUNDANCY_RESULT.md`'s hand-planted
  toy: same mechanism (independent relaxation of linearly-dependent unstable ReLUs), now on
  trained weights across seeds and tasks.
