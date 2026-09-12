# Provenance of the headline claims: which hardware produced which number

Written 2026-09-12 before the move from Hyak (UW) to a cluster with H200s. Every number in `PAPER_DRAFT.md`,
`FORMULA.md`, `README.md`, `RELATED_WORK.md` and the memory files was produced on one of the Hyak cards below.
The results JSONs written before 2026-09-12 do **not** record the device; this file is the record. From
2026-09-12 on, `deept_gauge.py` and `deept_unfolded.py` stamp `meta.device` / `meta.slurm_job` / `meta.host`
into every JSON they save, so later results carry their own provenance.

## Hardware on Hyak

| label | card / node | Slurm | capacity | notes |
|---|---|---|---|---|
| **L40S** | NVIDIA L40S, 48 GB (44.4 GiB usable) | `-p gpu-l40s` / `gpu-l40s-amath`, `-A gpu-l40s-amath` | 2 concurrent per group, non-preemptible, 12–16 h walltimes | the default card; docs say "44 GB" when they mean this card |
| **A100** | NVIDIA A100 80 GB (SXM) | `-p ckpt-all -A ckpt-amath --qos=ckpt-gpu --gres=gpu:a100:1` | 80 GB, preemptible (every 10–15 min by day) | every α run of `small_6` needed it; all jobs resumable per instance |
| **CPU** | ckpt CPU nodes | `-p ckpt-all --qos=ckpt -c 8 --mem=32G` | — | signed per-class optimisations (`sgnopt*.sbatch`), archive build |
| login | klone login node | — | ≈ 9.6 GiB cgroup | only tiny probes; never a bound computation |

Software for every number: torch 2.11 + cu130, auto_LiRPA at alpha-beta-CROWN `e5c7e17` / auto_LiRPA `5a098e8` with the
one-file softmax patch (`NNs/verifier_patches/`), fp32 throughout, `torch.get_float32_matmul_precision() == "highest"`,
no TF32 override. Gauge exactness gates are fp64.

## Sensitivity classes (cite one per claim)

- **A — time-capped verdicts.** The ViT official pipeline (`vit.yaml`: complex softmax, α-CROWN 50 it, β-CROWN BaB,
  100 s per instance) and the GenBaB stock runs (300 s). The verified count depends on card speed because BaB runs until the
  cap. A faster card verifies more in both arms. **On a new card both arms must be rerun on that card, alone on the GPU,
  and the paper must name the card.** Differences of ±1 between repeat runs on the same card were observed (BaB noise).
- **B — deterministic bounds.** Plain CROWN (bisection radii on a 1e-4 grid, fixed-ε lower bounds) and α-CROWN with a fixed
  iteration count (20 it in `eval_alpha`). Card-independent up to fp32 reduction order: lower-bound deltas below ≈1e-4
  are noise, radii may move by one grid step on isolated instances, verified counts are robust. A rerun on another card is a
  sanity check, not new evidence.
- **C — memory-chosen instance sets.** The α-tier sets were dictated by the card: `small_6` α needs 36 / 62 / ≈84 GiB at
  5 / 6 / 7 tokens, so the α tier is 49 instances ≤ 6 tokens on the A100; `big_3` α at 6 tokens needs 77.6 GiB, so 29
  instances ≤ 5 tokens; the `small_12` learner OOMs at ≥ 6 tokens on the L40S **and** at 6 tokens on the A100 (78.6 GiB), so
  its only gauge is a 5-box ≤ 5-token one (indicative only). An H200 (141 GB) relaxes all three and thereby **changes the
  population, not just the count** — a longer-token rerun is a new experiment, reported next to the old one, not a replacement.
- **D — timings and ratios.** Same-card only: unfolded-interval 2.3× per CROWN call and 1.65× total (A100), PBVerification
  Baseline ≈ 3 min per `small_6` instance (L40S), learner wall times (3 h 26 `small_6` alternating learner on A100; 5 h 53
  for the `small_6` per-class paired eval on a slow L40S node). Re-measure on the new card before quoting.
- **E — gauge artifacts.** The `.pt` gauges are the evidence; retraining with the same seed on another card will not
  bit-reproduce (fp32 gradients through CROWN, nondeterministic reductions). Evaluate the **archived** gauges (all 786 are
  in `repo_untracked.tar.zst`; the 37 referenced by committed results are in git). A retrained gauge is a separate row.

## The claims

Columns: claim → results file(s) under `NNs/transformer_rewrite/results/` unless another path is given → gauge → Slurm job
(name) → hardware → date → class → what an H200 rerun means.

### ViT tier (`NNs/vit_rewrite/`, VNN-COMP'23 `vit` benchmark, ε = 1/255)

| claim | results | gauge | job | hardware | date | class | H200 |
|---|---|---|---|---|---|---|---|
| `pgd_2_3_16` vanilla CROWN (lse) 24 → 36/100, tighter 100/100, width −23 % | `vit_rewrite/results/pgd_2_3_16__base__lse.json` vs `…__base__lse__G_pgd_mix_svdinit.json` | `vit_rewrite/gauges/pgd_mix_svdinit.pt` (CIFAR-train boxes) | interactive `salloc` on gpu-l40s (jobs 39459033 / 39516544 windows) | L40S | 2026-09-04 | B | sanity smoke only |
| id-init replication 24 → 37/100 | `pgd_2_3_16__base__lse__G_pgd_mix_idinit.json` | `gauges/pgd_mix_idinit.pt` | interactive salloc | L40S | 2026-09-04 | B (E for the gauge) | reuse gauge |
| **Full official pipeline 58 → 64/65/65 /100** (α-CROWN 41 → 52, initial CROWN 0 → 13 tighter 100/100, 0 reverse) | `vit_rewrite/results/official_stock_pgd.json`, `official_learnedG_patched.json`, `official_learnedG_pgd.json`, `official_idinitG_patched.json` | learned G patched into the stock ONNX (`vit_patch_onnx.py`) | `run_chain*.sh` / `run_official_one.sh`, each run **alone on the GPU** (interactive salloc, `_scratch/official_sequence.log` has START/DONE lines) | L40S | 2026-09-05 | **A** (BaB 100 s cap; α pass 30 s cap never hit) | **rerun both arms on the H200, alone, same config** |
| export-path controls (identity re-export vs stock max Δ 6.3e-5; deterministic levels match) | `official_base_export.json`, `official_base_export_cpu.json`, `official_idinitG_patched_cpu.json` | — | mixed: GPU runs and CPU runs (`run_controls_cpu.sh`) | L40S + CPU | 2026-09-05 | B (only the deterministic initial-CROWN level is compared across CPU/GPU) | none |
| `ibp_3_3_8`: SVD gauge looser 100/100 (15 → 12); learned gauge neutral (15 → 15); official stock 59/100 | `ibp_3_3_8__*.json`, `official_stock_ibp.json` | `R45_both_svd`, `G_ibp_mix_*` | interactive salloc, official run alone on g3120 | L40S | 2026-09-04/05 | B (vanilla), A (official) | rerun only if the paper quotes the official count |
| GenBaB ViTs `vit_{1_3,1_6,2_3,2_6}`: attention constant over the box, gauge Δ ≤ 5e-5 | `_scratch/` probe logs, diary 2026-09-05 cont. 3 | random / SVD probes | `run_genbab_smoke.sh`, `run_official_genbab.sh` (stock, alone) | L40S | 2026-09-05 | B (probe), A (300 s stock run) | none — mechanistic null |

### DeepT text models, plain-CROWN paired protocol (`deept_gauge.py eval`, certify-radius bisection + fixed ε)

| claim | results | gauge | job | hardware | date | class | H200 |
|---|---|---|---|---|---|---|---|
| SST `small_3` +1.7 % radius (attention 9 % of width), long-sentence transfer fades, lb tighter 251/251 | `deept_small3_eval_short_seed{0,1}.json`, `deept_small3_eval_long_seed{0,1}.json`, `deept_small3_alldev_*` | `gauges/deept_small3_seed{0,1}.pt` | short eval: L40S per diary (interactive salloc, job id not recorded); long/seed-0: `deept_A` 39641025, seed-1 long: `deept_B` 39641026 | L40S | 2026-09-05/06 | B | sanity smoke |
| **SST `small_6` +13.1 % radius, larger 273/294, smaller 0, ε 0.03 verified 41 → 95, 0 reverse** | `deept_small6_eval_short_seed0.json` (`_nancliff` = pre-fix run) | `gauges/deept_small6_seed0.pt` (68 dev boxes ≤ 8 tokens, learned in the same chain) | `deept_B` 39641026 (`deept_chain.sh small6`) | L40S | 2026-09-06 | B (E gauge) | sanity smoke (`unf_smoke` golden set) |
| `small_6` OOD tuning (random tokens / Yelp text) still helps → data-independence | `deept_small6_oodrandom_eval_short_seed0.json`, `deept_small6_oodyelp_eval_short_seed0.json` | `gauges/deept_small6_ood{random,yelp}*.pt` | `deept_ood_chain.sh` (L40S) | L40S | 2026-09-06/07 | B | extend (more probes) |
| `small_12` 5-box gauge +26.6 % (120/120), ε 0.01 verified 67 → 96 — **indicative only** | `deept_small12_eval_short_seed0.json` | `deept_small12_seed0.pt` (5 boxes ≤ 5 tokens: the learner OOM'd at ≥ 6 tokens on 44 GB and at 6 on 80 GB) | 39659465 `small12b` fallback | L40S (learner OOM also on A100 job 39662797) | 2026-09-06 | **C** | **run the full protocol with a ≥ 6-token learner on 141 GB** |
| Yelp `small_3` +9.5 %, SST `big_3` +9.3 %, Yelp `small_6` +4.9 % (NaN cliff), SST `smaller_3` neutral/mixed; share screen (≥ 30 % ⇒ gain, ≤ 14 % ⇒ don't) | `deept_big3_eval_short_seed0.json`, `deept_yelp3_eval_short_seed0.json`, `deept_yelp6_eval_short_seed0.json`, `deept_smaller3_eval_short_seed0.json`, `deept_*_attrib.json` | `deept_{big3,yelp3,yelp6,smaller3}_seed0.pt` | `deept_yelp` 39697769, `deept_width` 39697876 (`deept_yelp_chain.sh`) | L40S | 2026-09-06 | B | sanity smoke; Yelp `small_6` beyond the NaN cliff is open |
| attention-share attribution (`attrib`, `attrib_split`) | `deept_*_attrib.json`, `deept_small6_attrib_split.json` | — | `attrib_split_s6` 39741703/4 | A100 (ckpt) | 2026-09-07 | B | none |

### Manual gauge (FORMULA.md), plain-CROWN paired protocol

| claim | results | gauge | job | hardware | date | class | H200 |
|---|---|---|---|---|---|---|---|
| unified rule paired share 0.91 `big_3` / 0.84 Yelp `small_3` / **1.00 `small_6`** (closed form alone 0.92) | `deept_formula_{big3,yelp3,small6}_eval_short_seed0.json`, `formula_*.json` screens | `gauges/formula_*_l1N_*.pt` | round 5/6 chains (`gauge_formula_chain.sh`, `deept_formula_eval.sh`), screens `jobs/screen_*.sbatch` | L40S (evals, screens) and ckpt A100 (learner warm starts, some screens) | 2026-09-08/09 | B (E gauges) | sanity smoke |
| warm-started learner ties the learned gauge (learned = ceiling of its objective) | `formula_small6_init.json`, `deept_formula_*_r3*` | `formula_*_init.pt` | 39968836 (L40S, 1 h 57) | L40S | 2026-09-10 | B | none |
| sign-flip / diag(±1) tests: plain CROWN neutral, CROWN-Optimized moves | `sign_flip_small6_signs.json` | diag gauges | 39969671, 39971187 | L40S | 2026-09-10 | B | none |
| sign-aware per-class screens (`formula_big3_sgn{0,1}.json`, `formula_small6_sgn*`) | as named | `formula_*_sgn*_lab{0,1}.pt` | 39973109, 39976535, 39976552, 39979851, 39981960 (L40S); optimisations `sgnopt*.sbatch` on ckpt CPU | L40S + CPU | 2026-09-10 | B (E gauges) | none |
| **Round 7 paired: `small_6` single learned +13.1 % / per-class learners +18.6 % (232/0) / manual per-class +18.0 % (246/0); `big_3` +11.9 % / +13.3 % (167/15, 17-box label-1 learner) / +14.3 % (208/0)** | `deept_formula_small6_{sgnlab,lrnlab}_eval_short_seed0.json`, `deept_formula_big3_{sgnlab,lrnlab}_eval_short_seed0.json`, combined by `perclass_paired.py` | `formula_*_sgn_mix_qk_mix_lab{0,1}.pt`, `deept_*_lab{0,1}.pt` | `feval_s6_sgnlab` 39996390, `feval_s6_lrnlab` 39996391, `feval_b3_sgnlab` 39996392, `feval_b3_lrnlab` 39996393 | L40S | 2026-09-10/11 | B (E gauges) | sanity smoke; the "one-gauge bar" (0.94 / 0.93) is still open and card-independent |

### α-CROWN tier (`deept_gauge.py eval_alpha`, CROWN-Optimized 20 it, fixed ε)

| claim | results | gauge | job | hardware | date | class | H200 |
|---|---|---|---|---|---|---|---|
| `small_3` α: tighter 202/205 | `deept_small3_eval_alpha_seed0.json` | `deept_small3_seed0.pt` | `deept_A` 39641025 | L40S | 2026-09-06 | B | none |
| **`small_6` α (49 inst ≤ 6 tokens, ε 0.02): 24 → 27, tighter 47/47, NaN 2 → 0** | `deept_small6_eval_alpha_seed0.json` | `deept_small6_seed0.pt` | `deept_a100b` 39663976 | **A100** (OOM on L40S) | 2026-09-06 | B + **C** | **extend to ≤ 7–8 tokens** |
| `small_6` α ε curve 0.015 / 0.025; `big_3` α 29 inst ≤ 5 tokens | `deept_small6_eval_alpha_eps2_seed0.json`, `deept_big3_eval_alpha_l5_seed0.json` | single seed-0 gauges | `alpha_s6_eps` 39969259, `alpha_b3_l5` 39969630 | A100 | 2026-09-10 | B + C | extend `big_3` to ≥ 6 tokens |
| alternating α / learner (`alt`) | `deept_small6_alt_eval_alpha_seed0.json` | `deept_small6_alt_seed0.pt` | 39969388 | A100 | 2026-09-10 | B, D (3 h 26 learner) | none |
| **per-class α: `big_3` manual tighter than learners 29/29 at both ε; `small_6` 27 → 28 verified, manual vs learners 27/22, vs single 37/12** | `deept_{small6,big3}_eval_alpha_{manlab,lrnlab}{0,1}.json`, combined by `perclass_alpha.py` | round-7 per-class gauges | `a_s6_man{0,1}` 40023058/60, `a_b3_man{0,1}` 40023059/61, `a_b3_lrn{0,1}` 40024560/1, `a_s6_lrn{0,1}` 40024562/3 (the L40S attempts 40022976–9 OOM'd) | A100 | 2026-09-11 | B + C | extend token limit |

### Rigorous certificate transfer (`deept_unfolded.py`)

| claim | results | gauge | job | hardware | date | class | H200 |
|---|---|---|---|---|---|---|---|
| **G⁻¹/A⁻¹ as 2-ulp fp32 intervals: 294/294 radii unchanged, `--confirm` all lb > 0 (min +3.96e-5), verified inclusion ρ ≤ 3.1e-12** | `deept_small6_unf_full_cert.json` (+ `unf_med8`, `unf_smoke`, `unf_smoke_d5e9`) | `deept_small6_seed0.pt` | `jobs/unf.sbatch`: smoke 39998394 / 39998395 (`smoke`, `smoke_d5e9`), 39999430 (`med8` + `full_cert`) | A100 (ckpt) | 2026-09-11 | B (radii), **D** (2.3× per call, 1.65× total: same-card ratios) | golden smoke (see `TILLICUM_SETUP.md`); re-measure the ratio |

### Composition with Huang et al. (AAAI-26), `RELATED_WORK.md`

| claim | results | job | hardware | date | class | H200 |
|---|---|---|---|---|---|---|
| gauge in their verifier: `small_6` Baseline +13.4 % (29/6), PBverifierT +3.3 % (35/0), PBverifierI +2.9 % (18/17); `small_3` neutral | grids in `RELATED_WORK.md`; logs `deept_benchmarks/PBVerification/…` | `pbv_s6_I` 39672839, `pbv_s3_I` 39672840, `pbv_s3_v0` 39885882, `pbv_s6_v0` 39885883 | L40S | 2026-09-06/09 | B; **D** for "≈ 3 min per instance" | none |
| their retrained `model_sst_3`: auto_LiRPA +9.5 % (244/276), their Baseline +8.4 % (52/52), PBverifierI/T +6.1 / +6.0 % (20/20) | `RELATED_WORK.md` grids | `pbvT_*` 39696050–39696054 | A100 (ckpt) | 2026-09-06/07 | B | none |
| gauges learned through their bound (`pbv_learn.py`): tangent-trained +13.2 % / Baseline-trained overfit −16.3 % under auto_LiRPA | `RELATED_WORK.md` | pbv learner jobs (ckpt A100; ≤ 6-token boxes, 80 GB OOM above) | A100 | 2026-09-07 | B + C, E | a ≥ 8-token learner on 141 GB would test the thin-set overfitting story |

### Memory and runtime numbers (all class C or D)

| number | measured on |
|---|---|
| `small_6` α 36 / 62 / ≈ 84 GiB at 5 / 6 / 7 tokens; `big_3` α 6 tokens 77.6 GiB | A100 80 GB (`probe_big3.sbatch`, `alpha_gauge_ckpt.sbatch`) |
| `small_12` learner OOM ≥ 6 tokens (43.9 GiB) / 6 tokens 78.6 GiB | L40S / A100 |
| ViT-tier alpha-CROWN 14.5 GiB at 6 tokens, 35.9 at 8, OOM at 10 (README line "L40S 44 GB") | L40S |
| `sparse_intermediate_bounds=False`: identical bound at 0.24 GiB vs 18.7 GiB peak | L40S |
| PBVerification learner OOM at ≤ 8-token boxes (`small_6`), ≤ 6 on their hidden-256 | A100 |
| unfolded 22 s per instance, 4 CROWN calls, 92 min for 294 instances | A100 |

## How to cite a claim in the paper

State the card for every class-A number ("on one L40S, 100 s per instance"), give the token limit and the reason for
every class-C set, and quote class-D numbers only with the card they were measured on. Class-B numbers need no card in the
text but the appendix should name it (this file). If the final experiments are rerun on the H200, the paper reports H200
numbers throughout for consistency, and the Hyak numbers stay here as the pre-migration record.
