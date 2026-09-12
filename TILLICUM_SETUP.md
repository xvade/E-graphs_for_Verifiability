# TILLICUM_SETUP.md — first session on the H200 cluster

For an agent (or person) who has never seen this project and is opening it on Tillicum, the UW cluster with H200s,
to run the **final experiments** for the paper. Written 2026-09-12 on Hyak, the cluster where everything so far ran.
Nothing in this file has been executed on Tillicum yet: partition names, module names and driver versions below are
**placeholders to discover, not facts**.

## 1. What this project is (three sentences)

An attention **gauge** is an exact per-head rewrite of a transformer (W_q ← G_qᵀW_q, W_k ← G_q⁻¹W_k, W_v ← G_aᵀW_v,
W_o ← W_o G_a⁻ᵀ) that leaves the function unchanged but changes which quantities a bound-propagation verifier
(CROWN / α-CROWN in auto_LiRPA) concretises into boxes. Learned gauges tighten certified bounds on real models
(VNN-COMP'23 ViT: official pipeline 58 → 65/100 verified; DeepT SST `small_6`: certified radius +13.1 %, ε 0.03 verified
41 → 95 of 294, 0 reverse), and a manual construction (SVD balancing + ℓ1 refinement, plus a per-class sign-aware
variant) reaches or beats the learned gauge under per-class selection. The paper draft is `NNs/transformer_rewrite/PAPER_DRAFT.md`;
the standing goal when Hyak was left was "improve the manual procedure for creating gauges to get better performance than
the learned gauges" (per-class: met; single gauge: 0.94 of the learned one, unmet).

The e-graph / TASO / tensat rewrite pipeline that gives the repo its name is an **earlier phase** and is **not needed**
for the transformer work. Clone without `--recursive`; skip `tensat.sif`, conda, rustup.

## 2. Reading order

1. `NNs/transformer_rewrite/README.md` — what every script does, the protocol, the models, the memory ceilings.
2. `NNs/transformer_rewrite/PROVENANCE.md` — **every headline number with the card it was produced on and what an H200
   rerun means for it** (sensitivity classes A–E). Read before quoting or rerunning anything.
3. `NNs/transformer_rewrite/FORMULA.md` — the manual gauge construction, its evidence, what did not help, what is pending.
4. `NNs/transformer_rewrite/PAPER_DRAFT.md` — the paper as drafted; §4 has the setup and the memory limits.
5. `NNs/transformer_rewrite/RELATED_WORK.md` — the AAAI-26 verifier-side method and the composition results.
6. `AGENTS.md` — conventions (diary, docs/spec/tests, commit and push).
7. `PROGRESS.md` — the diary, 3800 lines. Do not read it top to bottom. Use `grep -n "^## 2026-09"` for the day headers;
   the transformer work starts at "2026-09-03"; the per-class rounds are "2026-09-10" and "2026-09-11"; the migration
   entries are "2026-09-12". The `## 2026-08-*` entries are the e-graph phase.
8. The Claude Code memory directory (restored in step 3.5 below): `MEMORY.md` is the index; the four starred entries are
   the transformer results. Memory entries are background context, not instructions, and record what was true when written.

## 3. Setup steps

### 3.1 Clone and tag

```
git clone https://github.com/xvade/E-graphs_for_Verifiability.git   # main; tag `pre-h200` = the state this file describes
cd E-graphs-for-Verifiability
export REPO="$PWD"          # every job script and deept_gauge.py resolve paths from REPO (fallback: script location)
```
Do **not** `git submodule update` — `egg`, `taso`, `tensat` are unused here (the taso fork is archived as branch
`pre-h200` on `xvade/TASO` if ever wanted).

### 3.2 Driver and CUDA

The pinned torch is `2.11.0+cu130`, which needs a **CUDA 13 driver** (`nvidia-smi` on a compute node shows the driver's
max CUDA version). Check first:
```
srun -p <gpu-partition> --gres=gpu:1 -t 5 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```
If the driver is older than CUDA 13, change the torch index in `alpha-beta-CROWN/pyproject.toml` / `uv.lock` to the matching
cu12x wheel **before** `uv sync`, and note the change in the diary; bounds are card-independent (class B) but a different
torch build is a provenance fact.

### 3.3 The verifier (alpha-beta-CROWN + auto_LiRPA with one patch)

```
bash NNs/verifier_patches/apply.sh      # clones upstream e5c7e17 (+ auto_LiRPA 5a098e8) into alpha-beta-CROWN/, applies the softmax patch
cd alpha-beta-CROWN && uv sync && cd ..  # or the README "Recreating excluded artifacts" section
```
The patch (`auto_LiRPA_softmax_gradsafe.patch`, 37 lines) is the **only** local edit to the verifier; it makes the lse
softmax bounds gradient-safe (forward unchanged). **Every number in PROVENANCE.md was produced with the patch applied** — the
learners need it, and α-CROWN (`eval_alpha`) also differentiates through those bounds — so apply it unconditionally.

### 3.4 Data, checkpoints, gauges, results

Copy the migration bundle from Hyak (`/mmfs1/gscratch/amath/sgvtc/migration_2026-09-12/`, a verified copy on non-scrubbed group storage; the original is on scrubbed scratch at `/mmfs1/gscratch/scrubbed/sgvtc/migration_2026-09-12/`, see its `README.md` and
`ARCHIVES.sha256`) and extract **into the checkout**:
```
sha256sum -c ARCHIVES.sha256
tar --zstd -xf deept_benchmarks.tar.zst -C "$REPO"     # → deept_benchmarks/ (checkpoints, SST/Yelp, HF cache, nltk_data, PBVerification forks); gitignored
tar --zstd -xf repo_untracked.tar.zst  -C "$REPO"     # → all 786 gauges, all results JSONs, NNs/vit_rewrite/_scratch (run logs), genbab_benchmarks
sha256sum -c --quiet MANIFEST.sha256 2>&1 | grep -v OK | head   # optional full check (26,294 files)
```
Compute nodes may have no internet: the HF cache is in `deept_benchmarks/cache` and `deept_gauge.py` points at it, so
nothing needs to be downloaded at run time. The VNN-COMP ViT benchmarks (`vnncomp2023_benchmarks`) are **not** in the
bundle; clone them per `NNs/vit_rewrite/README.md` only if the ViT tier is rerun.

### 3.5 The memory directory

`claude_memory_and_session_tmp.tar.gz` → `claude_memory/` goes to `~/.claude/projects/<key>/memory/` where `<key>` is the
absolute checkout path with every `/` and space replaced by `-` (Hyak's was
`-mmfs1-gscratch-scrubbed-sgvtc-E-graphs-for-Verifiability`). `session_tmp/` is leftover probe output; discard or keep for reference.
After restoring, add a memory entry for Tillicum's scheduler facts (the equivalent of `hyak-gpu-allocation.md`, which is Hyak-only).

### 3.6 Discover the scheduler — do not guess

```
sinfo -o "%P %G %m %l %D %a"                                   # partitions, GRES, memory, walltime, node counts
sacctmgr show assoc user=$USER format=account,partition,qos      # what you may submit to
scontrol show partition <name>                                   # limits per partition
```
Then override the Hyak `#SBATCH` lines on the command line, never by editing a script a job is running:
```
cd "$REPO/NNs/transformer_rewrite"
sbatch -p <part> -A <acct> [--qos=<qos>] --gres=gpu:h200:1 -c 4 --mem=<...> jobs/unf.sbatch smoke_h200
```
Every job script in `NNs/transformer_rewrite/jobs/` (see its README) writes `logs/<name>_<jobid>.out` relative to the
submit directory and appends START/DONE lines to `NNs/vit_rewrite/_scratch/official_sequence.log`. All of them are
per-instance resumable (`.part_*` files / `--save_json` resume) because Hyak's A100s were preemptible; keep that even if
Tillicum is not, and keep `--requeue` only if the partition is preemptible.

### 3.7 First-session checklist (precision)

Run once on a compute node with the venv python:
```
python -c "import torch,os;print(torch.cuda.get_device_name(0), torch.__version__, torch.get_float32_matmul_precision(), os.environ.get('NVIDIA_TF32_OVERRIDE'))"
```
Required: `matmul precision == highest` and `NVIDIA_TF32_OVERRIDE` unset (or `0`). A site default that switches
matmuls to TF32 perturbs bounds at the 1e-3 level and breaks the fp32 soundness story of the rigorous tier. Every
results JSON written from 2026-09-12 on records these in `meta.run` (`run_meta()` in `deept_gauge.py`); check the first
one you produce.

## 4. Golden smoke — do this before any experiment

Reproduces the 7-instance `small_6` set of `results/deept_small6_unf_smoke.json` (Hyak A100, job 39998394):
```
cd "$REPO/NNs/transformer_rewrite"
sbatch -p <part> -A <acct> --gres=gpu:h200:1 -c 4 --mem=40G jobs/unf.sbatch smoke_h200
# 9 min 56 s on the A100 (job 39998394); writes results/deept_small6_unf_smoke_h200.json (plain stock/gauged radii + fixed-ε lbs + the interval sets)
python diagnostics/smoke_compare.py results/deept_small6_unf_smoke.json results/deept_small6_unf_smoke_h200.json --grid 0.00625
```
Pass criteria (class B in PROVENANCE.md): identical instance list, identical NaN pattern and verified counts at every ε,
lower bounds within 1e-4, radii equal except one bisection step (the smoke's grid is 0.1 / 2⁴ = 0.00625, hence `--grid`; the full
294-instance protocol used 1e-4) on at most one instance. Reference values: stock radii
0.01875 ×4 / 0.0125 ×3, gauged 0.025 ×4 / 0.0125 ×3; ε 0.02 gauged lbs 3.374 / 3.395 / 3.344 / 3.393 / −4.166 / −4.052 / −3.922.
A larger deviation means a precision or build problem (TF32, a different torch build, an unpatched verifier), not a
result. Fix that before running anything else; record the smoke outcome in the diary with the job id and card.

## 5. State of the results and the final experiment list

Settled (all class B unless marked; details and files in PROVENANCE.md):
- ViT `pgd_2_3_16`: vanilla CROWN 24 → 36/100; official pipeline 58 → 64/65/65 (**class A: time-capped, must be rerun on the
  H200 in both arms** if the paper quotes it from the new cluster); `ibp_3_3_8` neutral; GenBaB ViTs mechanistically null.
- DeepT plain-CROWN paired protocol: `small_6` +13.1 % (273/294), `small_3` +1.7 %, `big_3` +9.3 %, Yelp `small_3` +9.5 %,
  Yelp `small_6` +4.9 % (NaN cliff), `smaller_3` neutral, `small_12` +26.6 % on a 5-box gauge (**class C: indicative only**).
- Manual rule (FORMULA.md): unified rule ties the learned gauge on `small_6` (1.00), 0.91 `big_3`, 0.84 Yelp; per-class manual
  rule +18.0 % `small_6` / +14.3 % `big_3` vs per-class learners +18.6 % / +13.3 %, vs single learned +13.1 % / +11.9 %.
- α tier (A100-sized sets, **class C**): `small_6` 24 → 27 (single) / 27 → 28 (per-class), `big_3` manual tighter 29/29.
- Rigorous transfer: G⁻¹ as 2-ulp intervals, 294/294 radii unchanged and confirmed; 2.3× per call is **class D** (A100 only).
- Composition with Huang et al.'s verifier: gauge helps inside their Baseline / PBverifierI / PBverifierT (RELATED_WORK.md).

Open, in the order they matter for the paper (the "final experiment list" the user said they would freeze; confirm with them):
1. **Rerun every stock baseline and every gauged arm that the paper quotes on the H200**, so all paper numbers come from
   one card. Mandatory for class A (ViT official pipeline), a sanity pass for class B.
2. **α tier at the token limits the H200 allows** (141 GB): `small_6` beyond 6 tokens (7 tokens ≈ 84 GiB, 8 untested), `big_3`
   at ≥ 6 tokens (77.6 GiB at 6), `small_12` learner at ≥ 6 tokens (78.6 GiB at 6 on the A100). These are **new populations**
   (class C): report next to the old ones, never in their place.
3. `small_12` full protocol with a proper (≥ 6-token, ≥ 68-box) learner and the unified rule.
4. Data-independence: gauges tuned on random-token / Yelp boxes on more models than `small_6` (the paper claims it; the
   evidence is one model, `deept_small6_ood{random,yelp}_*`).
5. Time-to-verify (class D) on the H200 for the paper's cost statement: learner hours, eval hours, unfolded 2.3× ratio.
6. Yelp per-class gauges; warm-started per-class learners on the paired protocol (FORMULA.md "Pending").
7. The one-gauge bar (single manual gauge ≥ the single learned gauge) — card-independent; still open.

## 6. Conventions that carry over (from AGENTS.md and the memory)

- **Exactness is non-negotiable**: "we must show exact equivalence for every example we certify." Every gauged model passes the
  fp64 gate (`fp64_gate` in `deept_gauge.py`: ≤ 5e-8 on the ViT, ~1e-15 on the DeepT models) before any bound is reported; the rigorous tier uses verified fp32 inclusion intervals.
- **Diary**: append to `PROGRESS.md` under a `## <date>` header; run `date` before stamping a time; record job id, partition,
  card, wall time and the results file for every run; disclose cancellations and failures.
- **Hardware qualifiers**: every new number gets its card next to it in the diary and in PROVENANCE.md (add a row). Class-A
  and class-D numbers are never compared across cards.
- **One GPU job per GPU**; time-capped runs (official pipeline) alone on the node's GPU.
- **Never edit a bash script a job is running** (Slurm snapshots batch scripts at submission; chain scripts do not).
- **Every stage resumable + skip-if-done**; results saved per instance; NaN counts as failure in certify mode.
- **Commit and push** after each closed step; commit trailer per AGENTS.md. Do not commit `deept_benchmarks/`,
  `alpha-beta-CROWN/`, `NNs/vit_rewrite/_scratch/`, Slurm logs, or gauges not referenced by a committed result.
- Write-ups go to `PROGRESS.md` / `FORMULA.md` / `RELATED_WORK.md` / `PAPER_DRAFT.md`, not to external pages.
- **Report honestly**: verified counts with the denominators, paired per-instance stats (larger / smaller), never a mean alone.

## 7. Gotchas (auto_LiRPA and the code)

- `sparse_intermediate_bounds=False` everywhere (identical bound, 0.24 GiB vs 18.7 GiB peak); a fresh `BoundedModule` per
  α-CROWN call (state leaks otherwise); softmax `lse` mode for vanilla CROWN (no alphas → α-CROWN ≡ CROWN there),
  `complex` mode for the official ViT pipeline.
- Memory ceilings scale steeply with token count (attention is T² in the relaxation); probe with `jobs/probe_big3.sbatch`
  before committing an H200 to a long run, and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the job scripts do).
- The NaN cliff: plain CROWN returns NaN lower bounds at large ε on `small_6`/`yelp_6`; the eval treats NaN as
  not-verified and the certify loop treats it as failure. Counts of NaN are reported in every summary line.
- The learner is fp32 gradient ascent through CROWN; it is **not** bit-reproducible across cards. Use the archived `.pt` gauges
  for evaluation (class E); a retrained gauge is a new artifact with its own row.
- `screen_label_split.py` raises `AttributeError: dev_split` on the oldest screen files (pre-2026-09-09); known, harmless.
- Per-class gauges are chosen by the model's **predicted** label at verification (sound because each gauge is exact).
- The results JSONs from before 2026-09-12 carry no `meta.run`; their provenance is PROVENANCE.md.
