# transformer_rewrite — attention-gauge rewrite on downloaded transformers

Follow-up to `NNs/vit_rewrite/` (VNN-COMP 2023 ViT result). Same exact rewrite family — per-head invertible `G` inserted
as `(Q G)(K G^{-T})^T` and `(A V Ga)(Ga^{-1} W_o)` — applied to transformers downloaded from the internet that come with
a verification spec. Nothing hand-made; all models are the authors' released checkpoints.

| target | source | spec | outcome |
|---|---|---|---|
| GenBaB ViTs `vit_1_3/1_6/2_3/2_6` | HF `zhouxingshi/GenBaB` (TACAS'25), `genbab_benchmarks/` | vnnlib, CIFAR-10 ε=1/255, abcrown config | **no leverage**: attention is constant over the boxes (exactly uniform / saturated softmax) — see `NNs/vit_rewrite/genbab_gauge.py`, PROGRESS.md |
| DeepT SST transformers `sst_bert_small_3` (…6/12) | `eth-sri/DeepT` (PLDI'21), `deept_benchmarks/` | ℓ∞ ball (eps) around ONE word embedding of a test sentence; certified radius per (sentence, position) | `deept_gauge.py` — learner + paired eval (results in PROGRESS.md) |

## Files
- `deept_gauge.py` — the whole DeepT harness (one file, `python deept_gauge.py <cmd> --name sst_bert_small_3 ...`):
  - `probe` — fidelity of our re-implemented forward vs the DeepT forward (fp64), softmax interval widths, gauge sensitivity;
  - `radii` — stock certified radii (bisection on the vanilla CROWN lb, `--hi`, `--iters`);
  - `learn` — gauge learner: gradient ascent on the vanilla CROWN(lse) lb over one-word ε-boxes of SST **dev** sentences
    ≤ `--max_len` tokens (`--n_sent` sentences × `--pos_per_sent` positions), per-box eps = `--eps_scale` × stock certified
    radius; `--obj mean|hinge`, `--steps`, `--accum`, `--lr`, `--cond_pen`, `--clip`, `--seed`; saves `{"qk","av",...}` to `--out`;
  - `eval` — paired stock vs gauged on SST **test** sentences: certified radii on the same bisection grid + fixed-eps
    verified counts (`--eps_list`), flips, fp64 exactness gate; `--save_json` writes per-instance arrays;
  - `eval_alpha` — the same pairing with alpha-CROWN (`CROWN-Optimized`, 20 it, autograd on) at fixed eps — fits the 44 GB
    GPU only for sentences ≤ 8 tokens;
  - `attrib` — attention-slack attribution: CROWN width with the attention probabilities frozen at their box-centre values
    (removes the QKᵀ-bilinear, softmax and A·V slack) vs the real width, at eps = 1× and 1.5× the stock radius.
  The rewrite itself is `effective()` (nn.Linear convention, `W_q[h] ← Gᵀ W_q[h]`, `W_k[h] ← G⁻¹ W_k[h]`, `W_v[h] ← Gaᵀ W_v[h]`,
  `W_o[:, h] ← W_o[:, h] Ga⁻ᵀ`, biases likewise) and `load_eff()` copies the result into the live model (shared by all
  per-length `BoundedModule`s).
- `run_deept_cmd.sh <cmd> [args]` — generic Slurm-step wrapper (venv python, cwd = this directory); `run_deept_probe.sh`,
  `run_deept_radii.sh`, `run_deept_learn.sh <name> <out.pt> [args]` (absolutizes `--out`), `run_deept_eval_chain.sh <seed> <node>`
  (waits for the learner's DONE marker in `NNs/vit_rewrite/_scratch/official_sequence.log`, then short + long paired evals).
- `deept_chain.sh <what>...` — the follow-up chains as ONE script that runs either as a batch job (`sbatch -p gpu-l40s
  -A gpu-l40s-amath --gres=gpu:l40s:1 -c 5 --mem=25G -t 16:00:00 deept_chain.sh alpha small12 long0`) or inside an existing
  allocation (`JOBID=<id> NODE=<node> deept_chain.sh ...`); modes `alpha` (small_3 alpha tier), `alpha6`/`alpha5` (small_6 alpha
  tier at ≤ 6 / ≤ 5 tokens), `alldev`, `small6`, `small12`/`small12b`, `long0/1` are documented in its header.
- `export_gauged_ckpt.py --name <deept model> --gauge <pt|none> --out <dir>` — folds a learned gauge into a DeepT checkpoint and writes
  the Shi et al. 2020 directory layout (`checkpoint` + `ckpt-N/{config.json,vocab.txt,pytorch_model.bin}`), so third-party verifiers
  built on that codebase (DeepT, PBVerification) load stock and gauged weights the same way; used for the composition test in
  `RELATED_WORK.md` (their verifier lives in `deept_benchmarks/PBVerification/`, wrappers `run_pbv.sh`, `run_pbv_chain.sh`).
- `RELATED_WORK.md` — what others have done (AAAI-26 parameterised abstract interpretation, GaLileo, Vertex-Softmax, gauge-symmetry
  papers, …), how it relates to the gauge rewrite, and the stock-vs-gauged runs of their verifier.
- `run_on_bigger_gpu.sh <chain args>` — sbatch wrapper for steps that need > 44 GB (alpha-CROWN on small_6): checks the card has
  ≥ 60 GB, then runs `deept_chain.sh`; used with `-p ckpt-all -A ckpt-amath --qos=ckpt-gpu --gres=gpu:a100:1`.
- `diagnostics/` — the one-off memory / NaN / sharing investigations behind the facts below (`_mem_probe*.py`, `_alpha_mem_probe*.py` +
  wrappers, `_nan_alpha_check.py`, `_nan_cliff_probe.py` + `run_nan_cliff_probe.sh` (is the bisection radius a zero crossing or the lse NaN
  cliff?), `_freeze_check.py`, `_count_short.py`); run from this directory. Kept for the record, not maintained.
- `gauges/deept_small3_seed{0,1}.pt` — learned gauges (119 dev boxes, 120 steps); `deept_small6_seed0.pt` (68 boxes from 23 dev
  sentences ≤ 8 tokens), `deept_small3_alldev_seed0.pt` (305-box overfitting control), `deept_small12_seed0.pt` (5 boxes; ≤ 5 tokens); `results/*.json` — per-instance eval
  arrays (`inst=[sentence idx, position, n_tokens, label]`, `stock_rad`, `gauged_rad`, `fixed[eps][stock|gauged]`) and the
  attribution numbers. Logs live in `NNs/vit_rewrite/_scratch/deept_*.log` (gitignored).

## Results (DeepT `sst_bert_small_3`, vanilla CROWN, out-of-sample; details + caveats in PROGRESS.md 2026-09-05 cont. 3)
| test sentences ≤ 12 tokens, 278 positions | stock | gauge seed 0 | gauge seed 1 |
|---|---|---|---|
| mean certified radius | 0.0331 | 0.0340 (larger 211 / smaller 13 / equal 54) | 0.0336 (155 / 9 / 114) |
| verified at eps 0.03 | 147 | 149 (2 up, 0 down) | 150 (3 up, 0 down) |

| **`sst_bert_small_6`**, test sentences ≤ 12 tokens, 294 positions | stock | gauge seed 0 (tuned on 68 dev boxes ≤ 8 tokens) |
|---|---|---|
| mean certified radius | 0.0220 | **0.0249 (+13.1% in mean radius; +10.6% mean per-instance)**, larger 273 / smaller 0 / equal 21 |
| verified at eps 0.02 / 0.03 | 181 / 41 | 183 / **95** (0 reverse flips) |
| alpha-CROWN tier (A100 80 GB; 49 positions ≤ 6 tokens), verified at eps 0.02 | 24 | **27** (3 up, 0 down; tighter on 47/47 finite) |

small_12 (5-box ≤ 5-token gauge, the most the 12-layer learner fits in 44 GB; 120 test positions ≤ 10 tokens): radius +26.6 % (120/120 larger), eps 0.01 verified 67 → 96 (0 reverse), NaN 39 → 13 — NaN-cliff-limited radii, so only indicative (`results/deept_small12_eval_short_seed0.json`).

Attention nonlinearities are 9.2% of the CROWN width on small_3, 39.7% on small_6, 70.0% on small_12 (eps = stock radius;
77% on the VNN-COMP ViT where the gauge flipped 7/100 at the full tier, 3% on `ibp_3_3_8` where it was neutral):
**gauge leverage ≈ attention share of the width** — small_3 +1.7% radius, small_6 +13.1%.
fp64 exactness gate 8.9e-16; newly verified margins ≥ 1.3e-2 vs fp32 stock-vs-gauged discrepancy ≤ 4.8e-7.

## auto_LiRPA facts learned here
- `sparse_intermediate_bounds` (default True) makes lse-CROWN on a 12-token BERT-style transformer peak at 18.7 GiB and
  OOM at 16 tokens; `False` gives the identical bound at 0.24 GiB. Always set it for transformers with >5 tokens.
- With autograd enabled (gauge learning, alpha-CROWN) the retained graph costs ~21 GiB at 12 tokens (OOM at 20);
  alpha-CROWN: 14.5 GiB at 6 tokens, 35.9 GiB at 8, OOM at 10 (L40S 44 GB). So tuning is done on sentences ≤ 10 tokens and
  the alpha/BaB tier is only reachable for ≤ 8-token sentences on this hardware.
- `softmax: complex` mode needs `fixed_reducemax_index: True` and hit an AssertionError on some boxes here; `lse` is used
  throughout (NaN only far above the certified radius — counted and treated as unverified).
- alpha-CROWN memory on the 6-layer model: per-call peak 36 GiB (5 tokens) / 62 GiB (6 tokens) / ~84 GiB (7, extrapolated),
  and each BoundedModule retains 4.5–8 GiB after a call — so `eval_alpha` builds a fresh module per call and keeps the weights
  frozen (`diagnostics/_alpha_mem_probe*.py`). The small_6 alpha tier therefore needs an 80 GB card and ≤ 6-token sentences.
- `CROWN-Optimized` inside `torch.no_grad()` silently returns plain CROWN (alphas cannot step) — the earlier "alpha ==
  CROWN" reading was that artifact.

Provenance of the alpha-tier numbers: the small_3 alpha row came from the earlier `eval_alpha` (one BoundedModule per sentence
length, weights with autograd on); the small_6 row from the current one (fresh module per call, weights frozen). Rerunning 18 small_3
positions through the current code reproduced the earlier bounds bit-for-bit (`results/deept_small3_eval_alpha_check.json`).

## Reproduce (DeepT small_3; small_6 analogously via `deept_chain.sh small6` then `run_on_bigger_gpu.sh alpha6`)
```
python NNs/vit_rewrite/genbab_download.py                      # GenBaB models (105 MB)
git clone --depth 1 https://github.com/eth-sri/DeepT deept_benchmarks/DeepT ; # + data.tar.gz from their install.sh -> deept_benchmarks/data
uv pip install --python alpha-beta-CROWN/.venv/bin/python pytorch-pretrained-bert
NNs/transformer_rewrite/run_deept_learn.sh sst_bert_small_3 NNs/transformer_rewrite/gauges/deept_small3_seed0.pt --max_len 10 --n_sent 40 --pos_per_sent 3 --steps 120 --accum 4 --seed 0
NNs/transformer_rewrite/run_deept_cmd.sh eval --name sst_bert_small_3 --gauge gauges/deept_small3_seed0.pt --max_len 12 --n_sent 40 --seed 0 --eps_list 0.01,0.02,0.03 --hi 0.1 --iters 10 --save_json results/deept_small3_eval_short_seed0.json
NNs/transformer_rewrite/run_deept_cmd.sh attrib --name sst_bert_small_3 --max_len 12 --n_sent 12 --pos_per_sent 2 --seed 3 --save_json results/deept_small3_attrib.json
```
