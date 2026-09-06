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
- `diagnostics/` — the one-off memory / NaN / sharing investigations behind the facts below (`_mem_probe*.py` + wrappers,
  `_nan_alpha_check.py`, `_freeze_check.py`, `_count_short.py`); run from this directory. Kept for the record, not maintained.
- `gauges/deept_small3_seed{0,1}.pt` — learned gauges (119 dev boxes, 120 steps); `results/*.json` — per-instance eval
  arrays (`inst=[sentence idx, position, n_tokens, label]`, `stock_rad`, `gauged_rad`, `fixed[eps][stock|gauged]`) and the
  attribution numbers. Logs live in `NNs/vit_rewrite/_scratch/deept_*.log` (gitignored).

## Results (DeepT `sst_bert_small_3`, vanilla CROWN, out-of-sample; details + caveats in PROGRESS.md 2026-09-05 cont. 3)
| test sentences ≤ 12 tokens, 278 positions | stock | gauge seed 0 | gauge seed 1 |
|---|---|---|---|
| mean certified radius | 0.0331 | 0.0340 (larger 211 / smaller 13 / equal 54) | 0.0336 (155 / 9 / 114) |
| verified at eps 0.03 | 147 | 149 (2 up, 0 down) | 150 (3 up, 0 down) |

Attention nonlinearities are 9.2% (eps = radius) / 12.8% (1.5×) of the CROWN width on this model+spec (77% on the VNN-COMP
ViT where the gauge flipped 7/100, 3% on `ibp_3_3_8` where it was neutral): **gauge leverage ≈ attention share of the width.**
fp64 exactness gate 8.9e-16; newly verified margins ≥ 1.3e-2 vs fp32 stock-vs-gauged discrepancy ≤ 4.8e-7.

## auto_LiRPA facts learned here
- `sparse_intermediate_bounds` (default True) makes lse-CROWN on a 12-token BERT-style transformer peak at 18.7 GiB and
  OOM at 16 tokens; `False` gives the identical bound at 0.24 GiB. Always set it for transformers with >5 tokens.
- With autograd enabled (gauge learning, alpha-CROWN) the retained graph costs ~21 GiB at 12 tokens (OOM at 20);
  alpha-CROWN: 14.5 GiB at 6 tokens, 35.9 GiB at 8, OOM at 10 (L40S 44 GB). So tuning is done on sentences ≤ 10 tokens and
  the alpha/BaB tier is only reachable for ≤ 8-token sentences on this hardware.
- `softmax: complex` mode needs `fixed_reducemax_index: True` and hit an AssertionError on some boxes here; `lse` is used
  throughout (NaN only far above the certified radius — counted and treated as unverified).
- `CROWN-Optimized` inside `torch.no_grad()` silently returns plain CROWN (alphas cannot step) — the earlier "alpha ==
  CROWN" reading was that artifact.

## Reproduce (DeepT small_3)
```
python NNs/vit_rewrite/genbab_download.py                      # GenBaB models (105 MB)
git clone --depth 1 https://github.com/eth-sri/DeepT deept_benchmarks/DeepT ; # + data.tar.gz from their install.sh -> deept_benchmarks/data
uv pip install --python alpha-beta-CROWN/.venv/bin/python pytorch-pretrained-bert
NNs/transformer_rewrite/run_deept_learn.sh sst_bert_small_3 NNs/transformer_rewrite/gauges/deept_small3_seed0.pt --max_len 10 --n_sent 40 --pos_per_sent 3 --steps 120 --accum 4 --seed 0
NNs/transformer_rewrite/run_deept_cmd.sh eval --name sst_bert_small_3 --gauge gauges/deept_small3_seed0.pt --max_len 12 --n_sent 40 --seed 0 --eps_list 0.01,0.02,0.03 --hi 0.1 --iters 10 --save_json results/deept_small3_eval_short_seed0.json
NNs/transformer_rewrite/run_deept_cmd.sh attrib --name sst_bert_small_3 --max_len 12 --n_sent 12 --pos_per_sent 2 --seed 3 --save_json results/deept_small3_attrib.json
```
