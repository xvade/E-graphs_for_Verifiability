# ViT attention-gauge rewrites (VNN-COMP 2023 `vit` benchmark)

Exact rewrites of the two competition ViTs (`vnncomp2023_benchmarks/benchmarks/vit/onnx/{pgd_2_3_16,ibp_3_3_8}.onnx`)
that tighten CROWN bounds without changing the function. Results and discussion: `PROGRESS.md`, section
"2026-09-04 (cont. 2)".

## The rewrite family
Per head and layer, for any invertible `G` (dh×dh):

    (X W_q + b_q)(X W_k + b_k)^T = (X W_q G + b_q G)(X W_k G^{-T} + b_k G^{-T})^T
    A (X W_v + b_v) W_o          = A (X W_v G + b_v G)(G^{-1} W_o)

Diagonal `G` is neutral for CROWN's bilinear relaxation; only mixing matters. `G` is either the closed-form
SVD-balanced gauge (`--variant R4_qk_svd / R5_av_svd / R45_both_svd`) or LEARNED (`vit_gauge_opt.py`: gradient
ascent on the lse-CROWN lower bound through the exact gauge algebra, tuned on ε-boxes around CIFAR-10 TRAIN
images, evaluated on the disjoint TEST-set benchmark instances). fp64 exactness gate vs the stock ONNX ≤ 1e-7;
fp32 storage of the rewritten weights differs from the stock ONNX by ~3e-6 (same class as stock vs onnxruntime).

## Files
- `vit_model.py` — PyTorch re-implementation of the ONNX ViT with the gauge hooks (`qk_gauge`, `av_gauge`) and `VARIANTS`.
- `vit_bounds.py` — auto_LiRPA CROWN/IBP bounds on the 100 benchmark instances (`--softmax lse|complex`, `--gauge_file`).
- `vit_gauge_opt.py` — learns per-head gauges (`--init svd|id`, `--obj mix`, cond penalty). Output: `gauges/*.pt`.
- `vit_compare.py` — paired per-instance comparison of two `results/*.npz`.
- `vit_export.py` — exports a rewritten model to ONNX (+ instances.csv) for the official abcrown pipeline.
- `vit_official_parse.py` — parses official abcrown logs (initial CROWN, alpha-CROWN #specs, verdict) and pairs two runs.
- `cfg_vit_*.yaml` — copies of the official `vit.yaml` settings pointed at stock / R45 / learned-G benchmark dirs.
- `run_chain.sh`, `run_chain2.sh` — the compute-node sequences (official runs must be ALONE on the GPU:
  `auto_enlarge_batch_size` sizes BaB batches from free memory, so a shared card contaminates the comparison).

## GenBaB ViTs (TACAS'25 `zhouxingshi/GenBaB`, downloaded to `genbab_benchmarks/`, gitignored) — no gauge leverage
- `genbab_download.py` — fetches `cifar/models` + `vit_1_3/1_6/2_3/2_6` (config.yaml, instances.csv, vnnlib specs) from the HF API.
- `genbab_gauge.py` — same harness as `vit_gauge_opt.py` for the GenBaB checkpoints (`{state_dict, optimizer, ...}` wrapper,
  Dropout stripped): `eval` (stock vs gauged vanilla CROWN on the benchmark boxes), `learn`, `export` (writes a gauged
  checkpoint + config copy into `genbab_benchmarks/cifar/<name>_<tag>/` for the official abcrown run).
- `genbab_smoke.py` / `run_genbab_smoke.sh` — CPU smoke: fp64 exactness gate with a random gauge, stock eval, 3 debug learner steps.
- `run_official_genbab.sh <config.yaml> <tag>` — unmodified abcrown with the GenBaB config (300 s), alone on the GPU. The stock
  `vit_2_3` run OOM'd at instance 25 (batch 50) and was not repeated because the diagnosis below made it moot.
- Result (PROGRESS.md 2026-09-05 cont. 3): on all four ViTs the softmax interval width over the ε=1/255 boxes is exactly 0
  (attention exactly uniform, or saturated one-hot on `vit_1_6`), so attention is a constant linear map on the spec and any
  gauge is bound-neutral (random-gauge Δ ≤ 5e-5). The general rule — gauge leverage ≈ attention nonlinearities' share of
  the CROWN width — and the DeepT transfer are in `NNs/transformer_rewrite/`.

## Fork modification (alpha-beta-CROWN is git-ignored here, so the delta is recorded in this file)
`alpha-beta-CROWN/complete_verifier/auto_LiRPA/operators/softmax.py`, `_softmax_lse_lower` and
`_softmax_lse_upper`: the chordal-slope formulas use `torch.where(diff > 1e-5, num/diff, fallback)`. The
unselected branch is 0/0 when `diff == 0`; the forward is masked correctly but the backward yields NaN
(0·NaN). Fix: divide by `safe_diff = torch.where(diff > 1e-5, diff, 1)` (and `safe_lpd` for
`log_p_diff` in the upper bound). Forward values are bit-identical; only gradients change. Needed to
differentiate the lse-CROWN bound w.r.t. the gauges in `vit_gauge_opt.py`. Not needed for verification runs.

## Reproduce (compute node)
    PY=alpha-beta-CROWN/.venv/bin/python
    $PY NNs/vit_rewrite/vit_bounds.py --model pgd_2_3_16 --variant base --softmax lse --methods CROWN --instances all --width 1 --mc 100
    $PY NNs/vit_rewrite/vit_gauge_opt.py --model pgd_2_3_16 --steps 400 --batch 32 --n_train 512 --lr 0.01 --init svd --obj mix --softmax lse --seed 0 --out NNs/vit_rewrite/gauges/pgd_mix_svdinit.pt
    $PY NNs/vit_rewrite/vit_bounds.py --model pgd_2_3_16 --variant base --gauge_file NNs/vit_rewrite/gauges/pgd_mix_svdinit.pt --softmax lse --methods CROWN --instances all --width 1 --mc 100
    $PY NNs/vit_rewrite/vit_compare.py NNs/vit_rewrite/results/pgd_2_3_16__base__lse.npz NNs/vit_rewrite/results/pgd_2_3_16__base__lse__G_pgd_mix_svdinit.npz
