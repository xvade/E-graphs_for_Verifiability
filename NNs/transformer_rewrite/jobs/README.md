# Slurm job scripts (transformer gauge experiments)

Moved into the repo from the session's scratch directory on 2026-09-12. Every script sets
`REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"` — on another cluster `export REPO=<checkout>` once.
The `#SBATCH` partition / account / QoS / gres lines are Hyak's (gpu-l40s 48 GB non-preemptible; ckpt-all A100 80 GB
preemptible, hence `--requeue` + per-instance resume + skip-if-done in every script); override them on the command line,
e.g. `sbatch -p <part> -A <acct> --gres=gpu:h200:1 jobs/alpha_gauge_ckpt.sbatch ...`. Submit from `NNs/transformer_rewrite`
(the `-o logs/...` paths are relative to the submit directory). Logs of the Python runs go to `NNs/vit_rewrite/_scratch`
(gitignored) and every run appends START / DONE lines to `_scratch/official_sequence.log`.

| script | what |
|---|---|
| `alpha_single.sbatch <small6|big3|small3> <max_len> <eps_list> <tag>` | α-CROWN stock vs once-trained gauge |
| `alpha_gauge.sbatch` / `alpha_gauge_ckpt.sbatch <m> <max_len> <eps_list> <tag> <gauge.pt>` | α-CROWN with one given gauge, stock skipped (L40S / ckpt A100) |
| `alt_chain.sbatch smoke|full` | alternating α / gauge learner + α eval |
| `learn_init.sbatch <m> <label>` , `learn_lab*.sbatch <label>` | learners warm-started from the unified rule / per-class learners |
| `screen_init.sbatch`, `screen_lab*.sbatch` | held-out radius screens |
| `sgnopt.sbatch <m> <label>`, `sgnopt_var.sbatch <m> <label> <rot|c2|mix>` | sign-aware per-class manual rule (CPU) |
| `sign_flip*.sbatch` | diag(±1) / scale gauge tests, plain CROWN vs CROWN-Optimized |
| `probe_big3.sbatch`, `cost_only.sbatch` | α memory probe; cost-only signed screens |
| `unf.sbatch <tag> [deept_unfolded.py args]` | rigorous certificate transfer with G⁻¹ as interval parameters |
