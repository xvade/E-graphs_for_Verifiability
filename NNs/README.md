# `NNs/` — the verifiability pipeline

This directory is the concrete experiment code for the project: build small
neural nets, run them through TASO → tensat equality-saturation rewriting, pull
rewritten forms back out to ONNX with **real weights**, and measure how the
certified bound alpha-beta-CROWN computes changes before vs. after rewriting.

**New here? Read in this order:**
1. Top-level [`../README.md`](../README.md) — what the project is, how to
   recreate the excluded build artifacts (`tensat.sif`, `alpha-beta-CROWN/`).
2. [`../PROGRESS.md`](../PROGRESS.md) — chronological lab log (what was tried,
   what worked). The authoritative narrative.
3. [`../BUGS.md`](../BUGS.md) — bugs found in vanilla TASO/tensat and the fixes.
4. This file — a map of *which script does what*.
5. [`reassoc_results/`](reassoc_results/) — the lab notebook: per-experiment
   result write-ups (`*_RESULT.md`, `FINDINGS.md`, `REVERIFY_1097.md`, …). These
   are findings, not API docs; start from `FINDINGS.md`.
6. [`../PROBLEMATIC.md`](../PROBLEMATIC.md) — code that resists
   spec/test/documentation and is flagged for a possible later rewrite.

Every script also carries a header comment stating its own specific purpose and
the gotchas particular to it; this README is only the index — follow it to the
file, then read that header.

## Environment (read before running anything)

These landmines cost real debugging time; they are baked into
[`tests/run_tests.sh`](tests/run_tests.sh) so tests fail loudly instead of
pinning garbage:

- **Python for the rule-gen pipeline** (`pb2egg`, `prededup`, `z3_verify_egg`,
  anything importing `rules_pb2` or `z3`) is
  `toolchain-tensat/miniconda3/envs/taso_py/bin/python3` (3.10, protobuf 7.36,
  z3 5.1.0 with `z3pkg` on `PYTHONPATH`). The bare `miniconda3` python3.14 and
  `/usr/bin/python3` both fail to import — see `../PROBLEMATIC.md`.
- **`taso` / reconstruct scripts** run inside the Apptainer container
  (`apptainer exec tensat.sif …`) *or* against the CPU-only `taso/build` .so
  with `LD_LIBRARY_PATH` pointing at `taso/build` + the conda libs. The GPU
  `taso/build_gpu` .so still dies with `Cuda failure 35` (see `../PROBLEMATIC.md`).
- **The generator binary** needs `LD_LIBRARY_PATH=…/miniconda3/lib`
  (`libprotobuf.so.32`); `env -i` strips it and a *stale* `graph_subst.pb` gets
  copied silently.
- **Generator env flags are presence-checked** (`getenv("X") != nullptr`), so
  `GEN_COMMUTE=0` is still ON. Toggle by *unsetting*. See
  [`../taso/src/generator/README.md`](../taso/src/generator/README.md).

---

## 1. Model construction & training

Small nets chosen so a specific rewrite family has something to bite on.

| Script | Model | Notes |
|---|---|---|
| `build_maxout.py` | maxout net | The headline win: min/max reassociation moves the bound +2.37. |
| `build_lattice.py`, `build_tll_lattice.py` | lattice / two-level-lattice (TLL) | PWL nets; moved only with hand-added AC rules. |
| `inception_mnist_model.py` | InceptionMNIST (PyTorch `nn.Module`) | Genuine stride-1 parallel branch — matches tensat's `PRE_DEFINED_MULTI` conv-fusion rule. |
| `train_inception_mnist.py`, `train_inception_mnist_fast.py` | trains the above on raw MNIST idx files | ⚠ nondeterministic/slow — see `../PROBLEMATIC.md`. |
| `build_inception_convfused.py` | hand-built conv-weight-fused InceptionMNIST | The structure the conv-fusion rule *fires* into the e-graph (vs. the relu-merge). |
| `build_sweep_manifest.py` | writes `sweep_manifest.json` | Input to `run_verification_sweep.py`. |

## 2. ONNX ↔ TASO conversion, normalization, export

| Script | Purpose |
|---|---|
| `convert_mnist.py`, `convert_mnist_cnn_a.py`, `convert_inception_mnist.py`, `convert_resnet2b.py` | `load_onnx` → `export_to_file` (`.taso`) for one specific model each. |
| `export_mnist_cnn_a.py`, `export_resnet2b.py`, `export_inception_mnist.py` | PyTorch/model-def → ONNX with real trained weights. |
| `normalize_for_taso.py` | Make a VNN-COMP ONNX ingestible by TASO's narrow importer (MatMul+Add→Gemm, drop no-op Flatten, onnx-simplify), then assert numeric identity. |

## 3. Rule-generation pipeline

The generator lives in the TASO fork
([`../taso/src/generator/`](../taso/src/generator/)); these are the
protobuf→egg→verified-rules stages that follow it. End-to-end drivers are in §6.

```
generator (taso) → graph_subst.pb → pb2egg.py → prededup.py → z3_verify_egg.py → tensat -m redundancy → core rules
```

| Script | Stage | Spec highlights |
|---|---|---|
| `pb2egg.py` | protobuf → egg rewrite rules | Full-op (conv/pool/concat/matmul), `--multi-out` saves multi-output rewrites. |
| `prededup.py` | syntactic alpha-equivalence dedup | Canonically renames vars in first-appearance order; **keeps** comm-vs-assoc distinct. |
| `z3_verify_egg.py` | Z3 soundness check per rule (2 lanes) | Lane 1: ew ops exact, conv/concat/matmul uninterpreted. Lane 2 (`tensor_axioms.py`) on lane-1 non-verifieds. VERIFIED if either lane proves it. |
| `tensor_axioms.py` | Z3 lane 2: TASO tensor axioms | Port of `taso/verify/verify.py`'s quantified conv/concat/matmul/pool axioms; proves the op-algebra rewrites lane 1 can't (35→104/116 on the tracked pb). |
| `sexpr_to_functional.py` | egg s-expr → functional form | Shared helper. |

## 4. Extraction & reconstruction (rewritten form → real-weight ONNX)

`reconstruct_generic.py` is the **current, general tool** — it walks any
tensat/TASO `.model` DAG and resolves weight identity from tensat's
`<model>.weight_names.json` provenance sidecar (no per-model hand-tracing). The
others are its **model-specific ancestors**, kept for reference (the user
deferred rewriting them); use `reconstruct_generic.py` for new work.

| Script | Status | Model |
|---|---|---|
| **`reconstruct_generic.py`** | **CURRENT** | any (weight_names.json sidecar) |
| `reconstruct_optimized.py` | superseded | mnist_tiny_mlp (shape-matched weights) |
| `reconstruct_fused_relu.py` | superseded | resnet2b (axis-0 relu-merge demo) |
| `reconstruct_inception_unfused.py` | superseded | InceptionMNIST baseline |
| `reconstruct_inception_fused.py` | superseded | InceptionMNIST fusion v1 (axis-0 — unverifiable, BUGS #11) |
| `reconstruct_inception_fused_v2.py` | superseded | InceptionMNIST fusion v2 (axis-1 relu-batch) |

Supporting:
| Script | Purpose |
|---|---|
| `structural_signature.py` | GPU-free `.model` parse → structural signature/features (dedup samples, correlation features). Shared parser. |
| `derive_weight_names_baseline.py` | One-time guid→real-name sidecar for a never-hand-traced model (shape + positional matching). |

## 5. Bounds & verification

| Script | Purpose |
|---|---|
| `bound_forms.py` | Bound a set of reconstructed forms with alpha-beta-CROWN / auto_LiRPA; report cert_ub per form. |
| `bound_maxout_forms.py`, `bound_one.py` | Maxout-specific / single-form variants. |
| `gen_leaf_intervals.py` | IBP leaf intervals for the `verif_cost` extraction (deterministic gap metric). |
| `gen_sensitivity.py` | Backward-CROWN sensitivity weights (`--sensitivity_file` for tensat verif_cost). |
| `envelope_maxout.py` | Maxout envelope helper. |
| `compute_reference*.py` | Reference `(input, output)` pairs for numeric round-trip checks. |
| `verify_reconstruction*.py` | Assert a reconstructed ONNX matches its reference output (onnxruntime, 1 thread). |
| `run_verification_sweep.py` | Batch driver: ab-CROWN once per (manifest entry, epsilon); resumable; full-stdout capture. |
| `run_vnncomp_baselines.py` | Baseline ab-CROWN on unmodified VNN-COMP models + their vnnlib specs. |
| `aggregate_sweep_results.py` | Join sweep results with structural features → summary write-up. |

## 6. End-to-end drivers (shell)

| Driver | What it runs |
|---|---|
| `run_rule_gen.sh` | Full depth-3 all-relaxed PWL rule-gen pipeline. |
| `run_rule_gen_fullop.sh` | Same, full-op (no `PWL_FOCUS`) → conv-inclusive corpus. |
| `run_rule_gen_commute.sh` | ⚠ original commute driver — **broken** (set `RELAX_SUBST=1` *and* `GEN_COMMUTE=1` → 2 GB protobuf blow-up). Kept as the cautionary record. |
| `run_rule_gen_commute_fixed.sh` | Corrected commute driver (`GEN_COMMUTE=1`, `RELAX_SUBST` unset). Rationale in `reassoc_results/GEN_COMMUTE_SUBST_PROBE.md`. |
| `verif_cost_reverify.sh`, `verif_run.sh`, `vc_control_632.sh`, `vc_recon.sh` | `verif_cost` (deterministic IBP-gap) extraction + reconstruct. The **right** extraction for a verifiability win (immune to the `--n_diverse` collapse). |
| `reverify_model.sh`, `reverify_batch.sh`, `reverify_cpu.sh` | Reverify one/all models with a rule set, end to end (GPU or CPU). |
| `recon_forms.sh`, `maxout_reconstruct.sh` | Reconstruct all tensat forms of a model to ONNX (lowering ewmax/ewmin→relu), recording tree depth. |
| `multi_rule_match_probe.sh`, `sample_egraph_structures.sh` | Diagnostics: which multi-pattern rules fire; sample distinct e-graph structures. |

## 7. Tests

`tests/run_tests.sh` — plain-assert harness (no pytest, 22 assertions), run in
the container:
```
apptainer exec --no-mount bind-paths tensat.sif bash NNs/tests/run_tests.sh
```
(`--no-mount bind-paths` skips the site apptainer.conf binds — e.g.
`/var/run/slurm` — that abort container creation on non-slurm nodes.)
See [`tests/README.md`](tests/README.md) for the test list and what each pins.
