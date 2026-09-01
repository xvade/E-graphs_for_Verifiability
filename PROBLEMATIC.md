# Problematic code — flagged for possible later rewrite

Aggregated here (rather than scattered) so it can serve as input to a rewrite
decision. Each entry is code or infrastructure that **resists documentation,
specification, or testing** in its current state. Nothing here is a promise to
fix — it is a catalog. Deep dives live in `BUGS.md`; this file is the short,
findable index of "don't trust / can't test this yet."

Legend: **[infra]** environment/build, **[behavior]** possibly-wrong runtime
behavior, **[coverage]** correct but currently untestable here.

---

## 1. [infra] GPU TASO — `Cuda failure 35` (`taso/build_gpu`)
The GPU-built taso Python extension dies at `ops_cudnn.cu:24` (cuDNN init,
`Cuda failure 35`) even when the bare CUDA runtime works. Blocks any GPU taso
path. **Workaround in use:** the CPU build (`taso/build`) for structural stages
1–2 (no real kernels needed), abcrown's own venv for stage 3. Can't be
regression-tested without a working cuDNN. See `BUGS.md`.

## 2. [behavior] `--n_diverse` extraction collapse
The diverse sampler reports `"0 new enodes added"` after ~3 samples and returns
duplicates, collapsing to a single shallow depth — for both the 632 and 1097
rule sets, where the Aug-29 binary reached depths 10–18. Suspected regression
between the Aug-29 build and the Aug-31 rebuild (suspects: the VerifCost / cost /
CheckApply commits). **Do NOT write a test asserting the current `--n_diverse`
output is correct** — it is the thing under suspicion. Use `--verif_cost`
(deterministic) for verifiability wins instead. Full analysis:
`NNs/reassoc_results/REVERIFY_1097.md`.

## 3. [infra] `cargo test` needs network (offline sandbox)
`tensat/tests/parse.rs` can't build in the research sandbox: cargo must resolve
the crates.io registry index and no CARGO_HOME here has it cached offline
(`no matching package named 'arrayvec'`). The tensat *modes* are covered instead
by CLI-integration tests against the prebuilt binary (`NNs/tests/run_tests.sh`
tests 2/4/8). The Rust unit tests are expected to pass in the original networked
container build; they are simply unverifiable from here.

## 4. [infra] Broken `cmake install` for taso
The CPU taso `.so` is rebuilt by hand (manual `g++`) because `cmake install` is
broken. Documented in `BUGS.md`; the manual recipe is the supported path.

## 5. [infra] Two Pythons that look importable but aren't
- Bare `miniconda3/bin/python3` (3.14): `"Could not find platform independent
  libraries"`, `"no codec search functions"`.
- `/usr/bin/python3` (3.6.8): no `google.protobuf`.
The **only** working interpreter for the rule-gen pipeline is
`miniconda3/envs/taso_py/bin/python3` (3.10). A bare `import taso` can also
"succeed" as an empty namespace package (resolving to the repo's `taso/`
directory) — `structural_signature.py` guards against this but its op-name
fallback is then unverified. See `NNs/README.md` (Environment) and the
`structural_signature.py` header.

## 6. [behavior] Transpose ONNX export round-trip is broken
`core.pyx::get_operator_attr('perm')` decodes the permutation as a plain base-N
digit sequence, but `transpose.cc`'s encoder doesn't round-trip through it —
every exported Transpose came back with an invalid perm (e.g. `[0,0]`) that ONNX
rejects. **Workaround:** the reconstruct scripts fold weight-derived transposes
in numpy directly and never use the graph Transpose export path. Documented in
`reconstruct_optimized.py`'s header. Not chased further given the workaround.

## 7. [coverage] Z3 conv-axioms missing → conv rewrites rejected at verification
`pb2egg.py` now emits conv/pool/concat rules, but `z3_verify_egg.py` treats
conv/pool/concat as **uninterpreted** (sound but conservative), so most conv
rewrites are REJECTED at verification. Keeping them needs op
linearity/distribution axioms (cf. `taso/verify/validate_axioms.py`). Until then
the full-op corpus parses but conv rules won't survive Z3. Tracked in
`NNs/tests/README.md` (Outstanding).

## 8. [coverage] pb2egg tier-2 ops still dropped
`transpose`/`reshape` (config-as-name-string), `enlarge` (pb kernel-based vs egg
ref-based — a semantic mismatch), and `split` (multi-output → the multi-pattern
lane) are still dropped by `pb2egg.py`. See its `operator_data` comments.

## 9. [coverage] Training scripts are nondeterministic / slow
`train_inception_mnist.py` (+ `_fast`) train on real MNIST — not seeded for
bit-reproducibility and too slow for a unit test. The **inference** round-trip
*is* pinned (seed-0 reference + `verify_reconstruction_*.py`); the training step
is not, by design.

## 10. [coverage] Stale reference artifacts
`converted_full*.txt`, the `Mdl` arity comments, and the hand-committed
`taso_rules.txt` are known-stale (different/older egg formats). Do not test
against them — `-m parse_check` (tensat) is the authoritative current-format
oracle instead (`NNs/tests/run_tests.sh` test 2). Noted here so no future test
pins the stale files.
