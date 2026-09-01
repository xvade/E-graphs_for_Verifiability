# Problematic code — flagged for possible later rewrite

Aggregated here (rather than scattered) so it can serve as input to a rewrite
decision. Each entry is code or infrastructure that **resists documentation,
specification, or testing** in its current state. Nothing here is a promise to
fix — it is a catalog. Deep dives live in `BUGS.md`; this file is the short,
findable index of "don't trust / can't test this yet."

Legend: **[infra]** environment/build, **[behavior]** possibly-wrong runtime
behavior, **[coverage]** correct but currently untestable here.

---

## 1. [infra] GPU TASO — `Cuda failure 35` was a driver mismatch, NOT resolved-in-code but node-dependent
The GPU-built taso extension died at `ops_cudnn.cu:24` (cuDNN init) with
`Cuda failure 35`. **Diagnosed 2026-09-01:** CUDA error 35 is
`cudaErrorInsufficientDriver` (confirmed from the container's
`driver_types.h`) — "the installed NVIDIA driver is too old for the CUDA
runtime". The container is CUDA 12.4 (`tensat.def`), so the original failure was
simply a GPU node whose driver predated CUDA 12.4 support (~550.x), not a taso or
cuDNN bug.

**Verified on a modern node (slurm job on A40 `g3064`, driver 580.178.04):** in
the container, `cudaGetDeviceCount` returns `rc=0` (cudaSuccess), runtime 12040 /
driver 13000 — **CUDA 12.4 initializes fine; error 35 does not recur.** So the
GPU path is *not* categorically blocked; it just needs a node with a
sufficiently new driver (most current cluster GPU nodes: a40/a100/h200/l40).

**Still to confirm (separate, small):** an end-to-end `import taso` against
`build_gpu` on such a node — the probe's bonus attempt failed on a shell-quoting
bug (space in the repo path) and because the GPU cython ext is currently renamed
aside (`taso/python/core*.so.gpubak`), not on CUDA. Restoring/rebuilding the GPU
cython ext and re-running the import is the remaining check; the driver-mismatch
root cause is settled. Meanwhile the CPU build (`taso/build`) still serves
structural stages 1–2 and abcrown's venv serves stage 3. See `BUGS.md`.

## 2. [behavior] `--n_diverse` extraction collapse
The diverse sampler reports `"0 new enodes added"` after ~3 samples and returns
duplicates, collapsing to a single shallow depth — for both the 632 and 1097
rule sets, where the Aug-29 binary reached depths 10–18. **Do NOT write a test
asserting the current `--n_diverse` output is correct** — it is under suspicion.
Use `--verif_cost` (deterministic) for verifiability wins instead. Full history:
`NNs/reassoc_results/REVERIFY_1097.md`.

**Mechanism (current-binary repro, 2026-09-01, maxout + 632 `pwl_rules_ac.txt`,
CPU).** Sampled cost/used-set per sample: `2.4e-6 (136 enodes) → 1.02e8 (+90) →
1.76e8 (+53, total 279) → 1.76e8 (+0) → …`. After sample 2 the `used` set
plateaus at **279 distinct enodes** and every later sample re-picks the same
penalty-dominated (1.76e8) tree. So the collapse is **e-graph exhaustion**, not
a dead cost jitter: only ~279 distinct enodes are reachable from the root, and
`DiverseCost` (optimize.rs) only ever *penalizes* reuse (flat +1e6) — it never
rewards crossing a cost cliff (its own `ArchDiverseCost` doc comment documents
this ceiling). Once the reachable enodes are used up, there is nothing new to
extract.

**Regression status: narrowed, not settled.** The cost commits (`f52cc16` et al.)
are exonerated by mechanism — `Ewsub/Ewmax/Ewmin` carry real nonzero TASO
runtimes (optimize.rs:677), so the jitter is alive. All suspects date to
2026-08-29 (same day as the "good" binary); the leading suspects are the
**cycle-check commits `ddd6352` / `5e3e9e9`**, which change what saturates *into*
the e-graph — a thinner e-graph yields exactly this exhaustion. Settling it needs
the Aug-29 bisect build, blocked by #3 (offline cargo). The practical fix already
exists in-tree: `--verif_cost` and `ArchDiverseCost` both sidestep the ceiling.

## 3. [infra] `cargo test` / `cargo build` need network (offline sandbox)
The crate can't build in the research sandbox: cargo must resolve the crates.io
registry index and neither CARGO_HOME here has it cached offline —
`cargo build --offline --tests` fails with `no matching package named 'arrayvec'`
under both `toolchain-tensat/cargo` and `toolchain-tensat/cargo_container`.
(Caution: `cargo metadata --offline --no-deps` *succeeds* but is a false
positive — `--no-deps` skips dependency resolution; the full build still fails.)
This also blocks the #2 Aug-29 bisect. The tensat *modes* are covered instead by
CLI-integration tests against the prebuilt binary (`NNs/tests/run_tests.sh` tests
2/4/8). The Rust unit tests are expected to pass in the original networked
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
