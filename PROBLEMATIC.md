# Problematic code — flagged for possible later rewrite

Aggregated here (rather than scattered) so it can serve as input to a rewrite
decision. Each entry is code or infrastructure that **resists documentation,
specification, or testing** in its current state. Nothing here is a promise to
fix — it is a catalog. Deep dives live in `BUGS.md`; this file is the short,
findable index of "don't trust / can't test this yet."

Legend: **[infra]** environment/build, **[behavior]** possibly-wrong runtime
behavior, **[coverage]** correct but currently untestable here.

---

## 1. [RESOLVED] GPU TASO — `Cuda failure 35` was a too-old driver, not a bug
The GPU-built taso extension died at `ops_cudnn.cu:24` (cuDNN init) with
`Cuda failure 35`. **Diagnosed + verified resolved 2026-09-01.** CUDA error 35 is
`cudaErrorInsufficientDriver` (confirmed from the container's `driver_types.h`) —
"the installed NVIDIA driver is too old for the CUDA runtime". The container is
CUDA 12.4 (`tensat.def`), so the original failure was simply a GPU node whose
driver predated CUDA 12.4 support (~550.x), not a taso or cuDNN bug.

**Confirmed end-to-end on modern nodes** (in-container `--nv`, driver 580.178.04):
on both A40 (`g3064`) and **L40S (`g3124`)**, loaded the GPU cython ext directly
from `build_gpu` and ran `PyGraph().conv2d(...)` — `cudnnCreate` + a real cuDNN
conv kernel both succeeded (`GPU_TASO_OK`), the exact original failure site. Error
35 does not recur. (Method note: `import taso` also needs `onnx`, absent for the
container python3.14 — the probe bypassed `__init__.py` and loaded the `core` ext
by path; a full `import taso` additionally needs onnx installed. The GPU cython
ext lives at `taso/python/taso/core*.so.gpubak`, RPATH → `build_gpu`; the active
`.so` is RPATH → `build` (CPU), so the working CPU path is untouched.)

**Conclusion:** the GPU path is **not** blocked — it needs a node with a
driver new enough for CUDA 12.4 (current cluster a40/a100/h200/l40 nodes qualify;
driver 580 seen). To use it: run on such a node and swap the `.gpubak` ext in
(or rebuild the GPU cython ext), plus install `onnx` for the container python if
using the high-level `import taso` API. The CPU build (`taso/build`) still serves
structural stages 1–2 and abcrown's venv serves stage 3. See `BUGS.md`.

## 2. [RESOLVED — design limitation, not a regression] `--n_diverse` extraction collapse
The diverse sampler reports `"0 new enodes added"` after ~3 samples and returns
duplicates, collapsing to a single shallow depth. REVERIFY_1097 recorded an
Aug-29 run reaching depths 10–18 and suspected a code regression. **Bisected and
settled 2026-09-01: there is no regression.** Use `--verif_cost` (deterministic)
for verifiability wins. Full history: `NNs/reassoc_results/REVERIFY_1097.md`.

**Mechanism (maxout + 632 `pwl_rules_ac.txt`, CPU).** Sampled cost/used-set per
sample: `2.4e-6 (136 enodes) → 1.02e8 (+90) → 1.76e8 (+53, total 279) → 1.76e8
(+0) → …`. After sample 2 the `used` set plateaus at **279 distinct enodes** and
every later sample re-picks the same penalty-dominated tree. The collapse is
**e-graph exhaustion × `DiverseCost`'s design ceiling**: only ~279 distinct
enodes are reachable from the root, and `DiverseCost` (optimize.rs) only ever
*penalizes* reuse (flat +1e6) — it never rewards crossing a cost cliff (its own
`ArchDiverseCost` doc comment documents this ceiling). Once the reachable enodes
are used up, there is nothing new to extract.

**Why it is not a regression (four independent confirmations):**
1. **Bisect: `e9b139a` == HEAD, byte-for-byte.** Built the last Aug-29 commit
   (before all the VerifCost/redundancy/axiom work) and ran the identical
   invocation → same `136→226→279`-plateau, same costs. No commit in the window
   changed the behavior.
2. **The suspects were structurally impossible.** The cycle-check commits
   (`ddd6352`/`5e3e9e9`) *precede* the min/max language (`f52cc16`), so any binary
   able to run maxout at all already contained them. And `e9b139a` is where
   "end-to-end min/max" landed — the earliest maxout-capable state — so no earlier
   commit could have produced deep min/max reassociations either. The exoneration
   window is closed on both sides.
3. **Not a saturation-budget issue.** With a 6× larger budget (n_sec 300, 100
   iters, 2M nodes, ~17 s/extraction) the used-set still plateaus at 278. More
   saturation does not enlarge the reachable set.
4. **The historical inputs no longer reproduce.** `pwl_rules_ac.txt` is dated
   Aug 30 — it *postdates* the Aug-29 depths-10–18 run, so that run used a
   different rule set. The Aug-29 raw set `pwl_rules_egg.txt` also collapses (228
   plateau). REVERIFY conflated the verif_cost win's rules with the diverse run's;
   there was never a held-constant before/after.

**Fix (already in-tree):** `--verif_cost` (deterministic IBP-gap extraction) and
`ArchDiverseCost` (rewards rewrite-witness enodes, crossing the cliff) both
sidestep the `DiverseCost` ceiling. The remaining `DiverseCost` weakness is a
design limitation, safe to leave as-is unless diverse *structural* sampling is
needed again — in which case reward-based selection (à la `ArchDiverseCost`) is
the pattern, not the penalty-only `DiverseCost`.

## 3. [infra — workaround established] Building tensat needs a network step first
The crate can't build purely offline out of the box: cargo must resolve the
crates.io registry, and neither pre-existing CARGO_HOME has the full graph cached
(`cargo build --offline` fails on `arrayvec`). But **this node has network**
(git/https work), and the container does not — so the working split is:

1. **Host (networked), populate the cache:** put the rustup toolchain bin on PATH
   (`toolchain-tensat/rustup/toolchains/stable-*/bin`, so `rustc` isn't the
   default-less proxy), set a writable `CARGO_HOME`, run `cargo fetch` in
   `tensat/` (~700 MB, one time).
2. **Container (offline), build/test:** same `CARGO_HOME`, `cargo build --offline`
   / `cargo test --offline`. Two gotchas when building from a git worktree:
   symlink `../egg` for the path dependency, and set
   `LIBCLANG_PATH=/usr/lib/llvm-14/lib` for bindgen.

**Verified 2026-09-01:** with the fetched cache, `cargo test --offline --test
parse` → `test model_parser ... ok` (1 passed), and a full offline build of
`e9b139a` succeeded. So the Rust tests **are** runnable here via this route (this
unblocked the #2 bisect). `NNs/tests/run_tests.sh` (tests 2/4/8) remains the
faster CLI-integration coverage against the prebuilt binary.
(Caution: `cargo metadata --offline --no-deps` succeeds even without the cache —
a false positive, because `--no-deps` skips dependency resolution.)

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
