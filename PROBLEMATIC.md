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

## 6. [RESOLVED — was a Release-build uninitialized read] Transpose ONNX export perm
Exported Transpose ops came back with a garbage perm (e.g. `[0,0]`) that ONNX
rejects. **Root cause (commit `fb0b3db`, taso fork):** `Graph::get_operator_int_attr`
read the parameter via `assert(op.ptr->get_int_parameter(attr, &ret))` — and under
NDEBUG/Release the assert's argument is never evaluated, so `get_int_parameter`
was never called and `ret` stayed uninitialized. Every attribute read through that
path (Transpose perm, Conv/Pool strides/kernel/pads) came back as garbage **only in
Release builds** — exactly the config the GPU work used. `fb0b3db` calls
`get_int_parameter` unconditionally; the diff is the fix. (The decode arithmetic in
`core.pyx::get_operator_attr('perm')` was always correct.)

**Verified 2026-09-01:** the perm decode round-trips for 6/6 perms (2-D/3-D/4-D) in
the current CPU build, and `pb2egg._decode_perm` matches a hardcoded
`permutation_to_index` oracle (`NNs/tests/test_transpose_perm.py`, wired into
`run_tests.sh` Test 10). Caveats, stated honestly: (a) the CPU build is
assert-enabled, where the bug never manifested — so the round-trip corroborates the
decode but does not itself re-prove the Release fix; `fb0b3db`'s diff does. (b) the
full `export_onnx` → `onnx.checker` end-to-end is **not** re-verified here (onnx is
not importable in the container — a #5-adjacent issue). (c) the generator/optimizer
only ever emits the 2-D swap (`NUMDIM=2, PERM=2`), so the N-D path is verified
*correct* but not *exercised in production*.

**Workaround retained (by design):** `reconstruct_optimized.py` still folds
weight-derived transposes in numpy — it avoids a graph op for weight-only
transposes and is the simpler path, so it stays regardless of the fix. Its docstring
is updated to note the decode now round-trips.

## 7. [largely RESOLVED] Z3 conv/concat/matmul verification
`pb2egg.py` emits conv/pool/concat rules, but `z3_verify_egg.py` (lane 1) treats
them as **uninterpreted** — sound but conservative, so it proved only rules that
hold by congruence and REJECTED the op-algebra rewrites (conv linear in its
weight, conv/concat distribution, relu(conv)=conv+relu, matmul associativity…).

**Fixed 2026-09-01 by a second verifier lane.** `NNs/tensor_axioms.py` ports
TASO's own rule verifier — the quantified tensor axioms in
**`taso/verify/verify.py`** (not `validate_axioms.py`, which is only the
meta-checker that validates those axioms on small shapes) — to Python 3: tensors
are an uninterpreted sort, ops are Z3 functions, and the proven `axioms`/`lemmas`
are asserted. No shape inference (the axioms are shape-polymorphic).
`z3_verify_egg.py` runs lane 1 unchanged, then lane 2 on any rule lane 1 did not
verify; a rule is VERIFIED if **either** lane proves it (union of two sound
checks, monotone). Result on the tracked `graph_subst.pb`: **35 → 104 / 116
verified** (conv2d 8→26, concat 0→43), 0 regressions, 0 negative-canary
failures. Tests: `NNs/tests/test_z3_axioms.sh` (run with the `taso_py` env, since
z3 is not in the container python — #5).

**Residue (~12 rules, follow-up):** grouped-convolution and matmul/concat-fold
substitutions that TASO's *own* verifier also does not prove universally
(`verify.py` comments the grouped-conv axiom out as "wrong axiom — caught with
N=[1,3]" and blacklists such rules). Reaching them needs grouped-conv-aware
reasoning, not a missing basic axiom. Also still-open: rerunning the full 6k
`fullop` corpus through the two-lane verifier (this validated the method on the
116-rule tracked pb), and pool/split/enlarge remain dropped upstream at
`pb2egg` (#8), so their axioms (ported but present) are currently unexercised.

## 8. [partly resolved] pb2egg tier-2 ops
**Note first:** the *tracked* `graph_subst.pb` has **zero** single-output dirty
rules — its only tier-2 loss is 13 two-output split rules, already preserved in
`.multi.pb`. Tier-2 volume lives in the (untracked) fullop corpus: 20,972
single-output dirty rules — transpose 9,203, enlarge 8,239, const_* ~3,844,
reshape 0.

### ⚠ THE TENSAT APPLICATION GAP (const family + transpose resolved; pool/smul open)
Emitting a rule and *applying* it are different. tensat's apply path
(`rewrites.rs`, the tensor-building match) supports only a subset of ops and ends
in `other => { println!(...); todo!() }` — so a rule that uses any other op
**parses (`parse_check`) and Z3-verifies fine but PANICS tensat at application
time**. Confirmed empirically (2-iter saturation probes on `mnist_tiny_mlp`):

- **Apply-safe** (build without panic): `conv2d`, `concat` (binary), `matmul`,
  `ewadd/ewmul/ewsub/ewmax/ewmin`, `relu`, `sigmoid`, `tanh`, `enlarge`,
  `split/split_0/split_1`, `merge`, `transpose` (tier A, resolved 2026-09-01),
  and the whole const family **`Iewmul`/`Imatmul`/`Iconv`/`Cpool`**.
- **Apply-UNsafe** (still panic on apply): `smul`, `poolmax`, `poolavg`,
  `concat3/4/5`.

**pb2egg gates on this (RHS-only):** by default it emits a rule only if its
**RHS** is apply-safe — the LHS is *matched* against existing enodes, never built,
so it needs no apply-safety (this lets `poolavg(x) => conv2d(x, Cpool)` through).
`--emit-unapplicable` keeps the full parse-valid set for consumers that never
apply rules (Z3 corpus studies via `z3_verify_egg.py`). Guarded permanently by
`run_tests.sh` **Test 12** (apply-smoke: guaranteed-fire rules through a real
saturation — apply-safe ops must not panic, gated ops must). The tracked-pb
corpus is unaffected (its 116 rules are all apply-safe).

**Const family resolved — Iewmul/Imatmul/Iconv/Cpool (2026-09-01).** The
constant-tensor ops are `MagicConst` (`validate_axioms.py`): shape-polymorphic,
channel dims supplied by the *consuming* op, so a standalone const has no sound
bottom-up metadata — which is why the authors left `todo!()`. The fix avoids
materializing a tensor at all: a **`DataKind::Const` marker** + **consumer
resolution**. `make`/`apply` of a const emit a Const marker (its type tagged in
`val`/`name`); its **approved consumer** — `ewmul` for `Iewmul`, `matmul` for
`Imatmul`, `conv2d(1,1,SAME,NONE)` for `Iconv`, each an identity `== x` — detects
the Const child and returns the *other* operand's metadata directly, so no tensor
is built. A **central applier guard** declines any *non*-approved parent of a
Const child, and each consumer arm resolves only *its* const (checking the `val`
tag) and declines a mismatched one or a non-identity conv config (soundness).
`get_self_cost` charges the identity wrappers a small positive cost so extraction
strictly prefers the bare `x` — so the const vanishes from extracted forms and
`reconstruct` is untouched.

**`Cpool` too (the non-identity one).** `conv2d(x, Cpool) == poolavg(x)`, *not*
`== x` — but every Cpool rule is stride-1 SAME-pad, so the poolavg output SHAPE
equals `x`'s; the conv2d consumer therefore returns `x`'s shape-metadata (sound
for shape-tracking; the value differs but is only ever unioned with `poolavg`, not
`x`). The Cpool marker packs its `(kh,kw)` in `val`. Its wrapper gets a **large**
`get_self_cost` (1e6) — since `conv2d(x,Cpool)` isn't equal to a cheaper operand,
this forces extraction to pick the equivalent `poolavg` enode, so `reconstruct`
never sees a Cpool. (tensat `model.rs`/`rewrites.rs`/`optimize.rs`;
`tensat/MODIFICATIONS.md`.) **All four consts** emit by default and pass the
apply-smoke (`run_tests.sh` Test 12).

**Gate relaxed to RHS-only.** egg matches the LHS against existing enodes (never
builds it), so pb2egg now gates on the **RHS** only — which is what lets
`poolavg(x) => conv2d(x, Cpool)` through (poolavg on the LHS is merely matched).

**transpose — DONE (tier A, 2026-09-01).** Added the `rewrites.rs` apply arm: the
perm is read from the `perm_name` `Var` child via the egraph (`results[i].1` is its
egraph Id — `TData` has no `name`), and `shuffle` is forced `true` (it's
value-invariant, #6, and taso's `Transpose` ctor asserts it — `make()` was fixed
to force it too, else a rule-emitted `shuffle 0` panics). **Un-gates the 9,093
transpose rules.**

**Remaining (open):**
- **`poolmax`/`poolavg`/`smul` (tier A):** also need `rewrites.rs` apply arms, but
  pool's is less trivial — `get_or_create_pool2d` takes a kernel *weight* tensor
  (the Cpool `poolavg` shortcut doesn't help: these are matched/built as real pools,
  not synthesized from a conv). `smul` needs its scalar-mul API. Smaller rule counts.
See `docs/ADD_AN_OP.md` for the full op-addition contract.

**Process lesson:** the transpose and const increments below shipped rules that
panic tensat on application, because every test (emission counts, `parse_check`,
the two Z3 lanes) ran *before* application. Test 12 closes that class.

### transpose — emission + verification + application all done
`pb2egg.py` decodes `PM_PERM` (inverse of `transpose.cc::permutation_to_index`,
validated to be a real permutation) and emits `(transpose input perm_name
shuffle)` — the `Name`-leaf form tensat parses. **9,093** single-output rules use
only transpose. Both verifier lanes learned it (they previously *errored*, so it
never reached lane 2): perm folded into the op identity (lane 1 congruence), lane
2 maps the 2-D swap to `transpose_0`, other perms to their own uninterpreted
function (`transpose_0`'s 2-D-only axioms can't misfire — guarded by a
non-involutive `1_2_0` canary). Application resolved (tier A, above), so these
rules now **emit by default** (18/20 of the fixture; the 2 with an apply-unsafe
`smul` on the RHS stay gated). Tests: `run_tests.sh` Test 9 + Test 12
(apply-smoke), `test_z3_axioms.sh` (involution flip + 3-cycle canary).

**Shuffle invariance (found while verifying the transpose fixture).**
`transpose.cc:102` shows the `shuffle` flag changes only output *strides*, never
values, so `transpose(x,perm,0)` and `transpose(x,perm,1)` are value-identical
(TASO's `transpose_0` correctly has no shuffle param). Both lanes now key on
`perm` alone and ignore shuffle (sound), which made the fixture's shuffle-only
rewrites trivial identities: verification went 2/20 → **20/20**. This was a
missing-axiom gap, *not* a quantifier ceiling (the earlier characterization was
wrong; guarded by 2 shuffle-invariance asserts).

### const_* — emission + verification done; application-blocked (gated)
The four constant-tensor ops emit, mapping the pb enum to tensat's egg names
(confirmed by `parse_check`): `const_pool → (Cpool kh kw)`, `const_iconv →
(Iconv kh kw)`, `const_imm → Imatmul`, `const_one → Iewmul`. **3,640**
single-output rules use only const_*. Lane 2 maps them to the ported axiom
functions, so `matmul(x,Imatmul)=x`, `ewmul(x,Iewmul)=x`, `conv(x,Iconv)=x`,
`conv(x,Cpool)=poolavg` discharge (fixture verifies **24/24**). **But the const
ops are apply-UNsafe, so these rules are gated by default** (emit under
`--emit-unapplicable`). Tests: `run_tests.sh` Test 11 (gated by default; 24/24 +
parse_check under the flag), `test_z3_axioms.sh` (2 flips + 2 false canaries
guarding the identity-matrix-vs-all-ones confusion).

### Still un-emittable / deferred (with reasons)
- **enlarge (~8,239):** egg accepts only the ref-based `Enlarge([Id;2])` (kernel
  `(enlarge 3 3 x)` fails parse_check); the generator serialized enlarge
  kernel-based, dropping the ref tensor, and no bound tensor of the target size
  exists in the pattern. The one bridge — `(enlarge ?w (Iconv 3 3))`, using Iconv
  as a ground-term size-carrier — *parses* and the `Enlarge` ctor accepts it
  (takes only w2's spatial dims, no channel assert), **but it needs an Iconv
  enode, which is apply-UNsafe** (the application gap above). So enlarge is
  blocked behind the same core change. `enlarge` itself *is* apply-safe; only its
  ref is not.
- **reshape:** 0 occurrences in every corpus seen; no code written.
- **split / multi-output:** its own feature (tensat's multi-pattern lane); the
  rules are preserved in `.multi.pb`, not lost.

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
