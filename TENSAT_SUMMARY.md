# TENSAT — Paper Summary

Source: Yang, Phothilimthana, Wang, Willsey, Roy, Pienaar. "Equality Saturation
for Tensor Graph Superoptimization." MLSys 2021. [arXiv:2101.01332](https://arxiv.org/abs/2101.01332)

Local copy of PDF: `/private/tmp/.../scratchpad/tensat_paper.pdf` (session scratchpad,
not persisted — re-download from arXiv if needed later).

## 1. Problem it solves

Tensor computation graphs (the DAGs deep learning frameworks build for a model,
e.g. matmul/conv/relu/concat nodes) are optimized by rewriting: swap in
semantically-equivalent subgraphs that run faster. Two questions arise for any
rewrite system:

1. **Which rules are sound?** (verification)
2. **In what order do you apply them?** (optimization / search)

The prior state of the art, **TASO** (SOSP '19), auto-generates candidate
rewrite rules and *verifies* them with an SMT-style checker, then optimizes by
a **backtracking search** that applies rules sequentially. Sequential
application is sensitive to *order* — applying a locally-good rewrite can hide
a better rewrite two steps later (classic "phase-ordering" problem), and
backtracking search only explores a small slice of the exponential space of
equivalent graphs.

TENSAT replaces TASO's sequential search with **equality saturation**, applying
*all* rewrites "at once" via a shared data structure, and separately verifies
rules the same overlapping way. TENSAT reuses TASO's rewrite rules, TASO's cost
model (measured per-op runtime on real hardware), and TASO's C++/CUDA runtime
for execution — it is explicitly a *re-implementation of TASO's search/verify
layer*, not a new DL framework.

## 2. Background: e-graphs & equality saturation

- An **e-graph** encodes an equivalence relation over terms. It's a set of
  **e-classes** (equivalence classes), each holding a set of equivalent
  **e-nodes** (an operator + list of child e-classes, i.e. pointers to
  e-classes not literal subterms).
- Applying a rewrite `l → r` **adds** `r[σ]` into the e-class matched by `l`
  instead of destructively replacing anything — nothing is ever thrown away.
  This is what avoids the phase-ordering problem: bad-looking rewrites don't
  block later good ones because the original term is still there too.
- **Equality saturation** = keep applying every rule to every match until no
  new info is added (*saturation*) or a budget (time/size/iterations) is hit.
- Two decoupled phases:
  - **Exploration**: grow the e-graph by rule application (build phase).
  - **Extraction**: pick one e-node per e-class along the (single) root to
    reconstruct the lowest-cost concrete term/graph.
- Because e-graphs share structure, they can represent an exponential set of
  equivalent terms compactly, and can even prove equalities that ordered
  directed rewriting cannot (e.g. combining `f(a,b)→c` and `a→b` proves
  `f(a,b)=f(b,a)=f(b,b)` without ever having a rule that fires in the right
  order for directed rewriting).

## 3. TENSAT's representation extensions

Building an equality-saturation engine for tensor graphs (as opposed to scalar
expressions) required real extensions — this is the paper's technical core.

### 3.1 Graph representation
- Each op becomes a node `nᵢ`; the operator's *output tensor* is what the node
  represents. Table 2 in the paper lists the supported op set: `ewadd`,
  `ewmul`, `matmul`, `conv` (grouped conv covers normal/depthwise as special
  cases), `relu`/`tanh`/`sigmoid`, `poolmax`/`poolavg`, `transpose`, `enlarge`
  (pad conv kernel), `concat_n`, `split`/`split0`/`split1`, `merge` (grouped
  conv weight merge), `reshape`, `input`, `weight`, and a synthetic `noop` used
  to make multi-output graphs single-rooted (noop has no real semantics/cost).
- Types: Tensor (T), String (S, for things like axis-permutation strings),
  Integer (N, for stride/pad/activation-mode enums), Tensor-tuple (TT, for
  split's two outputs).

### 3.2 Rewrite rule representation
- A rule = source pattern → target pattern, both S-expressions with variable
  placeholders, with matched-output pairs stating which output tensors
  correspond across the two patterns.
- **Single-pattern rules**: source pattern has one output — straightforward,
  reuses standard e-graph pattern search (egg's built-in matcher).
- **Multi-pattern rules**: source/target patterns can have *multiple* outputs
  (e.g. "two matmuls sharing an input" — Figure 2 in the paper: fusing
  `matmul(w1,x)` and `matmul(w2,x)` into one `matmul` over a `concat`ed weight,
  then `split` back apart). Most e-graph libraries (including egg) only give
  you efficient search for single-pattern rules, so **TENSAT introduces its
  own multi-pattern matching algorithm** (Algorithm 1 in the paper):
  canonicalize each source-pattern sub-S-expr (map to a canonical form under
  variable renaming), run the single-pattern searcher per canonical
  sub-pattern, then take the Cartesian product of matches across a rule's
  sub-patterns and keep only combinations that agree on shared variables
  (`COMPATIBLE` check), then decanonicalize back to the rule's real variables
  before applying.
- **Danger**: multi-pattern rules can blow up the e-graph combinatorially —
  e.g. the "N matmuls sharing an input" pattern creates O(N²) new matmul
  nodes in one pass, and matching pairs-of-those next iteration creates
  O(N⁴), i.e. **double-exponential growth**. TENSAT therefore caps multi-pattern
  application to a separate, small iteration limit `k_multi` (default 1 in
  their experiments) and only runs single-pattern rules to full saturation
  after that.
- **Shape checking**: many rewrites are only valid if certain tensor shapes
  match (a syntactic-match precondition isn't enough). TENSAT verifies shape
  compatibility of a matched target pattern before applying, using an egg
  **e-class analysis** that carries shape/layout/split-location metadata
  alongside each e-class (same approach TASO uses).
- Graphs are made **single-rooted** by combining all real outputs under
  `noop` nodes, since equality saturation as formulated wants one root.

## 4. Extraction extensions (the other technical core)

Given a saturated (or budget-truncated) e-graph, you must pick one e-node per
e-class to reconstruct a concrete, **acyclic** graph of lowest total cost.

- **Cost model**: same as TASO — sum of each chosen op-node's independent,
  hardware-measured runtime cost (suits GPUs, which execute one op at a time;
  an "op" may itself be a fused primitive, e.g. fused conv+ReLU).
- **Greedy extraction**: per e-class, pick the e-node whose subtree has
  smallest total cost, computed bottom-up. Cheap but **not optimal** — ignores
  sharing (if two children reference the same subgraph, greedy double-counts
  it) and can miss globally-better choices when nodes are shared across
  outputs. The paper's Table 4 shows this materially loses to ILP on
  BERT/NasNet-A (though it's fine on NasRNN).
- **ILP extraction**: formulate extraction as an integer linear program.
  Binary var `xᵢ` = 1 iff e-node `i` is picked; constraints force exactly one
  node picked in the root e-class, force at least one child-e-class node
  picked whenever a parent that references it is picked, and objective
  minimizes total picked cost. This alone gives the **globally optimal**
  extraction w.r.t. the cost model — but naively it can select **cyclic**
  e-node choices (an e-graph can have a node in e-class A referencing e-class
  B whose chosen node references back to A), which doesn't correspond to a
  valid DAG. Figure 3 shows a concrete example of a valid rewrite creating
  exactly this cycle risk.
- **Cycle avoidance, two ways** — this is the paper's main scalability
  contribution for extraction:
  1. Add topological-order variables `t_m` per e-class to the ILP with
     big-M constraints forcing a valid topo order among picked nodes
     (real- or integer-valued `t_m`, both explored). Works but the ILP
     solver (SCIP via OR-Tools) gets *very* slow as e-graph size grows —
     cycle constraints are the main bottleneck, sometimes 10–1000x slower
     than without them, and can time out entirely (Table 5).
  2. Instead, **filter cycles during exploration** so the final e-graph is
     guaranteed acyclic, letting you drop the cycle constraints from ILP
     entirely. Two variants (Algorithm 2):
     - *Vanilla cycle filtering*: before applying each candidate
       substitution, check (full e-graph pass) whether it would introduce a
       cycle; skip if so. Correct but O(n_m · N) per iteration — slow,
       since the number of matches n_m scales with e-graph size N.
     - *Efficient cycle filtering*: precompute a descendants map once per
       iteration (one pass), use it as a fast pre-filter (sound but
       incomplete — misses cycles created *within* the same iteration by
       other rewrites), then do a cheap DFS-based post-processing pass at
       iteration end to catch and resolve any cycles that slipped through
       (drop the most-recently-added node in each detected cycle, add it to
       a filter list excluded from extraction). This is empirically up to
       ~2000x faster than vanilla filtering (Table 6) and is what TENSAT
       uses by default.

## 5. Evaluation highlights (context for what "success" looks like)

- Benchmarks: BERT, ResNeXt-50, NasNet-A, NasRNN, SqueezeNet, VGG-19,
  Inception-v3 — run on GCP with an NVIDIA T4 GPU, cost measured via TASO's
  cuDNN backend.
- Default settings: e-graph size cap `N_max = 50000` nodes, exploration
  iteration cap `k_max = 15`, `k_multi = 1`, ILP solver (SCIP) timeout 1 hour,
  full approach = efficient cycle filtering + ILP-without-cycle-constraints.
- Result: up to 16% additional runtime speedup over TASO's already-optimized
  output (up to 68.9% vs. the unoptimized graph), while optimizing **on
  average 48x faster** than TASO (up to ~300x), because the e-graph explores
  a much larger equivalent-graph space than backtracking search in less wall
  time.
- `k_multi` is a real knob with a real tradeoff: increasing it lets TENSAT
  find better rewrites on most benchmarks but e-graph size grows
  double-exponentially and the ILP solver can time out (`k_multi=3` times out
  at 1hr for several benchmarks); on SqueezeNet increasing it actually
  *hurts* speedup due to cost-model/real-runtime discrepancy on the rewrites
  it unlocks.
- Appendix gives concrete rewrite patterns that mattered in practice: fusing
  parallel matmuls/convs that share an input via concat+split (BERT,
  NasNet-A, Inception-v3, NasRNN), and folding parallel conv+add into a single
  conv when the extra concat operands are weight-only (so precomputable at
  compile time) (NasNet-A).

## 6. Implications for our project (verifiability angle)

- The paper's own "verifier" (separate from the optimizer, reusing the same
  e-graph machinery) is explicitly flagged in the codebase README as "complete,"
  while the optimizer is flagged "in progress" — worth checking current state
  before assuming feature-parity with the paper.
- Equality saturation gives *stronger* equivalence proofs than sequential
  rewriting (the `f(a,b)=f(b,a)=f(b,b)` example) — this is directly relevant
  to "verifiability": an e-graph-based verifier can prove a rewrite rule sound
  by composing smaller, more obviously-sound rules, without needing a
  hand-tuned rule ordering, which is likely the crux of why this project cares
  about TENSAT.
- Extraction correctness (cycle-freedom of the extracted graph) is *load-bearing*
  for TENSAT's own correctness, not just performance — good to understand if we
  intend to modify extraction or the cost model.
- The cost model is a straightforward sum of independent per-op measured
  costs; if our project changes the cost model (e.g. to something
  verification-cost-aware rather than runtime-aware) that's a fairly
  contained change (Section 5.1's ILP objective + greedy extraction), but the
  cycle-filtering exploration-phase code and shape-checking e-class analysis
  should be unaffected.

## 7. Codebase notes (uwplse/tensat @ this repo's fork, `xvade/tensat`)

Repo layout (`tensat/`):
- `src/main.rs`, `src/optimize.rs` — CLI entry / optimizer driver (toggle
  between `prove_taso_rules()` verifier mode and `optimize()` optimizer mode
  by commenting/uncommenting in `main()`).
- `src/model.rs`, `src/rewrites.rs`, `src/input.rs`, `src/parse.rs`,
  `src/equation.pest` — graph/rule representation, rule set loading, and the
  S-expression parser (pest grammar) described in §3 above.
- `src/{bert,inceptionv3,mobilenetv2,nasneta,nasrnn,resnet50,resnext50,squeezenet,vgg}.rs`
  — per-benchmark graph construction, matching the paper's eval set.
- `taso_rules.txt`, `single_rules.txt`, `multi_rules.txt`,
  `multi_cleaned.txt`, `converted*.txt`, `op_table.txt` — the actual rewrite
  rule sets (TASO-derived) and op tables referenced in §3.2/§6.2.
- `model/*.model` — saved pretrained weights/graphs for some benchmarks.
- `extractor/`, `analysis/` — ILP extraction glue and the `analysis/stats.py`
  plotting/analysis script mentioned in the README.
- `run_exp_main.sh`, `efficient_vs_vanilla.sh`, `greedy_vs_ilp.sh`,
  `real_vs_int.sh`, `save_models.sh` — scripts reproducing the paper's
  Tables/Figures (cycle-filtering ablation, greedy-vs-ILP ablation, etc.).

**Build requirements — this is the part likely to bite us:**
- TENSAT is Rust (`Cargo.toml`), built via `build.rs` using `bindgen` to
  generate C++ FFI bindings directly against **TASO's** C++ headers/runtime
  (`taso::Graph`, `taso::Tensor`) and links against `libtaso_runtime` +
  `libprotobuf`. So we need a **built TASO install** (with its CUDA/cuDNN
  runtime) present before `cargo build` will succeed — this is not optional
  even just to run the verifier's rule *logic*, since the crate always links
  taso_runtime (README says GPU is only needed if you want to *run* the
  optimizer, not build it, but linking still requires TASO's shared lib to
  exist on the system).
- `Cargo.toml` depends on `egg` via a **local path** (`path = "../egg"`), and
  the README says this must be the **forked** egg
  (`https://github.com/yycdavid/egg`), not upstream egg — upstream may lack
  the hooks TENSAT's multi-pattern matcher / e-class analysis rely on. Also
  needs the **forked TASO** (`https://github.com/yycdavid/taso`), not
  upstream TASO.
- Recommended layout: `tensat/`, `egg/` (forked), and TASO all cloned as
  sibling directories, then use the provided `docker/Dockerfile` /
  `docker/run_docker.sh` (base: `nvidia/cuda:10.0-devel-ubuntu16.04`, cuDNN
  7.6.0, Miniconda w/ protobuf 3.9 + onnx + cython, Rust via rustup,
  llvm-dev/libclang-dev/clang for bindgen, Google OR-Tools/SCIP via
  `pip install ortools` for ILP extraction). Building TASO itself still needs
  a manual `cmake && make install` + `python setup.py install` step inside
  the container per the tensat README.
- Practical implication: **CUDA 10.0 / cuDNN 7.6 / Ubuntu 16.04-era toolchain**
  is quite old (2019-2020 vintage) — expect friction getting this running on
  modern hardware/drivers or Apple Silicon (no CUDA at all on macOS), so
  running via the Docker image on a Linux box with an NVIDIA GPU (or a cloud
  GPU instance) is probably mandatory rather than optional, at least for the
  optimizer. The *verifier* path claims to not need a GPU, but still needs
  the TASO runtime *library* built and linked, so a full TASO source build is
  likely unavoidable either way given how `build.rs` is wired.
- `wrapper.h` is the bindgen entry header — worth checking early since it
  pins the exact TASO surface tensat depends on; any TASO fork API drift here
  is the most likely build breakage point.

## 8. Project scope (resolved 2026-08-22)

- **Goal**: measure how TENSAT optimization changes network *verifiability*,
  where verifiability = tightness of the bounds `alpha-beta-CROWN` finds
  given a fixed **1-minute** compute budget per query. Pipeline is roughly:
  take a network → run alpha-beta-CROWN (60s) → record bound tightness →
  run TENSAT's optimizer on the same network → run alpha-beta-CROWN on the
  optimized graph (60s) → compare bound tightness. This means we mainly need
  TENSAT's **optimizer** path (equality saturation + extraction) to produce
  optimized graphs, not the separate verifier-for-rewrite-rules path — those
  are different things despite both being called "verif*" in this project
  (TENSAT's rule verifier checks rewrite *soundness*; our verifiability
  metric is about alpha-beta-CROWN's *output bound tightness* on a given
  concrete graph).
- **Forked deps**: cloned as siblings of `tensat/` under this project root —
  `egg/` (`yycdavid/egg`) and `taso/` (`yycdavid/taso`). Both present as of
  2026-08-22.
- **CUDA/GPU**: deliberately deferred on the Mac dev machine, not skipped —
  see §9 below for the now-resolved plan to bring it up on a GPU cluster.

## 9. Session state as of 2026-08-22 (read this first on a new machine)

Everything below happened on a MacBook (arm64, no GPU) and is about to move
to a Rocky Linux 8.10 HPC cluster with 2x NVIDIA L40S. **This section is the
authoritative, current state — memory notes from that session live in
Claude's local `~/.claude/projects/...` on the Mac and do NOT transfer to a
new machine or a fresh Claude Code session; this file is what does.**

### 9.1 What's built and validated (CPU-only, on the Mac)

TASO's `yycdavid/taso` fork has **no CPU/DNNL backend at all** (unlike
upstream `jiazhihao/TASO`, which added one later) — its `CMakeLists.txt`
declares `project(TASO LANGUAGES CXX CUDA)` unconditionally and every
`measure_*_cost`/`forward()`/`map()`/`unmap()` for every op except
`NoOp`/`Split` is implemented only in `src/cudnn/*.cu`. Rather than require
a GPU just to get TENSAT's optimizer running, we built a real (not fake)
CPU-only path:

- **`taso/CMakeLists.txt`**: now reads `USE_CUDA` from `config.cmake`
  *before* calling `project()`, so CUDA is only requested as a language
  when actually enabled (`config.cmake` currently has `USE_CUDA OFF`).
  Also bumped `cxx_std_11` → `cxx_std_17` (needed for modern
  protobuf/abseil headers) and requires protobuf 3.21 specifically
  (`brew install protobuf@21` on macOS) — anything newer pulls in an
  abseil logging dependency TASO's CMake doesn't link against. **Flipping
  back to GPU is just `set(USE_CUDA ON)` in `config.cmake` plus a real
  CUDA/cuDNN toolchain — no other source changes needed.**
- **`taso/src/cpu/measure_cost_cpu.cc`** (new): synthetic cost model for
  the 22 CUDA-only `measure_*_cost` functions. Reuses TASO's own existing
  per-op FLOP/mem-traffic formulas (the same ones each op's
  `collect_costs()` in `src/core/*.cc` already computes, CUDA-independent)
  — `runtime = (flops + mem_acc) / UNITS_PER_MS`. Explicitly **not** real
  hardware measurement; only needs to rank rewrites in roughly the right
  relative order for extraction to have something to minimize.
- **`taso/src/cpu/ops_cpu.cc`** (new): CPU-only `Model::Model()`,
  `allocate_memory`/`copy_memory` (`malloc`/`memcpy` instead of
  `cudaMalloc`/`cudaMemcpy`), `measure_oplist_runtime` (sums already-set
  per-op `runtime`s instead of calling `forward()`).
- **`taso/src/cpu/execution_stubs_cpu.cc`** (new): C++ vtables need every
  virtual method resolved at *link* time even if never called at runtime —
  so `map()`/`unmap()`/`forward()` needed stub bodies for the 24 op
  classes that don't already have CPU-native ones (`NoOp`, `Split` already
  did). These stubs **fail loudly** (`fprintf` + `abort()`) if ever
  actually invoked, rather than silently returning wrong data — this is
  intentional and already caught one real thing (see `--no_runtime_report`
  below).
- **`tensat/build.rs`** / **`tensat/wrapper.h`**: originally hardcoded a
  specific Docker mount layout (`/usr/tensat`, `/usr/TASO`→`/usr/local`)
  that doesn't match a plain sibling checkout. Replaced with
  `TASO_LIB_DIR`/`TASO_INCLUDE_DIR`/`PROTOBUF_LIB_DIR` env vars (default
  `../taso/build`, `../taso/include`, no default — machine-specific) plus
  a real `-I` clang arg for bindgen instead of a fragile relative
  `#include "../include/..."`.
- **`tensat/Cargo.toml`**: bumped `bindgen` 0.54.0 → 0.70 (`build.rs`
  updated to match: `allowlist_type`, `CargoCallbacks::new()`) — 0.54
  can't parse anonymous structs in current libc++ headers.
- **`tensat/src/main.rs`**: added `--no_runtime_report` CLI flag. Without
  it, `optimize()` unconditionally calls `get_full_graph_runtime()` after
  extraction (real hardware runtime measurement, via
  `Graph::preprocess_weights()`/`Graph::run()`) — on the CPU build this
  hits the `execution_stubs_cpu.cc` abort (confirmed: `Concat::forward()`
  called from constant-folding in `preprocess_weights`). The flag skips
  those two calls and writes `null` (not a misleading `0.0`) into the
  `--out_file` JSON stats for `original`/`optimized` runtime. **Always
  pass this flag when running the CPU build.**
- **`tensat/.envrc`** (new, gitignored — machine-specific, do not copy to
  the cluster as-is): sets the four env vars above for local dev via
  direnv. `PROTOBUF_LIB_DIR` points at a Homebrew path
  (`/opt/homebrew/opt/protobuf@21/lib`) that won't exist on Linux — needs
  a fresh `.envrc` on the cluster pointing at wherever protobuf lives
  there (or, once containerized, this whole env-var dance may be
  unnecessary if the build happens entirely inside the Apptainer image).
- **`~/.zshrc`** (Mac-local, not part of the repo): rustup's installer
  failed to add `~/.cargo/bin` to `PATH` (it hit a permissions error on a
  root-owned `~/.bashrc` before reaching `~/.zshrc`) — fixed by manually
  appending `. "$HOME/.cargo/env"`.

**Validated working (Mac, CPU-only):**
```bash
cd tensat
cargo build                     # succeeds
cargo test                      # 1 test (tests/parse.rs), passes
./target/debug/tensat -r converted.txt -d nasrnn -e greedy \
    --n_iter 3 --n_sec 30 -s none --no_runtime_report \
    -x tmp/nasrnn                # -x works: writes tmp/nasrnn_{start,optimized}.model
```
Saturates (490 nodes, 40 equivalent programs found for nasrnn), extracts a
best-cost graph via greedy extraction, exits 0. `-x` output is confirmed
structurally valid (matches the format `tests/parse.rs` round-trips) and
the optimized graph is genuinely smaller than the start graph (1124 vs
1264 lines for nasrnn) — i.e. real rewriting is happening, not a no-op.

**Known limitation, not yet addressed:** the built-in benchmark graph
constructors (`tensat::nasrnn`, `bert`, etc., used via `-d <model>`) fill
every weight with **fresh uniform random noise**
(`tensat/src/model.rs` ~line 390: `(0..num_entries).map(|_| rand::random())`),
not trained weights. Fine for TASO/TENSAT's own runtime-speedup goals, but
means alpha-beta-CROWN bound-tightness numbers on `nasrnn_start.model` /
`nasrnn_optimized.model` as they stand would reflect random-weight
geometry, not anything about a trained network's verifiability — not
useful for [[project-goal]]. The real path: source a real trained model
(e.g. one of alpha-beta-CROWN's own `examples_abcrown/` nets), load it via
TASO's Python binding `taso.load_onnx()`, `graph.export_to_file()`, then
feed that file into `tensat` via `-f <file>` (already supported — the
`None =>` branch in `main.rs`'s model-selection `match`, same code path
`tests/parse.rs` exercises) instead of `-d nasrnn`. TASO's Python bindings
(`taso/python/setup.py`) aren't built yet — same class of fix needed there
as `CMakeLists.txt` got (hardcoded `-DUSE_CUDNN` flag).

### 9.2 Cluster migration — GPU build (completed 2026-08-22)

Done, on the actual cluster (Hyak Klone, `klone-login03` + compute node
`g3109`, `gpu-l40s` partition, 2x L40S). Key facts + what actually happened,
so no need to re-derive:

- `yycdavid/taso`'s `src/cudnn/*.cu` uses cuDNN's "Find" convolution API
  (`cudnnFindConvolutionForwardAlgorithmEx`), not the "Get" API that was
  removed in cuDNN 8 — porting from cuDNN 7 to cuDNN 9 needed **zero source
  changes**, confirmed by an actual GPU compile. This was flagged as "the
  one real technical unknown" going in; it wasn't one.
- Container: `tensat.def` (project root) — Ubuntu 22.04 + CUDA 12.4.1 devel
  base (`nvidia/cuda:12.4.1-devel-ubuntu22.04`), built with
  `apptainer build --fakeroot tensat.sif tensat.def`. `CMAKE_CUDA_ARCHITECTURES
  89` (Ada/L40S) lives in `taso/build_gpu/config.cmake` (a build-dir-local
  override — see below), not baked into the image.
- **Build gotchas actually hit** (all fixed, reflected in `tensat.def` and
  `taso/build_gpu/config.cmake` as they stand now):
  - Apptainer's OCI/build cache defaults to `~/.apptainer` — home dirs on
    this cluster have a small quota and it filled instantly. Fix:
    `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` must point at scratch space
    (`/mmfs1/gscratch/scrubbed/sgvtc/apptainer_{cache,tmp}`) for every
    `apptainer build`/`exec` call.
  - Anaconda's `defaults` channels (pkgs/main, pkgs/r) now require
    interactive Terms-of-Service acceptance, which silently breaks
    non-interactive `conda install` in a container `%post`. Fix:
    `conda config --system --remove channels defaults`, use conda-forge
    only.
  - conda-forge's top-level `protobuf` package (Python bindings) jumped
    from 3.20.x straight to 6.x/7.x — `protobuf=3.21` doesn't exist as a
    package there. The C++ library at that version lives under
    **`libprotobuf`** instead (`libprotobuf=3.21.12`). Also: conda-forge's
    current `onnx`/`numpy`/`cython` builds hard-pin a *matching modern*
    libprotobuf, so they're not installable alongside `libprotobuf=3.21.x`
    in one env — dropped from the container for now; only needed later for
    TASO's own Python bindings (loading real ONNX models), which will need
    its own separate env/pinning pass.
  - `apptainer exec` (no `--fakeroot`/`--writable-tmpfs`) runs the SIF
    read-only, so `apt-get update` fails inside a plain `exec` (fine inside
    `%post` during `build`, which *is* writable).
  - CMake ≥4.0 removed support for `cmake_minimum_required(VERSION 3.2)`
    (taso's declared minimum) entirely — needs
    `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` passed to every `cmake ..`
    invocation, container or not.
  - `find_package(Protobuf REQUIRED)` doesn't search `/opt/conda` by
    default — needs `-DCMAKE_PREFIX_PATH=/opt/conda
    -DPROTOBUF_LIBRARY=/opt/conda/lib/libprotobuf.so
    -DPROTOBUF_INCLUDE_DIR=/opt/conda/include` passed explicitly.
  - cuDNN 9's split into sub-libraries (`libcudnn_ops`, `libcudnn_cnn`,
    etc., flagged in the original migration plan as a linking risk) turned
    out to be a non-issue: `find_library(CUDA_CUDNN_LIBRARY cudnn ...)` in
    `taso/cmake/FindCUDA.cmake` has no `NO_DEFAULT_PATH`, so it falls back
    to normal system search paths and finds apt's
    `/usr/lib/x86_64-linux-gnu/libcudnn.so` (the cuDNN 9 frontend, which
    dlopens the sub-libraries itself at runtime) without any CMake changes.
  - `CARGO_HOME=/opt/cargo` (baked into the image at container-build time)
    is inside the read-only SIF at runtime — `cargo build` fails trying to
    write its registry cache there. Fix: override `CARGO_HOME` to a
    writable host path at `apptainer exec` time (e.g.
    `/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/cargo_container`);
    `/opt/cargo/bin/cargo`/`rustc` themselves are read fine from the image.
- **Validated working (GPU, `g3109`, 2x L40S, inside `tensat.sif` via
  `apptainer exec --nv`):**
  ```bash
  cd taso/build_gpu   # config.cmake here sets USE_CUDA=ON, ARCHITECTURES=89
  cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_PREFIX_PATH=/opt/conda \
      -DPROTOBUF_LIBRARY=/opt/conda/lib/libprotobuf.so \
      -DPROTOBUF_INCLUDE_DIR=/opt/conda/include
  make -j$(nproc)      # builds libtaso_runtime.so against real cudnn/*.cu

  cd ../../tensat
  export CARGO_HOME=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/cargo_container
  export TASO_LIB_DIR=$PWD/../taso/build_gpu TASO_INCLUDE_DIR=$PWD/../taso/include
  export PROTOBUF_LIB_DIR=/opt/conda/lib
  export LD_LIBRARY_PATH=$TASO_LIB_DIR:$PROTOBUF_LIB_DIR:$LD_LIBRARY_PATH
  cargo build

  ./target/debug/tensat -r converted.txt -d nasrnn -e greedy \
      --n_iter 3 --n_sec 30 -s none -o tmp/stats.json -x tmp/nasrnn
      # NOTE: no --no_runtime_report this time -- real GPU measurement path
  ```
  Result: Runner 28.6ms (490 nodes/450 classes/40 programs — same
  saturation shape as CPU), Extractor 7.5ms, best cost 1333.94 (now real
  cudnn-measured units, not comparable to the CPU synthetic-model number).
  **Start graph runtime 0.6125 vs extracted graph runtime 0.00518** — a
  real, non-zero, GPU-measured runtime improvement, confirming the full
  pipeline (equality saturation → real per-op cuDNN cost measurement →
  extraction → real full-graph execution timing) works end-to-end. The
  ~118x gap is plausible specifically for nasrnn (many tiny sequential
  RNN-cell matmuls, so kernel-launch-overhead-dominated — exactly where
  TENSAT's parallel-matmul-fusion rewrite pays off most) but this is one
  smoke test on random weights with a tiny search budget, not a benchmark
  result.

### 9.1b CPU build reproduced on the cluster login node (2026-08-22)

Separately from the GPU container (§9.2), the CPU-only path from §9.1 was
also rebuilt directly on `klone-login03` (no container) — useful as a
lightweight sanity check that doesn't need a GPU allocation. Toolchain
lives entirely under `/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/`
(miniconda3, cargo, rustup — kept outside the project dir because its path
has spaces, which breaks the Miniconda installer). Gotchas beyond the
shared ones in §9.2 (protobuf/libprotobuf naming, conda ToS, CMake policy
version): the login node's system `libclang.so.17` has no accompanying
`clang` driver binary, so `bindgen` can't auto-detect its resource dir —
needs `BINDGEN_EXTRA_CLANG_ARGS=-resource-dir=/usr/lib64/llvm17/lib/clang/17`
set explicitly. Full env-var recipe and validated command are in the
conversation; smoke-test output (490 nodes, 40 programs,
`nasrnn_{start,optimized}.model`) is byte-identical to the original Mac run.

**Housekeeping note:** early in this session, `egg`/`taso`/`tensat` briefly
existed as two divergent checkouts — the intended ones inside this project
dir (uncommitted local edits on an older base commit) and a duplicate
directly under `/mmfs1/gscratch/scrubbed/sgvtc/` (same edits, but properly
committed: `taso@9499b9c`, `tensat@88e9933`, `egg@12cc1ee`). Content was
byte-identical between the two; the properly-committed copies were moved
into the project dir and the stale duplicates removed. If a future session
finds repos at the project root missing expected commits, this is why —
there's only one copy now.

### 9.3 Don't copy these to the cluster as-is

Found while migrating (2026-08-22) — all platform-specific, all safe to
exclude from the transfer and regenerate natively on the cluster instead:
- `alpha-beta-CROWN/.venv/` (881M) — macOS-built venv; recreate from
  `alpha-beta-CROWN/pyproject.toml`/`uv.lock` (or
  `complete_verifier/environment_pyt28{0,...}.yaml` for a conda-based
  setup) on the cluster instead.
- `tensat/target/` (501M) — Rust build artifacts; `cargo build` regenerates.
- `taso/build/` (54M) — our CPU-only CMake build; needs a full GPU rebuild
  there per §9.2 anyway.
- `tensat/.envrc` — see §9.1, Homebrew-specific path, will exist after
  rsync but needs rewriting for Linux (or dropping if the build moves
  fully inside the Apptainer image).
