# Progress Log

Chronological record of what's been done on this project. Append new
entries at the bottom under today's date (add a new `##` date header if
it's a new day). See `TENSAT_SUMMARY.md` for the technical deep-dive
(paper summary, build internals, session state) and `BUGS.md` for
full write-ups of bugs found in vanilla TASO/tensat.

## Earlier (MacBook, arm64, no GPU — see TENSAT_SUMMARY.md §9.1)

- Built TASO's CPU-only path (`USE_CUDA=OFF`, synthetic FLOP/mem-traffic
  cost model) since `yycdavid/taso` has no CPU backend at all upstream.
- Validated `tensat` builds and runs against it (`cargo build`, `cargo
  test`, nasrnn smoke test via `-d nasrnn`) — but only with *random*
  weights (TENSAT's built-in benchmark models, not a trained network).

## 2026-08-22 — Migrated to the Hyak Klone cluster, GPU build working

- Consolidated a duplicate-checkout mixup: `egg`/`taso`/`tensat` briefly
  existed both inside the project dir (stale, uncommitted) and directly
  under `/mmfs1/gscratch/scrubbed/sgvtc/` (properly committed) — kept the
  latter, moved into the project dir.
- Reproduced the CPU-only build on the cluster login node (`klone-login03`)
  with a fresh toolchain (`toolchain-tensat/miniconda3` + rustup, kept
  outside the project dir because its path has spaces). Byte-identical
  nasrnn output to the original Mac run.
- Built the GPU path from scratch: `tensat.def` (Apptainer, Ubuntu 22.04 +
  CUDA 12.4 + cuDNN 9), built on compute node `g3109` (`gpu-l40s`
  partition, 2x L40S). Along the way fixed: Apptainer cache defaulting to
  a quota-limited home dir, Anaconda's Terms-of-Service block on
  non-interactive `conda install`, `protobuf` vs `libprotobuf` package
  naming on conda-forge, CMake ≥4.0 rejecting `cmake_minimum_required
  (VERSION 3.2)`, `find_package(Protobuf)` needing an explicit prefix
  path. `taso`'s `src/cudnn/*.cu` compiled against cuDNN 9 with **zero**
  source changes — the one real technical unknown from the migration plan
  turned out fine.
- Ran `tensat` on the GPU against nasrnn (still random weights): real
  cuDNN-measured costs, real end-to-end graph-runtime timing confirmed
  working (~118x runtime improvement, not directly comparable to any
  benchmark — one smoke test, random weights, tiny search budget).
- Picked `mnist_tiny_mlp` (alpha-beta-CROWN's own smallest example
  model, `Flatten→Linear(784,20)→ReLU→Linear(20,10)`, real trained
  weights from `models/toy/mnist_2_20.pth`) as the target for a real,
  trained-weight test case.
- Built TASO's Python bindings (Cython extension) against the GPU build;
  fixed `setup.py` linking against `taso_runtime` via shell-split
  `LDFLAGS` (broke on the project's space-containing path) by reading
  `TASO_LIB_DIR` directly instead.
- Converted `mnist_tiny_mlp` end-to-end: PyTorch `.pth` → ONNX →
  `taso.load_onnx()` → `export_to_file()`. Found two real gaps doing
  this: `tensat`'s `-f` flag never actually called the parser built for
  this exact format (`parse_model()` existed, tested, unused — used a
  generic egg S-expression parser instead, which silently produced a
  degenerate 1-node graph on real input); and TASO's own ONNX `Gemm`
  importer silently dropped the bias input. Fixed both:
  - `tensat` commit `e349c73` — `-f` now calls `parse_model()` directly.
  - `taso` commit `bd8ba5d` — `_gemm` now adds the bias via `graph.add()`.
- Ran the corrected, bias-complete graph through `tensat`'s GPU optimizer
  on the real trained weights: real cuDNN-measured cost, **~8.9x**
  real hardware-measured runtime improvement (0.01536 → 0.00173).
- Got alpha-beta-CROWN running (`uv sync`, fixing the same
  quota-limited-home-cache issue as Apptainer hit) and produced a real
  verified bound on this same model/checkpoint: PGD found no adversarial
  example, initial CROWN bounds alone proved robustness (all margins > 0)
  in 0.93s — resolved without needing branch-and-bound.
- Closed the loop: built `NNs/reconstruct_optimized.py` to take tensat's
  optimized graph, match its weight nodes back to the real original
  arrays (unambiguous here — this model's four weight shapes are all
  distinct), and export real, correct ONNX. Found three more bugs doing
  this, all in vanilla TASO (`preprocess_weights()`'s C++ map-iteration
  bug that collapsed the whole graph; `graph.transpose()`'s `perm`
  attribute not surviving ONNX export; `export_onnx()` emitting the
  invalid op name `"Matmul"` instead of ONNX's `"MatMul"`) — worked
  around the first two (folding weights in numpy directly instead of
  relying on TASO's broken fold/transpose path), fixed the third
  (`taso` commit `1b7bcda`). Final result: **the reconstructed ONNX
  model's output matches the original PyTorch model to ~1e-6** (float32
  rounding) — the full TENSAT round-trip preserves real, correct
  numeric weights, not just graph structure.
- Wrote `BUGS.md` cataloguing all five bugs found (one in `tensat`, four
  in `taso`) with file/line references, root-cause analysis, and fix
  status, for filing as GitHub issues later.
- Initialized this repo's own git history and pushed it to GitHub
  (`xvade/E-graphs_for_Verifiability`). `egg`/`taso`/`tensat` are wired in
  as proper submodules; `taso` and `tensat` carry local fixup commits that
  only existed on this wipeable scratch filesystem, so those got pushed to
  personal forks first (`xvade/TASO` on a new branch, since that fork's
  `master` turned out to be an unrelated pre-existing history;
  `xvade/tensat`'s `master` directly, since that was already this
  project's tracked origin). `tensat.sif` and the `alpha-beta-CROWN`
  checkout (mostly its own 8G `.venv` + datasets) are excluded from git;
  `README.md` documents how to regenerate both.
- Extended the round-trip pipeline to a second, larger alpha-beta-CROWN
  example model: `mnist_cnn_a` (`model_defs.mnist_cnn_4layer` —
  `Conv(1,16,4x4,s2,p1)→ReLU→Conv(16,32,4x4,s2,p1)→ReLU→Flatten→
  Linear(1568,100)→ReLU→Linear(100,10)`, real trained weights from
  `models/sdp/mnist_cnn_a_adv.model`), the first model in this project
  with convolutions. Extended `taso`'s ONNX `Conv` importer (`_conv2d`)
  with the same bias fix `_gemm` got earlier — this time needing a
  `[1,C,1,1]` reshape first, since a conv bias broadcasts against the
  channel axis rather than the trailing axis Gemm's bias aligns with —
  and extended `NNs/reconstruct_optimized.py` with a `Conv` dispatch
  branch. Hit and root-caused a much bigger bug doing this: TASO's
  `Graph::get_operator_int_attr` computes its return value *inside* an
  `assert(...)`, which is compiled to nothing under `NDEBUG` (i.e. any
  CMake `Release` build — exactly what this project's GPU build uses),
  silently returning uninitialized garbage for every attribute read
  through that path (Conv/Pool's strides/kernel_shape/pads, but also —
  turns out — `Transpose`'s `perm`, retroactively explaining the earlier
  "not conclusively identified" transpose bug from bug #4). Fixed with a
  two-line change (`taso` commit `fb0b3db`, pushed alongside the Conv bias
  fix). The identical assert-with-side-effect pattern recurs many times in
  `substitution.cc` (the rewrite-matching engine itself) — flagged in
  `BUGS.md` as an open, unaudited risk, not fixed. Ran `mnist_cnn_a`
  through `tensat`'s GPU optimizer (real cuDNN-measured cost, 0.0646 →
  0.0169, ~3.8x runtime improvement) — worth noting honestly that the
  extracted graph came back structurally isomorphic to the input for this
  run (no rewrite fired), so this specific result demonstrates the
  pipeline working end-to-end on a conv model rather than TENSAT finding a
  nontrivial restructuring. Reconstructed real weights and verified
  numerically against the PyTorch reference: **max abs diff 9.5e-07**,
  same as the tiny MLP case.

## 2026-08-22/23 — Chasing (and finally getting) a real nontrivial rewrite

Goal for this stretch: find a real, TENSAT-*selected* structural rewrite
(not just an isomorphic re-extraction) on a real-trained-weight model, and
verify it numerically. Long chase, several dead ends, real bugs found
along the way, genuine success by the end.

- Picked `resnet2b` (`model_defs.resnet2b`, real CIFAR-10 weights from
  `models/cifar10_resnet/resnet2b.pth`) as the next candidate — a real
  ResNet with a residual/shortcut branch, the first model this session
  with anything resembling a parallel structure. Hit a segfault
  converting it: `taso`'s ONNX `Reshape` importer only looked for its
  shape argument in the graph's `initializer` list, but this model's real
  `torch.view()` call exports the shape via a separate `Constant` node
  instead (every earlier model's `Flatten`-based export needed no shape
  arg at all, so this path was never exercised before). Root-caused and
  fixed (`taso` commit `af3770a`, `BUGS.md` #8) — see the entry logged
  under 2026-08-22 above for the fix itself; write-up in `BUGS.md` was
  filled in this stretch.
- Ran `resnet2b` through `tensat`'s optimizer (single-pattern rules only,
  various `--n_iter`/`--n_sec` budgets, `-e greedy`): every run extracted
  a graph structurally *isomorphic* to the input (same op-type multiset,
  same params, just guid-renumbered) — no rewrite ever won under the real
  cost model, even after letting saturation fully converge.
- Built genuine random sampling into `tensat` itself (`--n_random N
  --random_seed S`, new CLI flags; `RandomCost` in `tensat/src/
  optimize.rs`) since egg has no built-in "extract N alternates" mode.
  First version (pure random cost, summed over children) reliably
  reproduced the same isomorphic structure every time — traced this to a
  real bias: summing random per-node costs penalizes any multi-node
  equivalent subtree, since it accumulates more random draws than a
  single-node alternative almost regardless of the draws.
- Enabled `tensat`'s multi-pattern rewrite mechanism (`--use_multi -t
  converted_multi.txt`) for the first time this session — it implements
  the classic TASO parallel-conv-fusion rule (pad/enlarge one conv's
  kernel, concat weights, one wider conv, split back), gated behind a
  flag no run had used yet. Zero matches on `resnet2b`. Root-caused:
  `PRE_DEFINED_MULTI`'s patterns hardcode stride=(1,1) literally, and
  `resnet2b`'s only same-input parallel-conv point (its downsampling
  shortcut) is stride=2 by construction, like every model in `alpha-beta-
  CROWN`'s whole ResNet family — shortcuts there only ever exist
  *together with* stride-2 downsampling, never at stride 1.
- Same zero-match result on `tensat`'s own built-in synthetic ResNet50
  benchmark (`-d resnet50`, random weights) — surprising, since that's
  exactly the kind of model these rules were written against. Root cause
  #2: `--iter_multi` (how many saturation iterations the multi-pattern
  search actually runs on) defaults to 1, so it only ever searched
  iteration 0, before single-pattern rewriting had normalized anything.
  Raising it let the search run properly, but on ResNet50 the egraph then
  exploded (562k → 2.97M nodes across attempts) from combinatorial
  single-pattern axiom growth, without ever finding a multi-pattern match
  either — inconclusive on real matching, informative on scale limits.
- Went back to `resnet2b` with instrumented multi-pattern matching (added
  debug counters through every stage of `MultiPatterns::apply_match_pair`
  in `tensat/src/rewrites.rs` — search hits, compatibility, validity,
  cycle-check) to get real data instead of guessing. Found the rewrite
  *does* apply thousands of times successfully (`cycle_ok` in the
  thousands) — but manual causal-chain tracing of `resnet2b`'s four
  same-shape relu positions showed all four sit on one strict sequential
  chain (each computed using the previous one's output), so these
  "successes" are near-certainly fusing duplicate representations of the
  *same* underlying value (spawned by single-pattern associativity/
  commutativity axioms), not genuine independent-branch batching. Tried
  hand-constructing a real fusion anyway (`NNs/reconstruct_fused_relu.py`,
  concat+relu+split on two same-shape relus) — hit exactly this causality
  wall directly (`KeyError` on a guid whose relu was needed as an
  ordinary intermediate before the "fusion" could complete), confirming
  the analysis empirically as well as by hand.
- Searched exhaustively for a real ab-CROWN model with a genuine
  independent parallel branch (`model_defs.py`'s full class list, a
  direct `torch.cat`/`.cat(` grep across every model file): none exist.
  The one architecture that would qualify (`resnet4b1`/`resnet4b2`, via
  `BasicBlock_eth`'s stride-1 channel-changing shortcut) has no trained
  checkpoint in this checkout.
- Trained a small custom model instead: `InceptionMNIST`
  (`NNs/inception_mnist_model.py`) — a stem conv followed by two parallel
  branches (1x1 and 3x3 conv, both stride=1, same input) merged by
  addition, specifically shaped to match `PRE_DEFINED_MULTI`'s literal
  pattern. Trained on real MNIST (idx files already cached under
  `alpha-beta-CROWN`'s dataset dir, parsed directly with no torchvision
  dependency) — 85.65% test accuracy on a 1-epoch/10k-image run (the
  full 3-epoch/60k run was killed after 30+ min with no output; the
  shared login node was under heavy resource contention overnight, see
  below).
- Confirmed via the exported `.taso` file that this model's two branch
  convs really do share one input at stride=1/padding=SAME/activation=
  NONE — the first model all session to genuinely qualify. Running with
  `--use_multi -t converted_multi.txt --iter_multi 15` alone still didn't
  win under the real cost model (same story as `resnet2b`'s relu-merge
  rule: the fused form is real ops, not free). Added a new `--favor_fusion`
  flag (`CostModel::with_favor_fusion` in `tensat/src/optimize.rs`) that
  deliberately discounts `Concat`/`Split`/`Enlarge`'s real measured cost
  by 20x for deterministic greedy extraction — a knob to surface an
  already-proven-valid equivalence for comparison, explicitly not a real
  cost-model claim. First attempt (`--iter_multi 15`) let the search
  explode combinatorially (2.97M nodes) and extracted a tangled,
  redundant fusion (an 88-output-channel conv via nested Concat/Enlarge
  chains) that wasn't worth the risk of hand-reconstructing at this hour.
  A more conservative retry (`--n_iter 3 --iter_multi 1`) produced a
  clean, small result: **Conv count 3→2, plus Concat/Split/Enlarge
  appearing for the first time this session** — the genuine nontrivial
  rewrite this whole multi-day chase was after.
- Reconstructed it (`NNs/reconstruct_inception_fused.py`) with real
  weights. The extracted graph turned out to be a valid but *hybrid*
  program — it keeps one of the two original convs computed the ordinary
  way, and additionally builds the wider fused conv whose second half
  stands in for the other branch's contribution (confirmed algebraically
  by hand: `conv_a(x) + (bias_a+bias_b) + conv_b(x)` still equals the
  original `(conv_a(x)+bias_a) + (conv_b(x)+bias_b)`, just computed via a
  different, partially-redundant path). Hit two more real, previously-
  unexercised bugs doing this:
  - Three of this model's weights share shape `(8,)` (stem/branchA/
    branchB bias), breaking pure shape-based weight matching for the
    first time — worked around by loading named weights directly from
    the PyTorch checkpoint and mapping specific guids to specific roles
    by hand-tracing the exported graph's structure (not a general fix,
    documented as a known limitation in the script itself).
  - `export_onnx()` always emits `Split`'s sizes as a node *attribute*,
    which is invalid once opset 13 is declared (the attribute form was
    dropped from ONNX's own `Split` spec in favor of an input tensor) —
    `BUGS.md` #9. Worked around by exporting this one model at opset 11
    instead of the usual 13.
  - Also worked around a second, cosmetic issue: the hybrid graph leaves
    one `Split` output genuinely unused, which `export_onnx()` correctly
    (if unhelpfully) treats as an extra ONNX graph output — filtered
    `onnx_model.graph.output` down to the one real `(1,10)`-shaped output
    before saving.
  - **Verified: max abs diff 1.67e-06** against the PyTorch reference —
    numerically correct.
- Also reconstructed the *unfused* baseline the same way
  (`NNs/reconstruct_inception_unfused.py`, straight from
  `NNs/inception_mnist.taso`) for a clean side-by-side pair — **verified:
  max abs diff 2.15e-06**. Both real weights, both numerically confirmed
  correct, ready for an eventual `alpha-beta-CROWN` verifiability
  comparison (not yet run this stretch).
- Infrastructure note: this stretch ran through the night while the user
  slept, working autonomously per their instruction. Two real
  environmental problems came up along the way, both resolved without
  touching any user data:
  - The GPU SLURM allocation expired partway through and was
    auto-renewed under a new job ID at least twice — commands needed
    re-pointing at the current `--jobid` (checked via `squeue`) each time.
  - The host-side `taso_py` conda env's `numpy` package (and, separately,
    the base `miniconda3` environment's own Python stdlib) turned up
    corrupted partway through — `numpy/__init__.py` missing entirely
    despite all its submodule directories being present, `conda` itself
    unable to run (`Fatal Python error: init_fs_encoding`). Root cause
    not conclusively identified (most likely an interrupted/OOM-killed
    package operation earlier in the session, given real memory pressure
    observed on the shared login node around the same time — `dmesg`
    showed an actual OOM kill, of an unrelated user's process, around
    when this started). Fixed narrowly: reinstalled just `numpy` via
    `taso_py`'s own `pip` (`rm -rf` the broken package dir first, since
    it had no `pip` `RECORD` file to let `pip` uninstall it cleanly) —
    didn't touch the still-broken base `miniconda3` env, since nothing
    in this session's pipeline actually depends on it directly.

## 2026-08-23 — Ran the fused/unfused InceptionMNIST pair through alpha-beta-CROWN

Goal: the actual comparison this whole project exists for — real
verification results on the TENSAT-optimized model vs. the original,
same real weights. First time any of this session's own TASO/tensat-
reconstructed ONNX files (as opposed to a PyTorch `model_defs` class) had
been run through `alpha-beta-CROWN` at all. Hit four more real,
previously-unexercised integration bugs getting there, each root-caused
and fixed or worked around in turn:

- **`onnx2pytorch`'s `Add` is order-sensitive.** It converts ONNX `Add`
  as an in-place `out += inp`, which needs the first operand to already
  be the broadcast-target (larger) shape — our exported bias-adds
  sometimes listed the smaller (bias) tensor first. Fixed by reordering
  `Add`'s operands by real tensor volume in both reconstruction scripts
  (`add_larger_first()` — mathematically free, `Add` is commutative).
- **PGD attack batching collides with the fused model's own axis-0
  fusion trick.** `onnx2pytorch`'s `Split` choked once the PGD attack's
  internal batching (many parallel restarts) made the real tensor size
  along axis 0 diverge from the model's hardcoded `[1,1]` split sizes.
  Worked around with `pgd_order: skip` (applied to both configs, for a
  fair identical methodology) — CROWN/BaB bound computation is the
  metric that actually matters here anyway.
- **`auto_LiRPA`'s bound propagation choked on TASO's Reshape-shape-
  tensor decomposition** (`RuntimeError: shape '[1, 1519]' is invalid
  for input of size 12152`, identically on both fused and unfused
  models). Fixed on the `taso` side: `export_onnx()` was listing every
  `Weight` and every `Reshape`'s shape constant as *both* a real
  initializer and a formal graph input (`BUGS.md` #10, `taso` commit
  `e73ced7` — the fix onnxruntime's own long-standing, previously-
  ignored warning already pointed at). That alone didn't fully resolve
  this specific crash, but running the (now-cleaner) exported ONNX
  through `onnxsim` did — it folds the whole Reshape-shape
  reconstruction machinery into a plain static reshape, cutting node
  count roughly in half (21→14 fused, 18→10 unfused) and eliminating the
  crash. Re-verified numerically correct after simplification (~1e-6,
  same as before).
- **The exported models hardcoded batch size 1**, breaking once BaB
  needed to vectorize across multiple branch-and-bound sub-domains
  (`RuntimeError: shape '[1, 6272]' is invalid for input of size 37632`
  — exactly 6272×6, BaB's batch size at that point). Fixed by patching
  the ONNX graph directly: the flatten `Reshape`'s target-shape constant
  (`1` → `-1`, "infer from input") and the graph's declared input batch
  dimension (fixed `1` → a symbolic `dim_param`). Confirmed both models
  now handle batch>1 plain inference correctly.
- **Found the real, unfixable limit for the fused model specifically**:
  even with dynamic batching enabled, real batch>1 inference on the
  fused model fails outright (`Split129 ... Sum of sizes in 'split' ...
  was 2` against an actual axis-0 size of 4+) — and separately,
  `auto_LiRPA` explicitly asserts `Concat`'s axis must be `> 0`
  (`auto_LiRPA/operators/slice_concat.py`), never allowing bound
  propagation through the batch axis at all. This is `BUGS.md` #11: not
  a bug, a genuine structural fact. TENSAT's selected rewrite for this
  model batches two independent computations together *by concatenating
  along the input's own batch axis* — numerically correct and exactly
  what made it faster (real GPU runtime 0.147→0.029, per the earlier
  entry), but that exact same trick is fundamentally incompatible with
  any verifier (not just `auto_LiRPA`) that itself needs to batch
  multiple problem instances along that axis. **The fused model cannot
  be verified by alpha-beta-CROWN at all**, for a structural reason, not
  a tooling gap.

**Actual verifiability results — unfused baseline** (10 real MNIST test
images, `epsilon=0.1`, `Linf`, 60s `bab` timeout each,
`exp_configs/beta_crown/inception_mnist_unfused.yaml`):
- **Final verified accuracy: 20.0%** (2 of 10 safe-incomplete, 0 unsafe,
  8 timeout/unknown)
- Verified-safe indices: 0, 3. Mean time for verified instances: 22.5s.
  Mean time overall: 55.1s (most instances ran to the full 60s timeout
  without resolving either way).
- Context: this is a real but weak classifier (85.65% test accuracy from
  a deliberately short 1-epoch/10k-image training run — see the
  2026-08-22/23 entry) at a fairly large `epsilon=0.1` for MNIST, so a
  mostly-unresolved result at this compute budget is unsurprising rather
  than alarming.

**Fused model: not run** — structurally cannot be, per above. The
honest, complete comparison this session produced is therefore: a real
verifiability number for the original model, and a concrete,
well-understood *reason* the optimized model can't be given one at all
by this tool, which is itself a substantive finding for the project's
core question (does TENSAT's optimization affect verifiability) — in
this case, about as strongly as possible: it doesn't just change bound
tightness, it can produce a graph that formal verification tooling
cannot process, without changing the network's actual input-output
behavior one bit.

All ONNX-file patches applied directly to
`NNs/inception_mnist_{fused,unfused}_simplified.onnx` (the batch-
dimension and Reshape-shape edits) — not yet folded back into the
`reconstruct_inception_*.py` scripts themselves, so regenerating from
scratch would need those same patches reapplied by hand or scripted.

## 2026-08-23: fusion-v2 — a TENSAT rewrite that actually clears ab-CROWN, plus a controlled comparison

The above entry left the project with a real unfused verifiability
number but no fused counterpart, since the one fusion TENSAT had picked
used the batch axis and was structurally unverifiable (`BUGS.md` #11).
Rather than accept that as the final word, went back into `tensat`'s
cost model to see whether a *different*, ab-CROWN-compatible rewrite
was reachable for the same `InceptionMNIST` graph.

**The relu-merge rule has two axis variants.** `tensat`'s multi-pattern
relu-merge rewrite (concat two branches' pre-activations, apply `Relu`
once, split back apart) is encoded with both an axis=0 (batch) and an
axis=1 (channel) instance in the rule set. Mathematically identical
trick, either axis works for the network's own semantics — but only
axis 1 is usable by `auto_LiRPA` (`assert self.axis > 0` in
`slice_concat.py`'s `BoundConcat.bound_backward`). The original
extraction picked axis=0 on its own.

**Why axis=0 kept winning, even without `--favor_fusion`.** First tried
making the `favor_fusion` discount axis-aware but still *neutral*
toward axis-0 (discount axis≠0 Concat/Split and `Enlarge`, leave axis=0
undiscounted) — re-extraction still produced the axis-0 graph. Tested
with `--favor_fusion` removed entirely to isolate the cause: axis=0
*still* won, on genuine cost-model merit, not a discount artifact — it
lets the stem's `Relu` be reused directly instead of redundantly
recomputed via the batched trick, which is a real compute saving the
axis=1 variant doesn't get. Neutral wasn't enough; getting the axis=1
variant out of the extractor required an active penalty. Changed
`CostModel::get_self_cost` (`tensat/src/optimize.rs`) to multiply cost
by `1000.0` for axis=0 `Concat`/`Concat3/4/5`/`Split` when
`favor_fusion` is set (while still discounting axis≠0 instances and
`Enlarge` by `0.05x`), which reliably produces the axis=1 extraction
instead.

**Reconstruction.** `NNs/reconstruct_inception_fused_v2.py` rebuilds
this new extraction (`tensat/tmp/inception_mnist_v2_optimized.model`)
with the real trained weights, same approach as the first fused script
but with an unambiguous weight-role mapping (biases aren't pre-summed
this time, so each traces directly to its own conv via the guid
graph). Verified numerically identical to the reference PyTorch
checkpoint (~1.4e-6 max abs diff,
`NNs/verify_reconstruction_inception_fused_v2.py`). Simplified via
`onnxsim` (20→12 nodes) and patched for batch-flexibility the same way
as the unfused/first-fused models
(`NNs/inception_mnist_fused_v2_simplified.onnx`); confirmed both
batch=1 correctness and batch=4 consistency manually before touching
ab-CROWN.

**A second, unrelated ab-CROWN limitation surfaced immediately**:
running this model crashed with `NotImplementedError:
<class 'auto_LiRPA.operators.slice_concat.BoundConcat'>` — but only
*after* bound propagation itself had already succeeded (real CROWN
bounds computed, 8/9 specs resolved directly). The crash was in the
default `kfsb`/`babsr` branching heuristic
(`heuristics/babsr.py`'s `get_babsr_biases()`), which only has cases
for standard layer types and was never written to score a `Concat`
layer for branching. Documented as `BUGS.md` #12. Worked around with
`bab.branching.method: random` (doesn't need per-layer scoring), the
only branching option compatible with a graph containing `Concat`.

**Controlled comparison.** Since the fused model is forced onto
`random` branching, comparing it against the unfused baseline's
existing `kfsb` result (`20.0%`, above) would confound the fusion
itself with the branching-heuristic change. Reran the unfused model
under `random` too
(`exp_configs/beta_crown/inception_mnist_unfused_randombranch.yaml`)
for an apples-to-apples number. Same 10 MNIST test images,
`epsilon=0.1`, `Linf`, 60s timeout:

| model | branching | verified accuracy | verified-safe indices |
|---|---|---|---|
| unfused | `kfsb` (default) | 20.0% (2/10) | 0, 3 |
| unfused | `random` | 20.0% (2/10) | 0, 3 (mean 2.78s — faster, likely variance) |
| fused_v2 | `random` | 10.0% (1/10) | 3 only |

Under the *same* branching method, the fused model verifies strictly
fewer instances than the unfused one (loses index 0) — a real,
fusion-attributable drop in verifiability, not an artifact of a
different search heuristic. This is the first end-to-end fused-vs-
unfused comparison the project has, and it directly supports the
project's core hypothesis: TENSAT's structural rewrite, despite being
numerically exact, measurably reduced how much of the network ab-CROWN
could verify within the same compute budget.

Full logs: `NNs/abcrown_out_inception_mnist_fused_v2.log`,
`NNs/abcrown_out_inception_mnist_unfused_randombranch.log`. Configs
mirrored into git-tracked copies at
`NNs/abcrown_config_inception_mnist_fused_v2.yaml` and
`NNs/abcrown_config_inception_mnist_unfused_randombranch.yaml` (the
`alpha-beta-CROWN/` directory itself is gitignored).
