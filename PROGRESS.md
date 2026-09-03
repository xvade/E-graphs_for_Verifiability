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

## 2026-08-24: structural-diversity-vs-verifiability campaign (Phases 1-7)

Goal: move past comparing exactly two hand-picked structures for one
model, toward real data on how graph *structure* (not just "fused vs.
not") relates to ab-CROWN verifiability, to inform a future custom
`tensat` cost function. Built as a 7-phase pipeline with pause points;
each phase is summarized below, ending with the actual sweep results.

**Phase 1 -- automatic weight-provenance tracking in `tensat`.** The
biggest blocker to sampling many extractions (rather than hand-tracing
one) was that TASO's `.model` export assigns fresh, meaningless guids on
every extraction and never stores real weight names at all (confirmed:
TASO's ONNX loader discards initializer names before calling
`graph.new_weight()`, whose binding takes no name argument; the `.model`
format itself has no name field). Fixed by extending `tensat/src/
model.rs`'s `ValTnsr` with a `weight_names: BTreeSet<String>` field,
propagated bottom-up through every `TensorAnalysis::make()` arm (mirrors
the existing `all_weights: bool` pattern -- union of children instead of
AND, singleton at `Weight` leaves) and reconciled in `merge()`. Real
names are seeded once per model at baseline parse time
(`tensat/src/parse.rs`'s new `parse_model_with_names`, `tensat/src/
input.rs`'s new `new_weight_named`, driven by a new `--weight_names_json`
CLI flag) from a `guid -> name` sidecar
(`NNs/<model>_weight_names_baseline.json` -- for InceptionMNIST this is
just the existing hand-derived `GUID_ROLES` dict moved to JSON; for
mnist_cnn_a/resnet2b, a new `NNs/derive_weight_names_baseline.py` derives
it automatically by shape-then-position matching against the ONNX
initializers, correctly handling resnet2b's real shape collisions -- 3x
`(16,16,3,3)` conv kernels, 5x `(16,)` biases -- and one genuine orphaned
Constant-derived node). At export time, `save_model_with_provenance`
(`tensat/src/main.rs`) walks the replayed egraph and writes a
`<file>.weight_names.json` sidecar (guid -> contributing names) for
*any* weight-derived eclass, not just literal `Weight` leaves. Verified:
zero behavior change without the new flag; with it, reproduces the
known-good InceptionMNIST mapping exactly.

**Phase 2 -- generalized reconstruction + a real TASO bug found.**
`NNs/reconstruct_generic.py` replaces the per-extraction hand-written
`reconstruct_*.py` scripts, resolving weight identity via the Phase 1
sidecar instead of a hardcoded dict; its output-selection logic
generalizes to correctly find the real final output even when a
fused/sampled extraction leaves an orphaned Split half in the graph.
Regression-testing it against all 3 baseline models plus the known-good
fusion surfaced a real, previously undocumented `taso` bug (**BUGS.md
#13**): `ts.export_onnx()` emits *asymmetric* TF-style SAME-padding for
a stride>1 conv with odd total padding, but TASO's own `Conv2D::
get_padding()` pads *symmetrically* when actually executing the op --
these disagree exactly on `resnet2b`'s stem conv and `layer1.0.conv1`
(both kernel=3/stride=2), producing a max abs diff of 1.34 against the
real reference output until patched (now fixed generically in
`reconstruct_generic.py`'s `fix_same_padding_symmetric()`, ~8e-7 after).
All 3 baselines plus a fresh fusion sample now pass regression at
~1e-6.

**Phase 3 -- two new extraction modes in `tensat`.** `--random_mode
{jitter,uniform}` on the existing `--n_random` (new `uniform`:
`UniformRandomCost`, i.i.d. cost per enode independent of the real cost
model, documented small-tree bias). `--n_diverse N` (new `DiverseCost`):
samples in sequence, penalizing re-use of any enode a previous sample
already used, to push toward structurally distinct regions of the
egraph. Both smoke-tested successfully.

**Phase 4 -- pre-flight diversity check, and a real methodological
discovery.** Running both new modes (15 samples each) on all 3
candidate models initially showed *zero* structural diversity anywhere
-- including InceptionMNIST, which has a known fusion. Investigating
why (rather than accepting "no diversity") found: (a) `mnist_cnn_a` and
`resnet2b` genuinely never produce a `Concat`/`Split` under any setting,
confirmed twice each (`--n_diverse`/`--random_mode uniform`, and a
direct `--use_multi` check) -- `mnist_cnn_a` has no parallel branches at
all, and `resnet2b`'s only same-shape relu positions are causally
chained, not independent (matches this project's earlier finding); (b)
InceptionMNIST's fused/unfused real-cost gap, even after
`--favor_fusion`'s discount, is *marginal* rather than decisive --
deterministic extraction with the same settings sometimes finds the
fusion and sometimes doesn't, from ordinary run-to-run noise in TASO's
own GPU cost measurement (confirmed: identical repeated invocations gave
"Best cost" 0.221 once and ~0.112 four times out of five). Made
`favor_fusion` continuous (`CostModel::favor_fusion_strength: f32`,
1.0=neutral, replacing the old bool) so this can be dialed deliberately
rather than hoping jitter stumbles into it -- validated the mechanism
works (saturates around strength=0.05, since the *wider conv's own*
undiscounted real cost, not the discount target, becomes the limiting
factor beyond that). Even so, across 60 repeated deterministic
extractions at strength=0.05, the safe fusion won only ~3% of the time
(2/60) -- confirming InceptionMNIST has a small, *finite* set of
genuinely distinct structural types (unfused; the one safe channel-axis
fusion; the one unsafe batch-axis fusion already characterized in
`BUGS.md` #11) rather than a large continuous space, an intrinsic
property of this small hand-built model. Also found (not a safety bug,
documented for completeness): the axis-based safety check in
`get_self_cost` can't distinguish a *weight*-level axis-0 `Concat` (the
conv-fusion rule's safe output-channel concat, which numerically folds
away entirely during Python reconstruction and never reaches ONNX) from
an *activation*-level axis-0 `Concat` (the unsafe batch-axis one) --
doesn't affect final verifiability since the former never survives to a
real ONNX node either way. Both automated (`--n_diverse`) *and* manual
`--favor_fusion_strength` samples of the safe fusion always landed on
the same axis (`axis=1` on both `Concat` and `Split`), confirming the
axis-0 penalty correctly excludes the unsafe variant even under
sampling. Net result: `mnist_cnn_a`/`resnet2b` excluded from the sweep;
InceptionMNIST included with 3 structural types instead of a large
sample space -- which changed Phase 5's scope from "15 novel samples"
to "thorough epsilon coverage of the ~3 structures that actually
exist," and, since that shrank the sample count dramatically, let the
image count go back up to the full established 10 (not the originally
planned 4-image reduction, which only existed to control cost under a
much larger assumed sample count).

**Phase 5-6 -- batch driver and the sweep itself.**
`NNs/run_verification_sweep.py` runs alpha-beta-CROWN via CLI overrides
(`--onnx_path`, `--epsilon`, `--start`/`--end`) on one base YAML per
model (confirmed working, including passing `--onnx_path` as an absolute
path), capturing full untruncated stdout per run and parsing per-image
verdicts plus the summary block into a resumable
`NNs/sweep_results.jsonl`. Calibration (2 images each) confirmed real
per-image time can exceed the nominal `bab.timeout=60` by up to ~100s
(pre-BaB overhead not counted against the timeout -- consistent with
prior runs, e.g. fused_v2's previously-recorded 94s max at the same
60s setting) and each subprocess pays a fixed ~176-190s library-init
cost regardless of image count; revised worst-case budget ~5h, well
under the ~8-12h approved. All 13 planned runs (5-point MNIST-family
epsilon grid `{0.02,0.05,0.1,0.15,0.2}` x 10 images for InceptionMNIST
unfused and the hand-verified `fused_v2`; 1 epsilon x 10 images for a
freshly automated-pipeline-discovered fusion sample `fused_auto`
(`repvar1`) as a consistency check; 1 epsilon x 10 images each for
first-time `mnist_cnn_a`/`resnet2b` baselines, CIFAR-10 already cached
locally so no network-access risk materialized) completed successfully.

**Phase 7 -- results.** `NNs/aggregate_sweep_results.py` joins
`sweep_results.jsonl` with cheap structural features computed directly
from each sample's `.model` file (`NNs/structural_signature.py`,
factored out of the reconstruction scripts' duplicated parsing loop) --
full table in `NNs/sweep_summary.md`. The headline finding, now backed
by 5 epsilon points instead of 1:

| epsilon | unfused verified% | fused_v2 verified% |
|---|---|---|
| 0.02 | 90.0 | 90.0 |
| 0.05 | 70.0 | 50.0 |
| 0.1 | 20.0 | 10.0 |
| 0.15 | 10.0 | 0.0 |
| 0.2 | 0.0 | 0.0 |

At every epsilon, the fused structure is never *more* verifiable than
unfused, and is strictly worse at 3 of 5 points -- the two ties are
floor/ceiling saturation (both near-100% at the smallest epsilon, both
0% at the largest), not evidence the effect vanishes. `fused_auto`
(the automated-pipeline sample, structurally identical to `fused_v2` --
same op counts, same `axis=1` `Concat`/`Split`) reproduces `fused_v2`'s
exact result at their shared epsilon (10.0% each), a real consistency
check that the new automated pipeline (Phases 1-2) gives the same
answer as the original hand-traced reconstruction. `mnist_cnn_a`
verifies very well at its default epsilon (100%, 10/10, and fast --
mean 1.23s/image); `resnet2b` verifies poorly at the standard CIFAR
epsilon (0%, 10/10 timeout) -- both first-time numbers, descriptive
only (no fused counterpart exists to compare against).

This is now a real, epsilon-resolved, structure-attributable
verifiability effect -- exactly the kind of data the next step (a
custom `tensat` cost function that steers extraction toward more-
verifiable structures) needs as a starting signal: this one data point
says "prefer NOT introducing this channel-axis relu-merge fusion,"
though a single fusion pattern on one model is not yet enough to
generalize a cost function from -- the natural next step, not done in
this campaign, is applying the same Phase 1-4 pipeline to more models
with genuinely different fusable structures once found.

All raw results: `NNs/sweep_results.jsonl`, `NNs/sweep_summary.md`,
per-run logs under `NNs/sweep_logs/`. Manifest/driver:
`NNs/sweep_manifest.json` (generated by `NNs/build_sweep_manifest.py`),
`NNs/run_verification_sweep.py`.

## 2026-08-29: which multi-pattern rules actually fire, and the conv-weight fusion verified (neutral)

Follow-up to the campaign above, chasing a specific question: the
sweep's "fused" InceptionMNIST was a channel-axis **relu-merge**
(`Concat`->`Relu`->`Split` on activations), but is that the fusion
TENSAT actually *produces*, or just the one that happened to get
reconstructed? And does the "fusion hurts verifiability" headline
generalize, or is it specific to that one rule on one model?

**Instrumented rule-firing probe.** tensat's `run_one` multi-pattern
hook already prints, per rule per saturation iteration, a 4-tuple funnel
`(pairs, compatible, valid, cycle_ok)` -- candidate operand pairs found
-> shape-compatible -> pass `check_pat` validity -> pass the cycle check
(= actually added to the e-graph). Ran it (`NNs/multi_rule_match_probe.sh`,
logs `NNs/matchprobe_logs/`) on all 3 models with `-u -t
converted_multi.txt --iter_multi 15` (the stale Aug-24 debug binary
already emits this; matching behaviour is unchanged by the provenance
commit, so no rebuild). Findings, per model:

- `mnist_cnn_a`: totals `(14,12,0,0)`. Relu-merges find pairs and are
  shape-compatible, but **valid=0** -- rejected at `check_pat` (the relus
  aren't independent). Convs don't even match the `conv2d 1 1 0 0`
  pattern. Nothing fires.
- `resnet2b`: totals `(74,60,12,0)`. Relu-merges reach **valid=12 but
  cycle_ok=0** -- every fusion is rejected by the *cycle* gate (the
  same-shape relus are causally chained). Different failure than
  `mnist_cnn_a`, same outcome: nothing fires. (Flag-sensitive: this is
  under `--no_cycle` / `check_cycle_partial`; an older note recorded
  "thousands of cycle_ok" for its relu-merge under a different cycle
  flag -- worth a controlled recheck, but under the sweep's own flags
  nothing fused.)
- `inception_mnist`: the rules that actually FIRE are the **weight-side
  conv fusions** (two convs sharing an input -> concat their *weight*
  tensors on the output-channel axis -> one wider conv -> split the
  output). `cycle_ok` climbs into the thousands and the e-graph blew up
  to 7.6M nodes before hitting the time limit. The **relu-merge does NOT
  fire** here either (`valid=1, cycle_ok=0`).

Two corrections this forced. (1) The verified `fused_v2` (relu-merge,
`Conv` count unchanged at 3) is a *rare* straggler that only appears
under the `favor_fusion` discount (won ~2/60 in the earlier campaign);
the fusion that *robustly* fires is the conv-weight one, whose structure
had never been verified. (2) "InceptionMNIST has only ~3 structures" was
about what *extraction selected*, not the e-graph -- which is
combinatorially rich in conv-fusion variants. `--n_diverse`/`DiverseCost`
was also checked directly: it produces 15 byte-distinct files that are
all structurally identical (31 nodes, no `Concat`/`Split`) -- it
diversifies *terms*, not *structures*, because its penalty is per-enode
while structural novelty lives behind a whole different (costlier)
subtree (the term-uniform != structure-uniform problem, made concrete).

**Verified the conv-weight-fused InceptionMNIST.** Hand-built rather
than coaxed out of the extractor (the `favor_fusion` axis-0 penalty
targets exactly this weight concat -- BUGS #14 -- so forcing it would
fight our own cost model). `NNs/build_inception_convfused.py`: enlarge
branch-B's 1x1 kernel to 3x3 center-only (run at pad 1 this is *exactly*
the 1x1-pad-0 result, borders included), concat with branch-A's 3x3 on
the output-channel axis -> a constant `[16,8,3,3]` wide conv (the weight
concat folds away, never an ONNX op), one wide `Conv` -> channel `Split`
-> `Add`. Numerically identical to unfused (**3.6e-07**), same ReLU
neurons. Output `NNs/inception_mnist_convfused.onnx`.

The activation `Split` breaks auto_LiRPA's default **Patches** conv_mode
(`BoundSplit.bound_backward` calls `torch.cat` on a `Patches` object ->
`TypeError`, `slice_concat.py:311`) -- a *core-bound-pass* failure,
branching-independent, unlike the relu-merge's `Concat` which broke the
*branching heuristic*. Fixed with `general: conv_mode: matrix` (dense
bounds, mathematically identical, slower). Then, under matched settings
(smart branching + matrix mode), against the fair unfused control run
the same way (`NNs/abcrown_config_inception_mnist_unfused_smart_matrix.yaml`):

| model | verified | images | mean SAFE time |
|---|---|---|---|
| unfused | 30% (3/10) | [0,1,3] | 10.34s |
| conv-fused | 30% (3/10) | [0,1,3] | 11.47s |

**Verifiability-neutral**: identical verified count and identical
verified *set*, ~10% slower (the `Split` overhead). The apparent
"beats unfused" one might read from the sweep's 20% is an artifact of
branching method -- the sweep used `random`; under `smart` both models
reach 30%.

Interpretation. A semantics-preserving, same-ReLU-neuron rewrite cannot
change *fundamental* verifiability -- only whether the verifier's
implementation trips on an op. The conv-fused `Split` trips Patches mode
(recoverable -> neutral); the relu-merge's `Concat` trips smart branching
(forces `random`, and even at *matched* random it was 10% vs unfused
20%, so it is genuinely worse -- it restructures the ReLU relaxation, not
just the linear algebra). Net lesson for the cost function, sharper than
the campaign's single data point: avoid activation-path `Concat` *and*
`Split` (both trip the verifier, in different ways), while weight-side
fusion is safe-but-neutral. No fusion available IMPROVES verifiability;
the realistic value is defensive steering, or finding a rewrite that
changes the ReLU relaxation *favorably* (which none of TENSAT's current
rules do).

Artifacts: `NNs/build_inception_convfused.py`,
`NNs/inception_mnist_convfused.onnx`,
`NNs/abcrown_config_inception_mnist_convfused_{smart,random}.yaml`,
`NNs/abcrown_config_inception_mnist_unfused_smart_matrix.yaml`,
`NNs/abcrown_out_inception_convfused_smart.log`,
`NNs/abcrown_out_inception_unfused_smart_matrix.log`,
`NNs/multi_rule_match_probe.sh`, `NNs/matchprobe_logs/`.

## 2026-08-29 (cont.): more models, ArchDiverseCost, and the rewrite-verify reach limit

Three threads, all pointing at the same conclusion about where the method
can operate.

**More models + a verifier-hostility matrix.** Scouted VNN-COMP for models
that ship WITH verification specs (no synthesized tasks) and span diverse
op-types, dropping the earlier fusability filter (un-fusing/splitting needs
no parallel branches, so all architectures are in scope). Baseline-verified
7 models unmodified at their real specs (`NNs/run_vnncomp_baselines.py`,
`NNs/baselines_results.jsonl`, logs under `NNs/baseline_logs/`). The
hostility picture:
- *Not hostile* (verify fast): ffnnSIGMOID (sigmoid MLP, 3/3), mnist-net_256x2
  (pure FC, 3/3), resnet_4b (residual CNN, 3/3), resnet_2b (2/3, one hard
  instance).
- *Bound-hostile* (full-timeout): **tll** (min/max lattice, 600s) and
  **cgan** (ConvTranspose generative, 900s) -- loose bounds BaB can't close.
- *Pipeline-hostile* (won't load without special handling, but bounds fine
  once loaded): **vit** (transformer). Default pipeline fails on the
  onnx->pytorch trace (`tensor size mismatch`); ab-CROWN's official
  transformer settings fix it (`NNs/candidate_models/cfg_vit.yaml`:
  `softmax:'complex'`, the `customized_vit_tuning` hook, forward-before-
  bounds) -- then it verifies (unsat) in 34s. So attention's difficulty here
  is setup, not looseness. cgan and vit needed git-LFS pulls from the 2023
  benchmark repo (`NNs/candidate_models/exotic2023/`).

**ArchDiverseCost (committed in tensat `c20ba5d`).** Fixes DiverseCost's
term-not-structure failure by tracking per-enode rewrite provenance
(`RewriteWitness`: which multi-pattern rule created each enode) and
*rewarding* a target rule's witnesses so the fused representative wins its
e-class, rotating the target across samples. On InceptionMNIST: 3 distinct
structures (unfused + two conv-weight fusion variants) vs DiverseCost's 1;
targeting the conv-weight rule reliably produces FUSED at lower cost.
Non-fusable models report 0 witness families and fall back to baseline.

**Rewrite-and-verify test on the scouted models: blocked, on all 7.** The
actual deliverable came back a clean negative that bounds the method's reach.
A two-part barrier:
1. *TASO's ONNX importer is narrow and CNN-oriented.* It ingests Conv-based
   nets (resnet_2b came through clean after `NNs/normalize_for_taso.py`
   handled Flatten -- 6 Conv/6 Relu/2 Matmul, numerically identical), but
   SKIPS bare `MatMul` (wants `Gemm`), trips its reorder-assert on `Flatten`,
   and produces degenerate weights-only graphs for pure-FC nets (tll ->
   1 Input + 29 Weight, zero compute). `ConvTranspose` (cgan) and `Softmax`
   (vit) are hard-blocked. Painfully, tll -- the one bound-hostile model
   that's theoretically rewriteable (ReLU-composition min/max) -- is a
   TASO-ingestion casualty.
2. *The models that DO ingest have no rewriteable structure.* resnet_2b
   saturates (e-graph 50->183 nodes) but fires 0 fusion rules and extracts a
   graph with an op histogram byte-identical to the input -- residual nets
   don't restructure (relus causally chained). Verifying it would verify the
   same graph twice; skipped per the structural gate.

Net: TENSAT rewriting needs Conv-based models WITH parallel branches sharing
an input. Found VNN-COMP benchmarks essentially never have this (all
sequential/residual), and the non-conv ones TASO can't ingest -- so the only
model in this project that ever produced a genuine structural rewrite remains
the hand-built InceptionMNIST. Running rewrite-vs-verify at scale needs
either hand-built parallel-branch conv models (losing the bundled-spec
property) or a real extension of TASO's importer to FC/Gemm graphs.

Artifacts: `NNs/run_vnncomp_baselines.py`, `NNs/normalize_for_taso.py`,
`NNs/candidate_models/` (staged models, specs, configs, normalized+.taso),
`NNs/baseline_logs/`, `NNs/baselines_results.jsonl`.


## 2026-08-29 (cont. 2): min/max reassociation moves verifiability + un-curated corpus

**Light version of "step 2" (hand-authored min/max reassociation) -- SUCCEEDED.**
First positive result that a semantics-preserving rewrite changes verifiability.
A piecewise-linear function has many ReLU decompositions; re-associating a min/max
reduction tree (`max(u,v)=u+relu(v-u)`) changes ReLU *topology* while holding the
function AND total ReLU count fixed -- escaping the neutrality wall (which assumed a
fixed ReLU skeleton). Hand-built max-of-affine distribution (N=16, 20 reps,
auto_LiRPA), measuring the certified upper bound + unstable-ReLU count:
- chain (deep) certifies TIGHTER than balanced (shallow): **17/20** alpha-CROWN,
  **14/20** vanilla CROWN (not an alpha artifact), budget-robust (chain-tighter at
  200 iters too).
- Mechanism = ReLU *stability*: chain 8.35/15 unstable vs balanced 12.70 (fewer in
  every rep) -- the running max keeps later `relu(cand - runmax)` inactive/exact.
- Direction is opposite the naive "shallower=tighter" guess -> design rule: chain-ify.
- Structure-dependent: nearly vanishes (8/20) for the tll-shaped min-of-max lattice
  (caps reduction depth). Real tll couldn't be lifted -- it's a deep sequential chain
  of MatMul->Relu->MatMul bank blocks with min/max baked into weights, not a
  rebalanceable tree. Artifacts: `NNs/reassoc_results/` (maxtree_bounds.py, FINDINGS.md, logs).

**Heavy version (rerun TASO generation without the speed bias).** Key discovery:
TASO's generator (`taso/src/generator/generator.cc`) has NO speed filter -- it
enumerates verified equivalences, direction = DFS discovery order, depth <=3, op set
has no min/max. The bias lives DOWNSTREAM in the 660->119 curation (converted.txt was
660 rules at tensat commit d4e0811, cut to 119 at cde6d36). Recovered the full 660
corpus from git; 621 rules were curated out, but analysis shows the learned corpus has
NO from-scratch structure-creating rule (0 `split` rules; the 8 concat-creating rules
need a pre-existing `ewadd(op,op)` = two parallel ops). The ONLY structure creators
are 4 hardcoded `PRE_DEFINED_MULTI` conv-splitters (bind a 2nd weight via multi-pattern
matching; need a 1x1 conv). So even the un-curated corpus can't create parallelism in
sequential/residual nets -- confirming the barrier, not breaking it. The 660 is in a
pre-`f2109cc` dialect (matmul arity, context-dependent concat NDIM, changed enlarge
sig) too costly to faithfully migrate; used a direction-unbiased **bidirectional-119
(232-rule)** set instead (`tensat/rules_full_bidir.txt`, `bidir_rules.py`) and ran
arch-diverse extraction. **VALIDATED result** (structural_signature dedup over 8
extractions, each reconstructed to ONNX and numerically checked vs the reference):
**InceptionMNIST curated(119-fwd) -> 3 distinct structures**, all semantically
correct (max|ref-recon| = 7e-7): the unfused-like form (x6) + two conv-weight fusion
levels (the 4 hardcoded PRE_DEFINED_MULTI conv-splitters firing on the 1x1 branch).
**resnet_2b -> 1, mnist_tiny -> 1** (isomorphic to input -- barrier). Exactly the
corpus-analysis prediction: arch diversity appears only where parallelism pre-exists
(InceptionMNIST's branches); sequential/residual nets stay isomorphic because no rule
creates parallelism from a monolithic op.

**The direction-unbiased (bidirectional) experiment failed a soundness check --
RETRACTED.** Naive LHS<->RHS reversal of TASO's learned rules is UNSOUND in tensat's
untyped egg language: bidir(232) extractions reconstruct but are numerically WRONG
(max|ref-recon| = 9.83 vs 7e-7 for curated), i.e. a reversed rule unions
non-equivalent e-classes. So the "un-speed-biased corpus" could not be validly run by
reversal; the validated heavy result is the curated arch-diverse extraction above.
This does NOT change the headline: the corpus analysis (a static fact) already showed
even the full un-curated corpus contains no structure-creating-from-scratch rule, so a
valid un-biased run would still not rewrite sequential/residual nets.

Verifiability of the InceptionMNIST fusion variants: prior measurements are neutral
(weight-path conv-fusion, 30%=30%, convfused) to WORSE (activation-path concat/split,
strictly worse at 3/5 eps, sweep-headline). Not re-verified here (predicted range
already established; resnet_4b and inception-convfused not rerun -- same family /
hand-built variant of the same net, prior probes already isomorphic/neutral).
Fixed two real tensat bugs: rule-file trailing-newline parse panic, and a
multi-pattern cycle-check panic (`descendents.get(id).unwrap()`) on expanded rule sets
(tensat ddd6352, blacklist-flag corrected). Artifacts: `tensat/converted_full.txt`,
`tensat/rules_full_bidir.txt`, `tensat/bidir_rules.py`.

## 2026-08-30 — AC-closure rules: lattice's first verifiability win + TASO speed-assumption audit

**Context.** Prior sessions established that verifiability-aware extraction (VerifCost)
improves the maxout net (+20%) but the min-of-max lattice stayed pinned at the input
bound (8.50) under every approach. Today we found *why*, fixed it, and audited the
root-cause class in TASO.

**Chain-query diagnostic (`tensat --query_chain`, new in `src/main.rs`).** Added a
non-mutating `egraph.lookup` probe that, after saturation, asks whether the tight
left-deep *chain* association of the lattice is materialized — order-independently (a
per-group subset-closure over all leaf permutations), plus natural-order break-depth,
cycle-blacklist membership, and root-equivalence. On the lattice with the 621 rules the
chain is **absent**; the natural-order spine breaks at depth 2/7.

**A wrong turn, corrected.** First reading was "saturation budget / breadth-first
starvation" (it hit the 120s TimeLimit). A 10× budget re-run (1200s, 224k nodes) still
produced no chain — but the frontier *didn't move at all*, which crowding can't explain.
The advisor flagged the tell: `max(max(g0,g1),g2)` is **one** associativity step from the
input, so its absence isn't a budget story. A grep settled it: **the 621 rule set contains
no pure associativity and no commutativity for *any* commutative-associative op**
(ewmax/ewmin/ewadd/ewmul all 0) — only the *idempotent* shared-operand max rule. Confirmed
by swap-in: lattice + a 4-rule file with pure assoc+comm → `Saturated` at 612 classes,
full depth-7 chain present, root-equivalent. So it was a **rule-set gap**, vindicating the
original "missing rewrite rules" intuition; the breadth-first note was demoted to an
untested hypothesis (`EGRAPH_BREADTH_LIMITATION.md`).

**Root cause — TASO's generator is AC-blind (`taso-generator-is-AC-blind`).** Traced the
gap to the generator itself (absent pre-Z3 too). `generator.cc`'s `variable_ordering` +
`same_via_subst` + common-sub/supergraph pruning canonicalize associative/commutative
operators, so pure assoc/comm never surface as distinct-graph pairs; the idempotent rule
survives only because its leaf multiset differs (a genuine simplification). TASO was built
to find runtime-*reducing* rewrites, and AC-rearrangements are runtime-neutral — exactly
the class verifiability needs (chain vs balanced ReLU tree: same runtime, different
certified bound). Systematic, not accidental.

**Fix + rerun on all models.** Hand-authored 12 AC rules (assoc both directions + comm for
ewmax/ewmin/ewadd/ewmul), **all Z3-verified**, deduped-unioned with the 621 →
`pwl_rules_ac.txt` (632 rules). verif_cost extraction → reconstruct → α-CROWN bound:
- **maxout: 9.6236 (5/120 unstable) vs input 12.0257 = +2.40 (20%)** — new best.
- **lattice: 7.6167 (8/120) vs input 8.5019 = +0.89 (10.4%)** — the min-of-max's *first*
  real verifiability improvement (was a documented null). Both numeric-gated (~7e-7).
- The 4 Conv/Matmul nets (mnist_tiny, cnn_a, resnet2b, inception) stay **inert** — each
  collapses to ≤1 distinct structure; commutativity gives mirror-identical ReLU topology
  and 2-operand residual adds have no ≥3-leaf chain to associate → bounds unchanged. The
  AC lever is specific to min/max-reduction-shaped models.

**TASO speed-assumption audit (`TASO_SPEED_ASSUMPTIONS.md`).** Documented six sites where
TASO assumes rewrites serve speed — as both issue points and insertion points for
verification-centric optimization. Deepest finding: `Graph::optimize` (`ops.cc:441`)
updates the best graph only on a **strict runtime decrease**, and the α-threshold
(`ops.cc:466` → `substitution.cc:1057`, default α=1.0) prunes anything not strictly faster
— so TASO's native search **can never return a cost-neutral (verifiability) rewrite**,
independent of the rule set. This is the structural reason the project runs on tensat/egg
(equality saturation keeps all equivalent forms) with our own VerifCost extraction. Other
sites: the cost oracle (`total_cost` = summed `cudaEventElapsedTime` GPU ms), the fusion
xfers, the generator AC-blindness (#4), and the speed-only public API.

**Artifacts:** `tensat/src/main.rs` (--query_chain); `NNs/reassoc_results/`:
`pwl_rules_ac.txt`, `ac_rules_raw.txt`, `ac_rules_verified.txt`, `pwl_rules_plus_assoc.txt`
(superseded), `ac_{maxout,lattice}_verif.onnx`, `lat_{union,assoc}_verif.onnx`,
`TASO_SPEED_ASSUMPTIONS.md`, `EGRAPH_BREADTH_LIMITATION.md`, updated `VERIF_COST_RESULT.md`.

## 2026-08-30 (cont.) — tll: first real-world (VNN-COMP) verifiability win via semantic lift

Turned the AC-closure result loose on a real model. The VNN-COMP **tll** (Two-Level
Lattice) ONNX is TLL-compiled to a sequential MatMul/Add/Relu MLP — min/max baked into
weights, no ops — so a mechanical importer would leave it inert (my initial "rules
transfer directly" claim was wrong; caught before building). Instead **semantically
lifted** it: read the 16 local affine fns (linearLayer), the one-hot selection (16 groups
× 16 members), and the min/max banks; rebuilt the explicit **max_g min_k (W_k·x+b_k)**
lattice with tll's real weights (`NNs/build_tll_lattice.py`, numeric-gated 4.8e-7).

Pipeline: build_tll_lattice → taso ingest → tensat (`pwl_rules_ac.txt`) n_diverse →
reconstruct(→relu) → α-CROWN, box x0=0/eps=1.0. **Baseline (original compiled tll) cert_ub
19.59 (628/1020 unstable) vs best lifted+reassociated 8.26 (89/904) = +11.33, 58% tighter
on the same function.** Honest decomposition: the LIFT does ~48% (compiled TLL relu-gadgets
are far less stable than an explicit lattice — 628 vs ~90 unstable), and REASSOCIATION (the
general tensat/AC-closure contribution) refines ~10.06→8.26 (~18%); both lattice levels are
reassociable here so the ewmin AC rules are load-bearing for the first time.

Fixed the *actual* tll ingestion barrier: **taso's `MatMul`-casing bug** — the importer
registered only lowercase `'Matmul'`, so standard-ONNX `MatMul` nodes were all skipped,
degenerating pure-FC graphs to inputs+weights (the long-standing "tll degenerate" mystery;
BUGS.md). Residual, separate: taso's SGEMM cost-measurement aborts on small-N matmuls
(tll's width-1 output) — the vector trick sidesteps it; documented. Artifacts:
`taso/python/taso/__init__.py` (MatMul alias), `NNs/build_tll_lattice.py`,
`NNs/derive_weight_names_baseline.py` (empty-param guard), `NNs/reassoc_results/TLL_RESULT.md`,
tll_lattice/recon onnx + sidecars.

## 2026-08-30 (cont. 3) — Redundancy pruner + relaxed regeneration: recover what the quotient dropped

Concern raised: TASO's rule generator systematically drops cost-neutral (AC) rewrite
families by design (canonicalization/pruning) — the exact families verifiability needs
(chain vs balanced = same runtime, different bound). Built the pipeline to recover them.

**Redundancy pruner (`tensat -m redundancy`).** Greedily removes any rule whose LHS=RHS is
re-derivable from the other kept rules within `--redundancy_iters B` e-graph iterations, in
tensat's own sound engine (grounds vars to fresh [4,4] Inputs, saturates the rest, checks
e-class equality). Sound (only removes); `B` is the reachability/minimality knob. Validated
on AC → keeps the minimal 3-rule generating set (assoc-L is derivable via comm+assoc-R). On
the 632-rule pwl_rules_ac corpus: 515 pruned → **117 core (82% redundant)**.

**Generator quotient relaxation.** Made generator.cc's four quotient checks env-toggleable
(RELAX_SUBGRAPH/SUPERGRAPH/VARORDER/SUBST). RELAX_SUBST (drop the renaming-dedup) is the
lever that re-emits the AC family. Found a hard limit: standalone binary commutativity is
never generated even fully relaxed — it's an ENUMERATION-ORDER artifact (`k=j+1` in the DFS
builds one operand order per commutative op), upstream of the filters. So associativity is
only ever emitted in a canonical operand order, and commutativity not at all.

**Full depth-3 relaxed pipeline (all cost-neutral families).** generator all-relaxed depth-3
(**849,839 transfers**) → pb2egg (36,976) → NEW prededup.py alpha-dedup (3,757, the safety
valve) → Z3 verify (2,658) → redundancy-prune → **1,097-rule minimal core**. The core has
34 ewmax + 34 ewmin reassociation rules the original 621 had ZERO of — recovered the
verifiability family autonomously. Honest gaps: core alone did NOT improve the lattice
(binary-comm gap → reassociation can't fire); the 12 hand-AC rules restore it.

**Infra saga (recorded so future sessions don't repeat it).** The container's `apptainer
--nv` CUDA broke cluster-wide (Cuda-35 / cudaErrorInsufficientDriver) on 2026-08-30 evening,
still broken 08-31 — a driver-injection skew (container .sif intact, native torch CUDA
works, both l40s and rtx6k affected). Report written for cluster support. Workaround: the
prune needs only shape inference, so a CPU-linked tensat (built against taso/build,
USE_CUDA=OFF) runs it GPU-free — validated identical to the GPU build.

## 2026-08-31 — pb2egg full-op coverage, first pipeline tests, and the axiom-verifier find

**Root-caused why conv/matmul models look inert: pb2egg was clean-only.** It silently
dropped ~84% of the generator's output — ALL conv/pool/concat/structural rules
(converted_full660: 553/660 non-clean). So tensat never received conv rewrites; the "conv
inert" conclusion was an artifact. (Original TASO/tensat "got by" because taso_rules.txt is
a HAND-COMMITTED static file; -m convert just reformats it. pb2egg fills a real automation
gap — it was just scoped too narrowly.)

**Extended pb2egg to full-op (Tier-1: conv2d/pool/concat).** Each op's exact egg child order
taken from tensat model.rs make() — NOT the Mdl comments or converted_full660 (BOTH stale).
Orders differ per op (conv2d params-first, pool INPUT-first, concat params-first + variable
arity). Verified with a new **`tensat -m parse_check`** oracle (the authoritative
"does this parse as Pattern<Mdl>" check). On the original taso/graph_subst.pb: 48 → 116
rules, 0 non-clean dropped, all parse. z3_verify_egg got uninterpreted entries for the new
ops so it doesn't crash (stopgap).

**Started a test culture (NNs/tests/run_tests.sh, plain-assert, no pytest).** (1) regression:
non-clean not dropped (fails on old pb2egg — 72 dropped, conv=0); (2) parse-validity: every
emitted rule parses (catches op-format drift forever); (3) reproduction/coverage from the
original pb. 8/8 pass; demonstrated the regression test catches the old bug.

**Discovered tensat ALREADY has an axiom verifier — DON'T reinvent** ([[tensat-already-has-axiom-verifier]]).
`tensat -m verify` (README's prove_taso_rules) is a GPU-free egraph axiom-saturation
verifier: verify() in lib.rs uses `Runner<Mdl,(),()>` (no analysis, no GPU), adds all rule
pairs, saturates with `rules()` (~40 bidirectional axioms in rewrites.rs), checks e-class
equality; ~30x faster than rule-by-rule. `rules()` already has matmul assoc/linear, conv
bilinear, matmul/conv-over-concat (grouped conv), enlarge, pooling, transpose, identities —
AND the activation-unfolding axiom (`operator-commutativity-4: conv acti=2 => relu(conv
acti=0)`; relu is acti=2). It LACKS ewmax/ewmin — that's the complementary reason z3_verify
exists. Clean unification (future): add min/max axioms + the max/min↔relu bridge to rules()
for one GPU-free verifier. So: for conv/matmul use `-m verify`, not the uninterpreted z3 path.

## 2026-08-31 (cont.) — Migrated the ~6-year-stale rules() axioms to current arities; `-m verify` restored

- **Root cause of the stale axioms (answered):** rules() was written 2020-06 (Remy Wang)
  against the op arities of the day (2-arg matmul, 1-arg transpose, 3-arg concat, 6-arg
  params-first pool). The Mdl `define_language!` grew params ~6 weeks later
  (f2109cc 2020-07-16 matmul+activation → [Id;3]; 86a2617 2020-07-31 transpose+perm/shuffle
  → [Id;3]; concat+ndim, pool→7-arg input-first, enlarge→2-arg ref-based) to let the
  OPTIMIZER represent real models. rules() is verify-only (the optimizer uses
  `rules_from_str`, always current), and `-m verify`/`prove_taso_rules` is off by default
  (README: "uncomment it"), so the drift went unrun and undetected for ~6 years. Dead code rots.

- **Migration (tensat, src/rewrites.rs `rules()`).** Guiding fact: `verify()` is PURE-EGG
  (`Runner::<Mdl,(),()>`), so an axiom only needs to PARSE and be a TRUE universal identity —
  `make()`/shape checks never run. So free-var params are sound ONLY where the identity is
  param-agnostic. Changes:
  - **matmul** → literal activation `0` (associativity/linearity hold only with no fused relu).
  - **concat** → free rank var `?n` (identity holds at any rank; concat preserves rank, so the
    same `?n` threads through nested concats).
  - **pool** → input-first order `[in,kh,kw,sh,sw,pad,acti]`; free `?c` on the concat-distribution
    rules, literal `0` on the two conv-equivalence (Cpool/pooling-by-conv) rules (avgpool==conv
    only with no activation).
  - **transpose** → free perm `?p` + shuffle `?s` (Name/Scalar leaves are written as pattern vars);
    only the *distribution over elementwise* axioms — sound for any perm.
  - **Added** the matmul relu-unfold `(matmul 2 ?x ?y) <=> relu(matmul 0 ?x ?y)` (couldn't exist
    in 2020 — matmul had no activation param). conv2d axioms were ALREADY current (6-arg
    params-first) and left untouched.
  - **DROPPED (unsound or unexpressible in pure-egg — documented in-file so they're not "recovered"):**
    transpose-is-its-own-inverse / matmul-and-transpose / concatenation-and-transpose (2D-transpose /
    involution-specific; a free perm asserts them for ALL perms = false); split-definition-0/1
    (split_0/1 now unary + only conditionally true); enlarge-convolution-kernel (2-arg ref-based now);
    and two INVERSE directions (`-concatenation-and-matrix-mul.-1`, `-concatenation-and-conv.-2`)
    whose RHS reintroduces a concat whose rank `?n` is unbound by the elementwise LHS (egg rejects it;
    the forward directions are kept).

- **Validated (NNs/tests/run_tests.sh Test 4 — the permanent guard):**
  - rules() constructs with no panic → every axiom parses at current arity.
  - min/max axioms still `Proved 8/8`.
  - orig 116 current-arity rules → `Proved 109/116`. The 7 gaps need left-argument
    matmul-distributivity / the other-argument concat-matmul / grouped-conv-merge (the
    multi-pattern enlarge/merge machinery) — axiom-set gaps, not regressions.
  - **5 known-FALSE negative canaries (`verify_canaries_false.txt`) ALL rejected** — the soundness
    guard: matmul/concat arg-swap, transpose-drops-input, matmul acti-swap, pool kernel-swap.
  - Full harness: 11/11 pass.
- New files: `NNs/reassoc_results/verify_canaries_false.txt` (functional `==` notation, whitespace-free
  per equation.pest), `NNs/sexpr_to_functional.py` (S-expr `=>` → functional `==` converter, used to
  feed orig_full_egg.txt to `-m verify`), `NNs/reassoc_results/orig_full_functional.txt`.
- Build note: the container's `/opt/cargo` registry isn't persisted in the .sif; build with
  `CARGO_HOME=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/cargo_container` + `--offline` (see
  `build_verify.sh`). Ran inside the existing cpu-g2 allocation via `srun --jobid=… --overlap`.

## 2026-08-31 (cont. 2) — pb2egg multi-output save + pre-prune retention; full pipeline reran

- **pb2egg now SAVES multi-output rewrites** (previously dropped). Rules with
  len(mappedOutput) != 1 have no single-pattern egg form (dst produces several outputs via
  `split`); pb2egg writes the exact source rules to a filtered RuleCollection protobuf
  (`--multi-out`, default `<out>.multi.pb`) for the multi-pattern lane to consume later.
  Single-output emission is unchanged.
- **New pipeline driver `NNs/run_rule_gen.sh`** runs the whole thing in one command
  (generator all-relaxed depth-3 -> pb2egg +multi save -> pre-dedup -> Z3-verify -> prune),
  every stage output durable. The Z3-verified set (`relaxed_d3_verified.txt`) is retained as
  the **"all rules pre-prune"** snapshot for future learned pruning.
- **Full rerun (job 39389879, cpu-g2, 32 min, COMPLETED):** reproduces the prior end-to-end
  numbers exactly AND preserves the multi-output family:
  | stage | count |
  |---|---|
  | generator transfers | 849,839 |
  | pb2egg single-output egg | 36,976 |
  | **multi-output saved (relaxed_d3_egg.multi.pb)** | **798,729** (all mappedOutput=2; loads clean) |
  | pre-dedup (alpha) | 3,757 |
  | Z3-verified (pre-prune) | 2,658 (min/max 1,440; 1,099 false-positives rejected) |
  | redundancy-prune (budget 4) -> core | 1,097 (1,906 groundable, 1,561 pruned; 752 non-PWL kept) |
  The tracked text artifacts (dedup/verified/core) came out byte-identical to the committed
  versions -> the axiom migration + multi-output save did not perturb the single-output pipeline.
- **Git note:** the two large binaries (`relaxed_d3_graph_subst.pb` 173 MB, `relaxed_d3_egg.multi.pb`
  164 MB) exceed GitHub's 100 MB limit -> gitignored; they live durably on gscratch and are
  regenerable via the driver. A tracked `.multi.pb.README` records the path, count, and load recipe.
- Fix folded in: the driver's PYTHONPATH now includes `toolchain-tensat/z3pkg` (the first rerun
  failed stage 4 on `ModuleNotFoundError: z3`).

## 2026-09-02 — Second CROWN door: a min/max-FREE exact rewrite that tightens full CROWN

Goal: manually rewrite a plain-ReLU network — **no min/max** — so its **full CROWN** bound
(not just IBP) gets tighter. This directly challenges the prior finding
(`plain-relu-rewrites-cant-move-crown-bound`): rearranging a plain-ReLU net's linear skeleton
is CROWN-neutral, and "min/max reassociation is the only CROWN door."

- **The hole in the neutrality induction.** That argument fixes the *neuron set* (up to
  nonneg-monomial relabeling). It says nothing about equivalent nets with a **different number
  of ReLU nodes whose pre-activations are linearly dependent** — i.e. **redundant ReLU
  structure**. Collapsing that redundancy is exact, min/max-free, and *does* move a CROWN bound.
- **Mechanism.** CROWN relaxes each unstable ReLU independently, with slack `|coeff|·gap`. Two
  duplicated neurons sharing pre-activation `z` but feeding the output with coeffs `c1, c2` cost
  `(|c1|+|c2|)·gap`; the merged single neuron (coeff `c1+c2`) costs `|c1+c2|·gap` — strictly less
  iff `sign(c1)≠sign(c2)` (a coefficient cancellation the duplicated form can't see). Survives
  **CROWN-Optimized**: the loosening lives in the ReLU chord's constant offset `−lu/(u−l)`, which
  α doesn't control (α narrows the gap but can't close it).
- **Two exact rules** (both demonstrated):
  1. **Merge proportional neurons** — rows `w`, `βw` (β>0) feeding `c1, c2` → one neuron, coeff
     `c1+βc2`. (This is what e-graph hashconsing does for free.)
  2. **Collapse complementary pairs** — `relu(−z)=relu(z)−z`, so
     `c1·relu(z)+c2·relu(−z) = (c1+c2)·relu(z) − c2·z`: two unstable ReLUs → **one ReLU + a linear
     (skip) correction**. The natural case (a two-sided feature), no artificial duplication.
- **Measured on the real verifier** (auto_LiRPA in the abcrown venv, 2-hidden MLP 8→16→·→1,
  ε=1.0; downstream coeffs **pinned moderate** cA=1.0, cB=∓0.8 → net 0.2, ratio 0.111, identical
  across pairs — deliberately not drawn, so no pair gets a near-zero net coeff that would inflate
  the headline):

  | rule | full **CROWN-Optimized** (opposite-sign) | same-sign control |
  |---|---|---|
  | 1 merge proportional | 88.6 → 44.8 = **−49.5%** | exactly neutral |
  | 2 collapse complementary | 124.9 → 98.9 = **−20.8%** | CROWN exactly neutral |

  (IBP/CROWN also tighter: rule 1 −62.3%/−59.4%, rule 2 −26.5%/−33.4%.) Forms function-identical
  on 50 samples; planted neurons genuinely unstable (pre-act gaps ~8–26); duplicates are **distinct
  rows** (IBP's large gap confirms no auto_LiRPA node-sharing). The **same-sign controls being
  exactly CROWN-neutral** is the load-bearing evidence that the effect is coefficient cancellation,
  not net-shrinking. Rule 2's same-sign IBP is *looser* (skip re-boxing) but CROWN-exact —
  confirming a genuine CROWN-relaxation effect, not an IBP artifact.
- **Honest scope.** The net is *constructed* to contain the redundancy compiled/exported nets
  exhibit (the `tll` lift's ~48% gain was de-compiling exactly this), then rewritten — a manual
  mechanism demonstration, not a rewrite found on an off-the-shelf model. Rule 2 is
  syntactically detectable (rows `w` and `−w` in a layer) and Z3-verifiable → a clean bridge to
  tensat automation (a rewrite rule + a CROWN-gap extraction cost), **not done here**.
- **Refines** `plain-relu-rewrites-cant-move-crown-bound`: "min/max is the only door" held only
  for *canonical/irreducible* nets; **redundancy-collapse is the second CROWN door**, min/max the
  first. Same "claim holds, with a named exception" pattern as the un-fusion correction.
- **Artifacts** (untracked in `reassoc_results`, per convention): the demonstrator
  `NNs/reassoc_results/crown_redundancy_collapse.py` and writeup
  `NNs/reassoc_results/CROWN_REDUNDANCY_RESULT.md`.

## 2026-09-02 (cont.) — CReLU-collapse: the CROWN improvement replicated on REAL trained nets

Followed the redundancy-collapse toy with the hard version: a **real** (not hand-crafted)
model, **no min/max**, improving **full CROWN-Optimized**, and — the anti-cheat constraint —
**replicating across independent trainings** so coincidental weight values can't be abused.

- **The vehicle is forced.** An exact non-min/max rewrite needs redundant ReLU structure, and
  standard training destroys *exact* redundancy — so no off-the-shelf plain-ReLU benchmark can
  satisfy the goal; the redundancy must be **architectural**. **CReLU** (Concatenated ReLU,
  Shang et al. ICML 2016; `CReLU(z)=[relu(z),relu(−z)]`, motivated by nets naturally learning
  opposite-phase filter pairs) is the canonical published activation with it — present in every
  training regardless of weights/task, which is exactly what makes the improvement replicate.
- **The rewrite** (exact, pure ReLU algebra, no min/max): per CReLU layer,
  `W₊·relu(z)+W₋·relu(−z) = (W₊+W₋)·relu(z) − W₋·z` (since `relu(−z)=relu(z)−z`) → **half** the
  unstable ReLUs + cascading DenseNet-style linear skips. CROWN relaxes the baseline's two copies
  independently (`(|W₊|+|W₋|)·gap`) vs the collapsed `|W₊+W₋|·gap`.
- **Measured** on real auto_LiRPA (CROWN-Optimized margin-lb spec), MLP
  784→CReLU(64)→CReLU(64)→10, 100 correctly-classified test images per training:

  | training | test acc | verified base→coll | mean per-img margin Δ (min) | 100% improved |
  |---|---|---|---|---|
  | MNIST seed 0/1/2 (ε=0.05) | 0.94 | 20→29, 19→25, 24→33 | +0.99/+1.15/+1.07 (min +0.24–0.38) | ✓✓✓ |
  | FashionMNIST 0/1/2 (ε=0.03) | 0.84 | 59→62, 49→54, 56→61 | +0.21/+0.24/+0.22 (min +0.03) | ✓✓✓ |

  **All 600 per-image CROWN-Optimized bounds are strictly tighter** (min Δ>0 in every training);
  verified accuracy rises in all 6. Exact float64 gate (~3e-7); 256→128 BoundRelu coordinates
  (auto_LiRPA does *not* share the pairs — the false-neutrality trap is dead); per-layer
  cancellation `|W₊+W₋|/(|W₊|+|W₋|)` ~0.63–0.70 (near the 0.71 random expectation, slightly below
  — the mechanism trace). Pilot confirmed the same under plain CROWN and that the delta grows with ε.
- **Constraints, checked:** (1) real — genuinely trained (81–94% acc), weights never hand-set;
  (2) no min/max — `relu(−z)=relu(z)−z`; (3) full CROWN-Optimized — the reported metric; (4)
  replicates across 6 independent trainings (2 tasks × 3 seeds) ⇒ architectural, not a
  coincidental-weight artifact. **Honest scope:** not a CROWN *theorem* (the measured 600/600
  distribution is the evidence); MLP only (CNN is the follow-on); the architecture was *chosen
  because* it instantiates the mechanism (surfaced, not hidden — the induction forces it, and
  constraint 4's replication is the direct answer to the weight-coincidence concern).
- **Artifacts** (untracked in `reassoc_results`): `NNs/reassoc_results/crelu_pilot.py`,
  `crelu_replicate.py`, writeup `CRELU_CROWN_RESULT.md`.
