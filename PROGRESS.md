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

## 2026-09-03 — Certified neuron-merging by row-proportionality snapping (the 7-step recipe)

User recipe: (1) MLP, (2) verify CROWN, (3) find near-proportional weight-row pairs
`row_j≈β·row_i`, (4) **snap** to exact proportionality, (5) **merge** the two now-proportional
ReLU neurons in the next layer, (6) reverify, (7) **certify** the snap changed nothing significant.
Executed end-to-end; `NNs/reassoc_results/snap_merge_pipeline.py` (+ probes `snap_merge_probe.py`,
`snap_merge_probe2.py`), real auto_LiRPA CROWN-Optimized. Writeup `SNAP_MERGE_RESULT.md`.

- **Three nets, exact algebra:** `orig →(lossy snap A1[j]:=β·A1[i])→ snapped →(exact merge:
  drop j, A2[:,i]+=β·A2[:,j])→ merged`; for β>0 `relu(βz)=β·relu(z)` so snapped≡merged (float64
  gate ≤1e-8). **Step-7 = a composed certificate for the ORIGINAL net:** the snap perturbs only
  `z_j`, bounded over the ε-box by `d_j=|r·c+r_b|+Σρ_k|r_k|` (r=residual row); 1-Lipschitz ReLU
  propagation gives `δ_m ≤ d_j·(|C||A3||A2[:,j]|)_m`; since `margin_orig ≥ margin_snap−δ` pointwise,
  **`lb_merged − δ` is a sound lower bound for the unmodified original net**. Headline metric =
  `(lb_merged−δ)` vs direct `CROWN-Opt(orig)`. δ-soundness **empirically validated**: 1000 random
  box points/image give `max|Δmargin|−δ ≤ 0` everywhere (worst +0.000).

- **Finding 1 (real nets, NEGATIVE — the answer to the recipe on a real net).** Min row-pair
  residual `‖row_j−β·row_i‖/‖row_j‖` across training regimes never drops below ~0.40: vanilla
  0.66 (H=64/128/256 all ~0.66–0.70), dropout-0.5 0.45–0.55, wd-1e-3 0.61, dropout+wd 0.40–0.45,
  long/small 0.62. Width doesn't help (near-orthogonal high-dim rows); dropout/wd help marginally.
  Full pipeline on vanilla: **0 snap-candidates (res<0.20) at every ε** — the "best" pair only
  prunes a low-impact neuron (β≈0.009). Crossover: a pair pays only around **res ≲ 0.02**, ~20×
  below the real-training floor. **On standard MLPs the technique does not fire: the required
  structure is absent.**

- **Finding 2 (POSITIVE, pipeline machinery).** On a net with structure *planted* (soft
  proportionality penalty, labelled a pipeline existence proof — NOT a real-net claim), 4 pairs at
  res=0.001/β≈1.0. The merge tightens full CROWN on average at every ε (step-5 isolation mean >0;
  per-image min >0 for ε≥0.05, −0.0000 α-noise at ε=0.03), δ negligible (~0.009), and the composed
  certificate **beats direct CROWN on the ORIGINAL net on 57/60 images at ε=0.08** (mean +0.0275),
  growing with ε. Caveats (honest): δ adds linearly per pair so at small ε fewer pairs pay (ε=0.03:
  1 pair 36/60 net-positive, 4 pairs 19/60 net-negative — tune pair count to ε); images where
  cert<lb_orig stay sound via `max(lb_orig,cert)`.

- **Relation to prior work:** the approximate / certified-surrogate cousin of the exact
  CReLU-collapse (`CRELU_CROWN_RESULT.md`) — same CROWN door (collapsing linearly-dependent
  unstable ReLUs), but the proportionality is inexact and the function change is **certified** and
  folded back into a valid bound for the unmodified original model (the delta over the
  compression/merging literature, which accepts uncertified error). δ uses the sound-but-loose
  1-Lipschitz `|W|` propagation; a tighter δ widens the crossover but not the ~20× to reach 0.4.
- **Artifacts** (untracked in `reassoc_results`): `snap_merge_pipeline.py`, `snap_merge_probe.py`,
  `snap_merge_probe2.py`, writeup `SNAP_MERGE_RESULT.md`.

---

## 2026-09-04 — CROWN cancellation probes, and the "hull-preserving ⇒ plain-CROWN-neutral" wall

Four minimal auto_LiRPA nets (real abcrown `.venv`, methods {IBP, CROWN, CROWN-Optimized}) pin down
*exactly* when a semantics-preserving rewrite can move a certified bound, and — importantly — expose a
tie artifact that had briefly made a reparametrization look like a plain-CROWN mover. Scripts are
ephemeral (session scratchpad); net definitions below reproduce them.

**Net 1 — linear residual `(-I)x + x ≡ 0`, no ReLU.** CROWN exact `[0,0]` at any ε; IBP loose
`[-2ε, 2ε]`. CROWN carries symbolic input coefficients so `-I + I` cancels before any interval is
taken; IBP forgets the two `x`'s are one variable. Purely-linear ⇒ CROWN exact.

**Net 2 — twin ReLU `relu(a) - relu(a) ≡ 0` (a ∈ [-2,2]).** Tracer CSE gotcha: the two `torch.relu(a)`
common-subexpression-eliminate to **one** `BoundRelu` (net backward coeff `+1-1=0`), so CROWN returns
exact `[0,0]` — *not* by seeing through two ReLUs. Force two independent nodes (route each relu through
its own identity `nn.Linear`): plain CROWN `[-2,+2]`, CROWN-Opt `[-1,+1]` (α→0.5). This is the minimal
redundancy-collapse instance: merging the duplicated unstable ReLUs (or BaB-splitting the one neuron)
recovers `[0,0]`. A "duplicate-ReLU" probe silently collapses under CSE unless nodes are forced apart.

**Net 3 — `c = a - relu(a) = min(a,0)`, true range `[-2,0]`.** On the box `[-2,2]`: IBP `[-4,+2]`,
plain CROWN `[-2,+2]`, **CROWN-Opt `[-2,0]` exact** (α→1 flattens `c ≤ a-a = 0`). Single-ReLU concave
function: α-optimization alone recovers exactness, no rewrite needed.

**Net 3b — SAME function, other decomposition.** `min(a,0)` spelled `-relu(-a)` traces to a
monotone-unary chain (`-a→relu→neg`), exact with no relaxation, so **IBP, CROWN, CROWN-Opt all give
`[-2,0]`**. So `a-relu(a) ⇝ -relu(-a)` turns an IBP-loose net **IBP-exact** — a genuine IBP win.
Tooling gotcha: `torch.clamp(x,max=c)` traces to `BoundHardTanh` and crashes at build
(`forward() missing min_val/max_val`) in this checkout; spell clamps as `minimum` / `-relu(-)`.

**The tie artifact (the reason this section exists).** On the `[-2,2]` box the two spellings of
`min(a,0)` gave *different* plain-CROWN bounds (`a-relu(a)→[-2,+2]` vs `-relu(-a)→[-2,0]`), which
looked like a plain-CROWN-moving rewrite and motivated a resnet2b "ReLU-flip" plan
(`relu(z)=z+relu(-z)`). A 3-box control killed it:

| box (a-range) | IBP `a-relu(a)` vs `-relu(-a)` | plain CROWN | CROWN-Opt |
|---|---|---|---|
| `[-2,2]`  (l=−u, tie) | `[-4,2]` vs `[-2,0]` — DIFFER | `[-2,2]` vs `[-2,0]` — **DIFFER** | `[-2,0]` both — SAME |
| `[-1,2]`  (u>\|l\|)   | `[-3,2]` vs `[-1,0]` — DIFFER | `[-1,0]` both — **SAME**       | `[-1,0]` both — SAME |
| `[-2,1]`  (\|l\|>u)   | `[-3,1]` vs `[-2,0]` — DIFFER | `[-2,1]` both — **SAME**       | `[-2,0]` both — SAME |

Plain CROWN differs between the spellings **only at the exact tie `l = −u`**, where auto_LiRPA's
adaptive slope heuristic `α = 1[u > |l|]` returns 0 (the wrong extreme). Off-tie the flip is a plain-
CROWN **no-op**. IBP, by contrast, differs in every box — its looseness is the correlated subtraction,
independent of the tie.

**Structural consequence (the wall).** On a fixed network the *only* difference between plain CROWN
and CROWN-Optimized is the ReLU lower-slope α, and the plain heuristic picks α as a function of each
neuron's convex hull `[l,u]`. Therefore **any hull-preserving reparametrization — flip
`relu(z)=z+relu(-z)`, positive scaling, duplicate-and-average — is *exactly* plain-CROWN-neutral off
the measure-zero tie set**, and CROWN-Opt-neutral always. A real trained net never sits at `l=−u`, so
there is no reparametrization of stock resnet2b that beats plain CROWN. **To beat plain CROWN a rewrite
must change the hull** — via multi-neuron cancellation (redundancy-collapse, `CRELU_CROWN_RESULT.md`)
or a reassociation that changes which intermediate bounds get computed (`REASSOC` results). Both need
structure resnet2b lacks natively (no duplicated/complementary unstable ReLUs, no min/max trees). This
retires the "flip beats plain CROWN, ties CROWN-Opt" target as a tie artifact **before** spending a
compute allocation on it. (Memory: `crown-relu-cancellation-probes`.)

**Addendum — snap-merge Finding 1b (CNN-as-MLP).** Extending the 2026-09-03 snap-merge negative
(real MLP rows never approach proportionality, residual floor ~0.40) to convolutional structure: an
im2col/CNN-as-MLP view of a trained conv net shows the same floor (channel-kernel residual ~0.42),
so the technique does not fire on CNNs either — the absent-structure negative is robust across
architecture families, not an MLP-specific artifact. (Commit `96e71f7`.)

### Follow-up — the `[-2,1]` box, and locating the *real* (IBP) win

Dissecting the one box where plain CROWN is loose but not at a tie: on `c = a-relu(a)`, `a∈[-2,1]`
(`l=-2,u=1`), plain CROWN returns `[-2,+1]` (true `[-2,0]`). The loose end is the **upper** bound.
Neuron slope `α = 1[u>|l|] = 1[1>2] = 0`, so the lower ReLU envelope is the flat line `relu(a)≥0`;
the upper bound of `c` (coeff `-1` on relu) uses it: `c ≤ a - 0 = a`, `max_{[-2,1]} a = +1`. The `+a`
identity term has nothing to cancel against because `α=0` flattened the relu term. The heuristic's
`α=0` is the *area-minimizing* choice for the neuron in isolation, but it is exactly the *wrong* slope
for the downstream `-1` coefficient — the heuristic can't see that coefficient. CROWN-Opt does, picks
`α=1`, gets `c ≤ a-a = 0` → `[-2,0]`.

`min(0,a)` (literal `torch.minimum`) and `-relu(-a)` inherit the SAME plain-CROWN `[-2,1]` — the flip
buys nothing off-tie (verified: all three spellings → plain CROWN `[-2,1]`, CROWN-Opt `[-2,0]`). The
**only** method the `min`/`-relu(-a)` spelling helps is **IBP**: `a-relu(a)` gives IBP `[-3,1]`, while
`-relu(-a)` and `min(0,a)` give IBP `[-2,0]` (exact) — the monotone-unary chain has no correlated
subtraction for interval arithmetic to lose. **So the genuine, tie-independent lever a semantics-
preserving ReLU rewrite has is IBP, not plain CROWN.** This reframes the resnet2b goal: not "beat plain
CROWN" (empty for reparametrizations) but **"improve IBP"** — via monotone-unary reshaping and, more
generally, linear-linear folding `B(Ax)→(BA)x` (tighter because `|BA| ≤ |B||A|` elementwise;
`plain-relu-rewrites-cant-move-crown-bound` measured ~37% IBP tightening, CROWN-neutral). Stretch
targets: IBP(rewritten) < CROWN(original) bound-gap, and IBP(rewritten) < CROWN(rewritten).

### resnet2b IBP-rewrite attempt (2026-09-04): no fold site, one neutral rewrite, IBP-vacuity hypothesis

Applying the "improve IBP" lever to stock resnet2b (`CResNet5`, in_planes=8, bn=False, dense).
Op-by-op, why no IBP-tightening `B(Ax)→(BA)x` fold site exists (every linear op is followed by a
ReLU or an Add that joins differently-rooted tensors):

| op | kind | what follows | foldable? |
|---|---|---|---|
| conv1 (stem) | linear | ReLU | no (ReLU) |
| conv1_A | linear | ReLU | no |
| conv2_A | linear | Add(+shortcut_A(s)) | no — shortcut reads s, conv2_A reads h=relu(conv1_A(s)) |
| shortcut_A | linear | Add | no — parallel, different sink timing |
| conv1_B | linear | ReLU | no |
| conv2_B | linear | Add(+z, identity) | no — z and h=relu(conv1_B(z)) differ |
| linear1 | linear | ReLU | no |
| linear2 | linear | output/spec | already folded into spec by auto_LiRPA |

ReLUs are IBP-exact (monotone), so there is also no `x−relu(x)` monotone-unary target. This reconfirms
the prior `plain-relu-rewrites-cant-move-crown-bound` finding ("resnet-v1/v2 have NO input-dependent
linear-linear fold site") from the op structure directly.

**The one exact structural rewrite resnet2b admits — block-B residual elimination — is IBP-neutral.**
`z` (block-A output) is a ReLU output ⇒ `z ≥ 0` ⇒ `z = relu(z)`, so
`conv2_B(relu(conv1_B(z))) + z  ≡  relu(wide([relu(conv1_B(z)), z]))` with `wide = [conv2_B | I]`
(identity kernel on the z half) — a plain conv-relu-conv-relu, residual removed. Built on the trained
weights as `NNs/resnet2b_ibp_vs_crown.py::Resnet2bResFree`; forward equivalence to stock is **bit-exact
(max|Δ| = 0.000e+00)**. Predicted IBP-neutral (IBP already does optimal interval addition on the Add:
width `|W2|·h_w + |I|·z_w` is unchanged) and CROWN-neutral (identity-routed channels are stably
active). Block A's shortcut is a mixed-sign CONV, not a ReLU output, so it is NOT eliminable without
adding ReLUs (strictly IBP-looser) — left as an Add.

**Bonus targets are unreachable independent of any rewrite (hypothesis to measure).** resnet2b is
standard-trained (not IBP-trained), so IBP width grows ~‖W‖₁ per layer over 7 linear layers and is
expected VACUOUS (10³–10⁵-wide) at any ε where CROWN is informative (~1–10). A constant-factor
tightening of a 10⁴-wide box is still 10⁴, so "IBP(rewritten) beats CROWN(original)" and "beats
CROWN(rewritten)" cannot hold on this net regardless of the rewrite. `NNs/resnet2b_ibp_vs_crown.py`
logs OUTPUT-INTERVAL WIDTHS (not just verified/not) under IBP/CROWN/CROWN-Opt for orig vs resfree, to
turn this expectation into a number — heavy CROWN-Opt gated behind `--full` for the compute node.
The genuine IBP-improvement demo (a rewrite that DOES tighten IBP ~37%) is the constructed
`plain_relu_more_verifiable.py`, which HAS the mixed-sign consecutive-linear fold site resnet2b lacks.

### resnet2b IBP rewrite — BUILT and MEASURED (2026-09-04): residual-fold, negative (no cancellation)

Following the "build up from small ReLU-rewrite experiments" directive, implemented the one valid ReLU
rewrite with an actual site in resnet2b and measured it end-to-end.

**Mechanism (exp 1, `NNs/ibp_residfold_mechanism.py`).** A residual block `relu(W2·relu(W1 s) + Ws s)`
rewrites exactly via `relu(u)=u+relu(-u)` to `relu(L s + W2·relu(-W1 s))` with `L = W2 W1 + Ws` — this
EXPOSES the main path's hidden linear skeleton `W2 W1` and FOLDS it with the shortcut `Ws` into one
operator `L`. Measured on a tiny gadget: when the paths cancel (`Ws=-W2W1 ⇒ L=0`) IBP output width
**halves** (21.98→10.67); with random `Ws` (`|L|` large) it is **looser** (15.4→23.8). So the rewrite
tightens IBP iff `|L|` is small — i.e. iff main and shortcut linearly cancel.

**resnet2b measurement (exp 2, `NNs/resnet2b_residual_fold.py`).** Formed `L=conv2∘conv1+shortcut` as a
single dense operator per block (basis-projection through the relu-free skeleton) and built folded
variants {foldA, foldB, foldAB}, each a valid rewrite of stock resnet2b. Diagnostic:

| block | `|conv2∘conv1 + shortcut|₁` | baseline |
|---|---|---|
| A | **10,740** | (shortcut = 1×1 conv, small) |
| B | **12,796** | `|I|₁ = 1024` |

No cancellation — `|L|` is huge. IBP output widths (mean over 4 imgs) vs orig:

| ε_pixel | orig | foldA | foldB | foldAB |
|---|---|---|---|---|
| 2/255 | 2063.8 | 2046.7 (−0.8%, noise) | 4205.9 (**+104%**) | 4197.6 (+103%) |
| 8/255 | 7646.2 | 8064.2 (+5.5%) | 12918.7 (+69%) | 13651.3 (+79%) |

**The fold does NOT improve resnet2b's IBP** (foldA within noise and sign-flipping; foldB/AB much
worse). Bit-exactness also degrades to ~6e-4 (float32 catastrophic cancellation from re-exposing the
large `A2A1`), itself a symptom of the no-cancellation structure. This upgrades the earlier *asserted*
negative to a *measured* one.

**Why it's structural, not a search miss.** resnet2b's IBP width is dominated by the MAIN FEEDFORWARD
path `|conv2|·width(relu(conv1(s)))` + head — the genuine computation, irreducible by any valid rewrite
(the only adjacent op is a ReLU, which can't be folded through). The shortcut is a minor addend
(`|As|·width(s)`, `As` a 1×1 conv), so even *perfect* shortcut cancellation would save a negligible
fraction. Both known IBP-improving families need structure resnet2b lacks: (i) linear folding needs
mixed-sign cancellation (measured absent, `|L|≈10–13k`); (ii) min/max consolidation needs a
correlated-ReLU-subtraction / linearly-dependent pair (absent — same ~0.4 proportionality floor as
snap-merge). CONCLUSION: no valid semantics-preserving rewrite meaningfully improves IBP on the *stock*
resnet2b function; an IBP win requires a net whose blocks were CONSTRUCTED/IBP-TRAINED to cancel (exp 1
shows the fold then fires) — which is a different function, not stock resnet2b.

**Airtight confirmation (16 imgs @ eps=2/255, `NNs/resnet2b_fold_confirm16.py`):** foldA tighter on **3/16** images, mean Δ **+2.59%** (worse), range −4.4%..+9.3% — the −0.8% on 4 imgs was small-sample luck. Definitive: no valid semantics-preserving rewrite improves IBP on stock resnet2b.

### resnet2b IBP: ALL FOUR rewrite doors checked and closed (2026-09-04, exhaustive)

For a valid (≤1e-4) rewrite to reduce IBP output width it must invoke one of exactly four mechanisms.
Each checked against stock resnet2b's trained weights:

| # | mechanism | requires | resnet2b status (measured) |
|---|---|---|---|
| 1 | linear fold `B(Ax)→(BA)x` | two consecutive linear layers (shared input / composed) | **no site** — every conv/linear is followed by a ReLU or a block-diagonal Add (different inputs) |
| 2 | residual fold via `relu(u)=u+relu(-u)` | main/shortcut linear **cancellation** (`|L|` small) | **absent** — `|conv2∘conv1+shortcut|₁`=10,740/12,796; foldB IBP **+104%**, foldA noise (3/16, +2.6%) |
| 3 | redundancy-collapse (merge parallel ReLU filters) | an **exactly parallel** filter/row pair | **absent** — closest pair snap-error **3.75e-2** (375× over 1e-4); all others ≥0.13 (`NNs/resnet2b_parallel_scan.py`) |
| 4 | min/max consolidation `x−relu(x)→−relu(−x)` | a correlated linear-minus-its-own-ReLU term | **no site** — resnet2b's residual is an ADD; no subtraction of a variable against its own ReLU |

Root obstruction (unifies 1,2,4): resnet2b's only shared-variable double-count is the residual, whose
two paths through the shared input are **separated by a ReLU** — nonlinear correlation, unfoldable. Door
3 fails independently (no exact redundancy; standard-trained weights, ~0.4 proportionality floor). The
IBP width is dominated by the irreducible main path `|conv2|·width(relu(conv1(s)))`.

~~**Definitive result: no valid semantics-preserving rewrite improves IBP on the stock resnet2b
function.**~~ **← OVERTURNED 2026-09-04 (see next section). The four-door table conflated two effects in
door 2 and missed a fifth door. A valid rewrite DOES improve IBP on stock resnet2b.**

### resnet2b IBP: the FIFTH door — stability-conditioned selective flip-and-fold (2026-09-04) — POSITIVE

The door-2 row above is wrong because it applied the flip `relu(u)=u+relu(−u)` to **all** coordinates and
attributed the net loss to "no residual cancellation." Split the flip by IBP neuron stability instead:

- For an **IBP-stable-active** coord `i` of `conv1_A(s)` (`l_i^IBP > 0`): `relu(−conv1(s))_i` has IBP width
  **exactly 0** — the flip is **free** — and coord `i`'s linear contribution folds with the shortcut into
  one op, `|L_S| ≤ |B_S||A_S|+|short|`, **strict generically, NO weight cancellation needed**.
- For an **unstable** coord the flip *adds* `|conv2[:,i]|·|l_i|` width (this is the door-2 cost). The full
  fold flipped these too, drowning the gain — hence the earlier +2.6%/+104%.

So the lever is **neuron stability, not residual weight cancellation.** At eps=2/255, `conv1_A` is
~43% stable-active / ~39% stable-inactive / ~18% unstable — a large live set. Selecting `S` = the
stable-active set and folding only those coordinates (`NNs/resnet2b_stability_fold.py`, `L_S` built in
float64, `relu(u)=u+relu(−u)` a **global** identity for any fixed `S` so every folded net is globally
equivalent to stock resnet2b to ~2e-6). **All numbers below are on REAL CIFAR-10 test images** (an earlier
draft used uniform-noise inputs — noise gives a rosier −53% / 37% majority and is NOT the honest figure):

| variant | S | IBP output width vs orig | tighter | note |
|---|---|---|---|---|
| **per-box S** (each net globally = resnet2b, ~2e-6) | ~43% per box | **−41.2%** (1691.5 → 994.2) | 16/16 | **SOUND**: MC in-box violation of fold bounds = −213 (≤0) |
| fixed majority S (one net, calib active ≥9/16) | 14.7% | **+8.4%** (worse) | 1/16 | measured on HELD-OUT — does NOT generalize |
| fixed conservative S (active on ALL calib) | 0.3–0.6% | −0.01% (≈0) | 15/16 | input-independent set is tiny |

**Headline: the fifth door is a PER-BOX exact rewrite.** For each verification box we emit a network that
is *globally* functionally identical to stock resnet2b (~2e-6 on random inputs) but on which IBP is **41%
tighter, soundly** (Monte-Carlo confirms all in-box outputs lie inside the folded IBP box). This meets the
goal ("a rewrite of resnet2b that causes IBP to improve, max error ≤1e-4") per instance — the same kind of
box-informed exact-rewrite selection every verifier already does internally. A *single input-independent*
fixed rewrite does **not** meaningfully help on real images: coords stable-active across all images at
2/255 are only 0.3–0.6% (box-stability is genuinely input-dependent), and the naive majority mask even
loses out-of-sample. So the correction to the "all four doors closed" claim is real but scoped: **a valid
rewrite tightens IBP on stock resnet2b, per box, not as one fixed net.**

Bonus **MEASURED UNREACHED**: plain CROWN(orig) mean width ≈ 3.4 ≪ IBP(fold) ≈ 994, so IBP stays vacuous
relative to CROWN on this standard-trained net — no rewrite makes IBP beat CROWN (CROWN linearizes stable
neurons *and* relaxes unstable ones; the fold only recovers the stable-neuron part). Door 1 was checked
only at the *graph* level ("no consecutive linears"); stable ReLUs create *box-conditional* consecutive
linears at every ReLU layer — that is the fifth door.

### Does the same fold improve VANILLA CROWN (no α-opt)? NO — measured neutral, structural (2026-09-04)

Ran the identical per-box stability fold under the CROWN family (`resnet2b_stability_fold.py` section D,
`conv_mode=matrix` — the constant-mask `Mul` breaks auto_LiRPA's default Patches mode, same class as the
`convfused-verified-neutral` Split note):

| method | intermediate bounds | orig → fold width | result |
|---|---|---|---|
| **vanilla CROWN** | backward-CROWN | 3.4156 → 3.4156 | **NEUTRAL, 0/16** (signed Δ ∈ [−9.5e-7, +4.3e-4] = float32 `L_S`-reconstruction, not a bound change) |
| CROWN-IBP | IBP | 1025.4 → 598.3 | **−41.7%, 16/16** |
| IBP | IBP | 1691.5 → 994.2 | −41.2%, 16/16 |

**The fold helps exactly the methods whose *intermediate* bounds come from IBP; vanilla CROWN is not one.**
It back-substitutes CROWN intermediates *exactly* through the stable (linear) ReLUs, so removing their box
slack is invisible to it (the 0/16 is the theorem, not a near-miss). CROWN-IBP takes IBP intermediates, so
the fold's 42%-tighter block-A box propagates to tighter block-B hulls → −41.7%. So the fifth door is
**IBP-specific: IBP and plain CROWN lose tightness in different places, and the stable-neuron place is
already exact for CROWN.**

Why no CROWN analog exists (induction — closes exact rewrites for plain CROWN on stock resnet2b): plain
CROWN's bound = exact linear back-substitution + Σ over *unstable* `i` of `|A_i|·gap(l_i,u_i,α_i)`, with the
area heuristic `α_i = 1[u_i > |l_i|]`. Layer-1 hulls are exact (linear in the input box) ⇒ rewrite-
invariant; layer-k hulls are CROWN bounds through layers <k ⇒ invariant by induction ⇒ `α_k`, `gap_k`
invariant; the `A_i` are the linear skeleton ⇒ invariant. The **only** escape is changing the neuron *set*
— merging proportional pre-activations (redundancy-collapse) or restructuring a min/max tree — both
**hull-CHANGING**. (The "create a complementary pair via the flip" idea also closes: un-sharing a ReLU into
`relu(q)` + `q+relu(−q)` is redundancy-collapse *in reverse* — always ≥ original, equality iff same-sign
downstream coeffs. Every "introduce a second ReLU" construction reduces to this; nested/clipped re-
expressions like `relu(relu(z))`, `min(u,relu(z))` add only *stable* ReLUs. So the neuron set is the only
lever.)

**Authored the CROWN-improving rewrite and measured it — then scanned stock resnet2b for its fire site
(2026-09-04).** (1) The rewrite that DOES move vanilla CROWN (`NNs/reassoc_results/crown_redundancy_collapse.py`,
`method="CROWN"` not just α-opt): proportional-merge opposite-sign **−59.4%** (132.75→53.94), complementary-
collapse opposite-sign **−33.4%** (193.42→128.81); same-sign controls exactly neutral (0.0%) — proving the
mechanism is downstream-coefficient sign cancellation. (2) Proper fire-condition scan on stock resnet2b
(`NNs/resnet2b_parallel_scan.py`-successor, AUGMENTED `[weight|bias]` pre-activation vectors, sign-separated):
NO proportional site (best cos +0.59); the closest pair anywhere is the stem's near-**complementary** pair
(cos −0.9932, β −0.948) — but exact-merge snap-error **4.25e-2** (14% rel), 425× over 1e-4. **Best exact
site tolerance anywhere = 4.25e-2.** So: **no EXACT rewrite improves vanilla CROWN at snap tolerance
≥ 4.25e-2 on these trained weights.** (3) The snap-merge SURROGATE (snap that pair exact, collapse, subtract
a composed certificate δ) LOSES under the composed-CROWN δ — but "vacuous" is not fully proven, and the δ
numbers are a good bracketing lesson. δ = sup_box|orig−snap| bracketed: the *sampled* change (2000 pts/box)
= **0.09** (a data-manifold LOWER bound; looked like a surprise win); the composed-CROWN bound of `orig−snap`
over the eps-box = **7.18 per logit** (an UPPER bound, and a LOOSE one — CROWN back-substitutes both branches'
ReLU relaxations independently, blind to their near-identity, so it over-charges). The true sup is somewhere
in **[0.09, 7.18]**. With δ=7.18 the certificate `margin ≥ margin_collapsed − δ` is destroyed (7.18 > the
3.42 width); with a tighter δ estimator (e.g. the single-neuron Lipschitz `|W|`-propagation bound used in
[[snap-merge-certified-surrogate]]) it is UNKNOWN — and I did not measure the actual collapse tightening
either (closed the case on δ alone). Snap alone makes CROWN slightly worse (+0.025). LESSON: **sampled δ is a
lower bound, composed-CROWN δ a loose upper bound; a surrogate's viability depends where the true sup falls —
measure both, conclude from neither alone.** So: **no EXACT rewrite improves vanilla CROWN on stock resnet2b
at snap tolerance ≥ 4.25e-2; the surrogate loses under composed-CROWN δ but tighter-δ + collapse-tightening
viability is OPEN.** (My earlier "huge Lipschitz" assertion is neither confirmed nor refuted — sample says
small, CROWN bound says large, true value unmeasured.) A resnet2b-*architecture* net with planted redundancy
or a redundancy-regularized retrain would show the collapse improving its vanilla CROWN directly.

## 2026-09-04 (cont. 2) — REAL modern model: exact attention-GAUGE rewrites improve CROWN on the VNN-COMP'23 ViT

**Goal reframed (user `/goal`): find REAL trained models (downloadable; NOT hand-made/modified) that a rewrite
makes more verifiable — best = modern architecture + full CROWN, then vanilla CROWN, then IBP. Planted/CReLU
results are disqualified as results.** Target chosen: the **VNN-COMP 2023 `vit` benchmark**
(`vnncomp2023_benchmarks/benchmarks/vit`, sparse-cloned): two real, competition-standard vision transformers,
`pgd_2_3_16` (PGD-trained; 2 layers, 3 heads×16, d=48, 5 tokens, BatchNorm pre-norm, ReLU MLP 48→96→48, softmax
attention) and `ibp_3_3_8` (IBP-trained, 3 layers, 17 tokens), 100 instances each, ε = 1/255 (normalized
0.0197), 100 s timeout, 9 margin specs `Y_label − Y_i`. Compute: user's 12 h L40S allocation (g3120) via
`srun --jobid --overlap`; nothing heavy on the login node.

**Harness.** `NNs/vit_rewrite/vit_model.py` = faithful PyTorch reimplementation loading the stock ONNX weights
(faithful gate: fp32 vs onnxruntime **2.4e-6**), with switchable EXACT rewrite variants (fp64 gate vs stock
≤ **2.7e-15** for R1/R3, ~2e-8 for the re-factored weights = fp32 storage of the fused matrices, same precision
class as the stock fp32 weights). `vit_bounds.py` = per-instance IBP / vanilla CROWN / alpha-CROWN lower
bounds on the exact vnnlib boxes, Monte-Carlo soundness check, `--diag` slack attribution, `--width`.
`vit_export.py` = ONNX export (strip BN `training_mode` for onnx2pytorch) + benchmark dir for the UNMODIFIED
official `abcrown.py` pipeline. `vit_gauge_opt.py` = gradient-learned gauge (below).

**Baseline landscape (instance 4021 first, then 100).** (a) auto_LiRPA softmax mode matters enormously and
*differently per tier*: vanilla CROWN **lse −0.70 ≫ complex −2.28** (official config uses complex); alpha-CROWN
**complex −0.174 (7/9 specs) ≫ lse −0.70** — in lse mode alpha-CROWN ≡ vanilla CROWN to 6 digits (the lse softmax
relaxation has no alphas; only the MLP ReLUs do, and they don't move it). So the BEST stock baseline is
mode-dependent: lse for vanilla CROWN, complex+alpha for full CROWN. (b) IBP is 5 orders of magnitude vacuous
(min lb −695,109) and even crashes in complex mode (exp underflow → `BoundReciprocal` positivity assert): the
IBP tier is dead on this ViT. (c) Standalone vanilla CROWN in complex mode raises `BoundReduceMax perturbed
indexes NotImplementedError` (the official pipeline avoids it via its intermediate-bound path). (d) Official
pipeline (complex+alpha+BaB, 100 s), pgd model, running: at 35/100 → 6 safe, 11 safe-incomplete, **18 unknown**
— plenty to flip. (e) Full 100, vanilla CROWN lse, stock: **24/100 verified by the incomplete bound alone**,
mean min-lb −0.614, mean width 4.264.

**Negative rewrites (all exact, all measured, all WORSE than the best baseline).** R1 softmax shift-invariance
(`softmax(s)=exp(s−c)/Σ`, c fixed, deletes the perturbed ReduceMax/Sub of the `complex` decomposition):
vanilla −10.65 (c=0) vs lse −0.70; alpha −1.59 (c=0), **−302 with c=10** (float conditioning), per-row
fixed-mean shift −1.46 @ 85 s — the numerical-stability max is ALSO the relaxation-friendly form, and any
primitive decomposition loses to the joint `lse` relaxation. R2 QKᵀ reassociation onto X (`(XM)Xᵀ`, `X(MXᵀ)`,
M_h=W_qW_kᵀ + exact bias cross-terms): −1.72 vs −0.70. R3 `(A·X)W_v` : −2.40. Lesson: the trained projections
COMPRESS the bilinear operands (16 tight symbolic dims vs 48 raw); pushing the product onto X adds terms and
loses cancellation.

**Slack attribution (lse, vanilla CROWN, 3 instances, linearize ONE nonlinearity at the box center; inexact,
diagnostic only).** mean width 3.949 full → 2.153 without QK-bilinear slack (−45%), 2.434 without softmax
(−38%), 1.956 without AV-bilinear slack (−50%), 0.904 without all three (77% of the width is the attention
nonlinearities; remainder = MLP ReLUs + interactions). The two bilinear products are the biggest levers.

**★ POSITIVE: the attention GAUGE rewrite.** `(X W_q)(X W_k)ᵀ = (X W_q G)(X W_k G^{-T})ᵀ` and
`A(X W_v)W_o = A(X W_v G)(G^{-1}W_o)` for ANY invertible G ∈ GL(16) per head/layer (biases transform with G) —
an exact rewrite FAMILY (matmul associativity + inserting GG⁻¹) that keeps 16 bilinear products but changes the
per-coordinate operand widths CROWN's McCormick relaxation sees (diagonal G is provably neutral, only mixing
matters). Closed-form choice R4/R5 = SVD-balanced factorization (W_q' = U√Σ, W_k' = V√Σ of M_h; same for
(W_v,W_o)). **Full 100 instances, vanilla CROWN lse: 24 → 26 verified (+2 flips); mean min-lb −0.614 → −0.525;
mean width 4.264 → 4.089 (−4.1%); every instance and method improved; MC-sound; random orthogonal gauges are
slightly WORSE than stock (3.97 vs 3.95 on 3 inst.), so SVD is a real basis, not luck.** R4 alone 26 (4.146),
R5 alone 26 (4.205), both 26 (4.089). Exported to ONNX (`vit_R45_both_svd/`, 2.1e-6 vs stock on all centers).
**Learned gauge** (`vit_gauge_opt.py`: gradient ascent on the CROWN lower bound, chain-ruled through the exact
gauge algebra, tuning boxes = ε-boxes around CIFAR-10 TRAIN images (disjoint from the test-set benchmark
instances), init SVD, tiny cond penalty): smoke test climbs monotonically 1.92→2.09 mean lb / −0.335→−0.178
mean min-lb in 8 steps, cond(G) ≤ 2.8, gate 8e-8. Required a gradient fix in the fork's
`auto_LiRPA/operators/softmax.py` (`_softmax_lse_lower/upper`: `torch.where` 0/0 → NaN grads; gradient-safe
denominators, forward values unchanged).

**Paired per-instance statistics (`vit_compare.py`), pgd_2_3_16, 100 benchmark instances, vanilla CROWN lse.**
SVD gauge R45: min-spec lb tighter on **100/100** instances (Δ mean +0.088, worst +0.021), 893/900 specs tighter,
width narrower on 100/100. R4 alone: 100/100 (Δ +0.058), R5 alone: 93/100 (Δ +0.032, 7 looser). So the
closed-form gauge is a monotone improvement on this model, not 2 lucky flips.

**★★ LEARNED gauge, OUT-OF-SAMPLE (400 Adam steps on 512 CIFAR-TRAIN ε-boxes, ~5 min on the L40S; held-in eval
climbed 1.92→2.32 mean lb, −0.335→−0.044 mean min-lb, frac_ver 0.36→0.43; max cond(G)=2.8; fp64 gate vs stock
4.9e-8; `gauges/pgd_mix_svdinit.pt`). Evaluated on the 100 TEST-set benchmark instances (never seen):
vanilla CROWN lse **24 → 36 verified (+12 flips, 0 reverse)**; mean min-lb **−0.614 → −0.154**; mean width
**4.264 → 3.282 (−23.0%)**; tighter on **100/100 instances and 900/900 specs** (Δ mean +0.459, worst +0.213);
MC-sound (max violation −0.75). vs the SVD gauge: 26 → 36, 100/100, 900/900.** A single fixed, input-independent
exact rewrite of the stock weights. Exported to ONNX (`vit_learnedG_pgd/`, 3.0e-6 vs stock on all 100 centers).

**Second real model, ibp_3_3_8 (3 layers, 17 tokens), 100 instances, vanilla CROWN lse.** Stock: 15/100,
mean min-lb −0.0296, width 1.156. **SVD gauge R45 is WORSE: 12/100, looser on 100/100 instances / 900/900
specs (Δ −0.002, width +0.4%).** The closed-form balancing is NOT universally good — it happened to align
with CROWN's slack on the pgd model; the learned gauge (which optimizes the actual bound) is the principled
version. Learned gauge for ibp_3_3_8: running (batch 2; batch 8/32 OOM the 44 GB GPU — the autograd graph of
CROWN through 3 layers × 17 tokens is large).

**Official-pipeline ("full CROWN") tier.** The first official baseline run was contaminated (GPU shared with
probes; `auto_enlarge_batch_size` sizes BaB batches from free memory; finally killed at 47/100 by my ibp gauge
learner OOM-ing the card) — treated as a pilot only: 47 done → 10 safe / 12 safe-incomplete / 25 unknown;
initial complex-mode vanilla CROWN verifies 0/47 (mean min-lb −2.67 — far looser than lse's −0.61), alpha-CROWN
verifies 4.83/9 specs on average. Clean comparison = `run_chain.sh`: learned-G, stock (pgd-only instances.csv),
R45, each ALONE on the GPU with the untouched vit.yaml settings; parsed by `vit_official_parse.py` (per-instance
initial CROWN = deterministic complex-mode vanilla CROWN, alpha-CROWN #specs verified, final verdict with the
100 s BaB caveat). IN PROGRESS.

**safenlp (VNN-COMP'24) checked as the "more modern NLP transformer" fallback: it is NOT a transformer** —
both `perturbations_0.onnx` are 30→128→2 ReLU MLPs (4,226 params) on precomputed sentence embeddings. No
Attention/Softmax/MatMul-bilinear ops; nothing for the gauge rewrite to act on and no more modern than the ViT.

## 2026-09-05 — ViT gauge rewrite: full-CROWN (official pipeline) tier, robustness checks, ibp hard-box re-learn

The 12 h interactive allocation from 2026-09-04 ended after 2 h 40 (shell exit, not a crash) and killed the clean
official chain before its first Result. Resumed 06:48 on a fresh 12 h L40S (g3114) via `run_chain2.sh`: the three
official abcrown runs (untouched vit.yaml settings) sequential and ALONE on the GPU, CPU-only side jobs alongside.

**ibp_3_3_8 learned gauge (last night's, tuned on 128 easy train boxes) is out-of-sample NEUTRAL:** 15 → 15
verified, tighter on 75/100 instances / 549/900 specs, Δ mean +0.0006 (width −0.0%). Cause: its tuning boxes were
82% verified at init (held-in mean min-lb +1.06 vs the benchmark's −0.03), so the objective had no signal. Fix:
`vit_gauge_opt.py --hard 1 --pool N` scores a pool of train boxes with stock CROWN and keeps the n_train with the
smallest |min-lb|. On ibp_3_3_8 the pool of 600 is 80% verified (mean +1.01); the hard 192 have mean +0.007,
range [−0.36, +0.35], 49.5% verified — benchmark-like. Re-learn running on CPU (`gauges/ibp_mix_hard.pt`).

**Init is not load-bearing (pgd_2_3_16).** `--init id --seed 1`, 200 steps on 512 train boxes (CPU): held-in
mean min-lb −0.558 → −0.110, frac_ver 0.289 → 0.398 (SVD init at 200 steps: −0.060 / 0.422). Out-of-sample on
the 100 test instances, vanilla CROWN lse: **24 → 37 verified (13 flips, 0 reverse), tighter on 100/100 instances
and 900/900 specs (Δ mean +0.442, worst +0.200), width −22.2%, MC-sound (−0.77)** — essentially the SVD-init
gauge's result (36; the two learned gauges differ by Δ −0.018 mean, mixed sign). Gate 3.5e-8. Two independent
runs (different init, seed, step count) land on the same improvement: it is the family + objective, not a lucky G.

**Official pipeline (full CROWN: complex softmax + alpha-CROWN 50 it + beta-CROWN BaB, 100 s), learned-G run
(`vit_learnedG_pgd/`), 100 instances, GPU exclusive:** 13 safe + 52 safe-incomplete = **65/100 verified**,
35 unknown; initial complex-mode vanilla CROWN verifies 13/100 (mean min-lb −1.06); alpha-CROWN verifies
7.93/9 specs on average. Interim pairing against the contaminated pilot stock run (initial CROWN is deterministic
and contention-free, so this part is valid; 20 common instances): initial CROWN tighter on 20/20 (Δ mean +1.19,
worst +0.36); alpha-CROWN more specs on 6, fewer on 0. Clean stock run (pgd-only instances.csv) IN PROGRESS, then
R45; final paired numbers below when done.

**★★★ FULL-CROWN TIER RESULT (official alpha-beta-CROWN pipeline, unmodified vit.yaml settings, each run ALONE on
the L40S, 100 s/instance, pgd_2_3_16, the 100 benchmark test instances; `vit_official_parse.py`
`_scratch/official_stock_pgd.log` vs `_scratch/official_learnedG_pgd.log`):**

| | stock ONNX | learned-G rewrite (exact) |
|---|---|---|
| initial complex-mode vanilla CROWN: all-9 verified | 0/100 (mean min-lb −2.52) | **13/100** (mean min-lb −1.06) |
| paired initial CROWN min-lb | | **tighter on 100/100**, Δ mean +1.46, median +1.30, worst +0.36 |
| alpha-CROWN (50 it): all-9 verified (safe-incomplete) | 41/100 | **52/100** |
| paired alpha-CROWN #specs verified | | more on **27**, fewer on **0**, same 73 (net +46 of 900) |
| final verdict (alpha + beta-CROWN BaB, 100 s) | 58/100 (17 safe + 41 safe-inc.) | **65/100** (13 safe + 52 safe-inc.) |
| verdict flips | | unknown→verified **7** [60, 388, 4671, 5927, 7064, 9106, 9145], verified→unknown **0** |
| mean time / instance | 49.5 s | 42.0 s |

The gauge was learned against vanilla lse-CROWN on TRAIN boxes, so this is a transfer result: it carries over to
the complex softmax relaxation, to alpha-optimized bounds, and to BaB. The two contention-free levels (initial
CROWN, alpha-CROWN) are monotone improvements (0 instances worse at either level); the BaB verdict is time-capped
but each run had the GPU to itself with identical settings. Same fp32-storage caveat as before (3.0e-6 vs stock).
SVD-gauge (R45) official run IN PROGRESS; ibp_3_3_8 hard-box learner IN PROGRESS.

**Controls added after advisor review (2026-09-05 09:30).** (i) Export-path confound: the learned-G run above used a
re-exported ONNX (opset 14, 210 nodes) while stock is the competition file (opset 9, 133 nodes). New
`vit_patch_onnx.py` writes the gauge-transformed weights INTO the stock graph (structure byte-identical, 14/16
attention initializers change, identity control changes 0/16 and is bit-identical) → `vit_learnedG_patched/`,
`vit_idinitG_patched/`; `vit_export.py --variant base` → `vit_base_export/` (identity weights through the export
path, 0.0 vs stock at centers). Early CPU result (7 instances × 9 specs): base_export initial CROWN vs stock
max|Δ| = 8.8e-6 — the export path does NOT move initial CROWN; the rewrite's Δ is 5 orders larger. GPU official
runs of learnedG_patched / base_export / idinitG_patched queued after R45 (`run_chain3.sh`). (ii) The initial
alpha-CROWN pass never hit its 30 s cap (max 21.6 s stock, 25.4 s learned-G) → initial CROWN and alpha-CROWN levels
are both deterministic; only BaB verdicts are time-capped. (iii) Margins of the newly verified instances vs the
3e-6 fp32-storage discrepancy: 7 BaB flips min 6.9e-4 (last-batch proxy), 11 new alpha-only verifications min
2.1e-4 — ≥70× the discrepancy everywhere.

**SVD (closed-form) gauge R45 at the full tier (official pipeline, alone on the GPU, 100 instances):** initial
complex-mode CROWN tighter on 99/100 (Δ mean +0.32, worst −0.07), all-9 verified 0 → 3; alpha-CROWN #specs more
on 5 / fewer on 1 (net +4); final verdicts identical (17 safe + 41 safe-incomplete both; 0 flips either way);
alpha pass never time-capped (max 25.3 s). So the closed form moves the deterministic levels modestly and the
verdicts not at all — the LEARNED gauge is what turns the tightening into verified instances (58 → 65).

**Export-path control complete (initial CROWN, 100 instances × 9 specs, base_export CPU run vs stock GPU run):**
max|Δ| = 6.3e-5, mean|Δ| = 3.9e-6, 0/100 verified either way → the re-export path is bound-neutral; the rewrite's
Δ (+1.46 mean, +0.36 worst) is 4–5 orders of magnitude larger. Independent replication at this level: the id-init
gauge patched into the stock graph (CPU run) is tighter on 100/100 vs stock, Δ mean +1.42, worst +0.34, 12/100
verified outright (SVD-init gauge: 13/100). CPU alpha passes are time-capped (59/88) → alpha/BaB levels for these
two models come from the GPU runs (queued).

**ibp_3_3_8 hard-box learned gauge — out-of-sample NEUTRAL too (measured negative):** 300 steps on the 192
benchmark-like train boxes (held-in min-lb −0.0095 → −0.0073, frac_ver 0.453 → 0.461, cond 41 → 1.7, gate 5.1e-8);
on the 100 test instances: 15 → 15 verified, Δ mean +0.0006 (worst −0.0007, best +0.0046), tighter on 75/100
instances / 505/900 specs, width −0.0%, MC-sound (−0.236). Same as the easy-box gauge → on this model the gauge
family has ~no leverage: its vanilla-CROWN slack is not in the per-head bilinear operand basis (3 layers × 17
tokens; the pgd model's 77 %-attention width decomposition does not carry over). The gauge is a real but
model-dependent lever: large on pgd_2_3_16, nil on ibp_3_3_8.

**★★★ CLEAN HEADLINE (learned gauge written INTO the stock ONNX graph, `vit_learnedG_patched/`; official pipeline,
alone on the GPU; paired vs the stock run, 100 instances):**

| level | stock | learned-G (stock graph, 14 initializer values changed) |
|---|---|---|
| initial complex-mode vanilla CROWN, all-9 verified | 0/100 | **13/100**; tighter on **100/100**, Δ mean +1.46, median +1.30, worst +0.36 |
| alpha-CROWN (50 it, never time-capped), all-9 verified | 41/100 | **52/100**; #specs more on **27**, fewer on **0** (net +46/900) |
| final verdict (alpha + BaB, 100 s) | 58/100 | **64/100**; unknown→verified **6** [60, 388, 4671, 5927, 9106, 9145], verified→unknown **0** |
| margins of newly verified (vs 3e-6 fp32 storage) | | BaB flips min 6.9e-4; alpha-only min 2.1e-4 |
| mean time / instance | 49.5 s | 43.1 s |

The exported-graph run and the stock-graph run of the SAME learned weights agree exactly at both deterministic
levels (initial CROWN Δ = 0 on 100/100; alpha-CROWN #specs identical on 100/100); they differ on one BaB verdict
(7064: safe in the exported run, unknown here — time-capped BaB noise, so the verdict gain is 6–7 depending on run).
Together with base_export ≈ stock (max|Δ| 6.3e-5), the export path is fully ruled out as the source of the gain.
GPU official runs of base_export and idinitG_patched (alpha/BaB-level replication) IN PROGRESS.

**Final checks (12:05).** (a) fp32-storage discrepancy over the BOX, not just the center (`vit_box_discrepancy.py`,
onnxruntime, 1000 uniform points + corners/center per box): learned-G patched vs stock sup = 2.98e-6 on the 18 newly
verified boxes, 3.34e-6 over all 100 boxes (200 pts each); id-init gauge 3.34e-6. Smallest newly verified margin
2.1e-4 ≥ 70× the box discrepancy → every new certificate transfers to the stock model. (b) Ceiling check (no
downloads; `git ls-tree` on the sparse clones, GitHub API for 2025): VNN-COMP 2023 benchmarks = acasxu cctsdb_yolo
cgan collins_rul_cnn collins_yolo_robustness dist_shift metaroom ml4acopf nn4sys tllverifybench
traffic_signs_recognition vggnet16 **vit** yolo; 2024 = the 2023 set re-listed + cifar100 tinyimagenet cora
linearizenn lsnc safenlp ml4acopf_2024 (**vit_2023** is the only transformer; safenlp is a 30→128→2 MLP, checked at
op level; the others are CNN/ResNet/MLP/ODE benchmarks by name). No vnncomp2025_benchmarks repo exists (HTTP 404).
So the two `vit` models are the only trained transformers in the competition suite; both were run (pgd_2_3_16:
large gain; ibp_3_3_8: neutral). Larger trained transformers are outside CROWN's reach at ε = 1/255 anyway —
ibp_3_3_8 (3 layers, 17 tokens) is already at 15 % stock-verified with mean min-lb −0.03. (c) The GPU base_export
official run was killed at 7/100 (PID 2773577) as redundant: exported-vs-patched learned-G had already shown the
export path is neutral at the alpha level; the chain moved to idinitG_patched (GPU, alpha/BaB replication).

**Replication at the full tier (id-init gauge, seed 1, patched into the stock graph; official GPU run, alone; DONE_ALL
13:14):** 15 safe + 50 safe-incomplete = **65/100** (stock 58); unknown→verified **7** — the SAME seven instances
[60, 388, 4671, 5927, 7064, 9106, 9145] as the SVD-init exported run — verified→unknown **0**; initial CROWN tighter
on 100/100 (Δ +1.42); alpha-CROWN more specs on 26, fewer on 0 (net +42); alpha pass never time-capped (max 21 s);
mean time 49.5 → 41.2 s. Smallest newly verified margin 9.0e-5 (instance 1546, alpha-only) vs box discrepancy
3.3e-6 → 27×; all others ≥ 4.3e-4. Two independently learned gauges (different init, seed, step count) reproduce
the same top-tier gain on the same instances: the result is the rewrite family + objective, not a particular G.

**Tier achieved: FULL CROWN on a modern (transformer) trained competition model; goal closed 2026-09-05 13:15.**
Summary table (pgd_2_3_16, 100 instances, official pipeline): stock 58 → learned-G 64/65/65 (three runs: patched
svd-init, exported svd-init, patched id-init); 6–7 unknown→verified, 0 reverse; alpha-CROWN 41 → 52/52/50;
initial CROWN 0 → 13/13/12. Vanilla CROWN lse: 24 → 36/37. Controls: export path neutral (6.3e-5), alpha never
time-capped, margins ≫ fp32 box discrepancy. Negatives: R1–R3 exact rewrites worse; SVD gauge verdict-neutral
(pgd) / worse (ibp); learned gauge neutral on ibp_3_3_8 (easy and hard boxes); IBP vacuous; safenlp is an MLP.

## 2026-09-05 (cont.) — Why the gauge result does NOT replicate on ibp_3_3_8 (mechanistic diagnosis)

Goal (user, 13:30): replicate the pgd_2_3_16 full-CROWN gain by training gauges for the ibp_3_3_8 model. Two
learned gauges already existed for it (easy train boxes → 15→15; hard/benchmark-like boxes → 15→15, Δ +0.0006,
see above), so the question became WHY, before spending GPU time on a third. New 24 h allocation: 2 × L40S
(job 39619518, g3120 + g3124).

**Training objective used for ALL gauges (pgd and ibp), for the record:** the vanilla CROWN lower bound itself
(auto_LiRPA method `CROWN`, softmax mode `lse`, no alpha optimization), objective `mix` = 0.5·mean over the 9
margin specs + 0.5·mean over boxes of the worst spec, + cond penalty 1e-4·(‖G‖²+‖G⁻¹‖²), grad-norm clip 1.0,
Adam lr 0.01, batch 32 (pgd) / 2 (ibp, GPU memory), boxes = ε-boxes around correctly classified CIFAR-10 TRAIN
images (disjoint from the test-set benchmark instances). NOT the IBP method (vacuous on these ViTs, −7e5, no
signal) and NOT bound width. Gradients flow CROWN → gauged weights → G (exact gauge algebra, differentiable;
needed the gradient-safe softmax.py denominators). Evaluation is always on the 100 benchmark test instances.

**The two models** (same block design: BN pre-norm, 3 heads × 16, d=48, ReLU MLP 48→96→48, softmax attention;
same ε=1/255, 9 specs): pgd_2_3_16 = PGD-adversarially trained, 2 layers, patch 16 → 5 tokens, stock vanilla
CROWN 24/100 (mean min-lb −0.61); ibp_3_3_8 = IBP-certified-trained, 3 layers, patch 8 → 17 tokens, stock 15/100
(mean min-lb −0.03, i.e. benchmark instances sit right at the boundary).

**Slack attribution on ibp_3_3_8 (`run_ibp_attrib.sh`; `--diag` linearizes ONE attention nonlinearity at the
box center — inexact, diagnostic only; lse vanilla CROWN, 8 instances, mean width):** full 1.186 → linQK 1.185
(−0.1%) → linSM 1.171 (−1.3%) → linAV 1.167 (−1.6%) → all three 1.152 (**−3%**). On pgd_2_3_16 the same three
were **77%** of the width (3.949 → 0.904). The gauge family can only change how CROWN relaxes QKᵀ, softmax and
A·V (it is provably neutral for linear ops and cannot touch ReLUs), so on ibp_3_3_8 it has at most ~3% of the
width to work with — consistent with the measured Δ ≈ +0.0006. IBP training makes the attention nearly
interval-friendly/linear; the remaining slack is the 3 × 17 × 96 = 4896 MLP ReLUs and their interaction across
layers. This is the mechanism behind "gauge = model-dependent lever".

**Sampling diagnostic (`vit_sample_diag.py`, 20 boxes × 256 uniform samples; not a bound):** both models look
alike at the sample level — attention probabilities move by ≤0.004 across a box, normalized attention entropy
0.63 (pgd) / 0.62 (ibp), and only 0.6% (pgd) / 0.5% (ibp) of MLP ReLUs change sign inside a box. So the
difference is in how loose CROWN's *relaxations* are, not in the functions' input sensitivity: on pgd the
bilinear/softmax relaxations are loose relative to the true ranges (hence the gauge lever); on ibp they are
already tight.

**ReLU-side exact rewrites checked:** the one exact ReLU rewrite known to tighten CROWN (redundancy collapse —
merge duplicated or complementary hidden units) does not fire: ibp_3_3_8 MLPs have 0 dead units and 0 pairs
with |cos| > 0.99 (max cos 0.949 / 0.922 / 0.850 per layer; pgd: 0.728 / 0.507). Permutation / positive
scaling of hidden units and any change of basis of the residual stream (BN → affine fold) are exact but
CROWN-neutral (linear-op rewrites compose exactly). No exact rewrite family with leverage on this model is
known; see the plain-ReLU neutrality results earlier in this file.

**Standalone complex-mode CROWN fix (for future complex-objective training):** the NotImplementedError
(`BoundReduceMax` with perturbed max indices) is avoided in the official pipeline by
`bound_opts['fixed_reducemax_index'] = True` (set in `beta_CROWN_solver.py`); the harness can pass the same
option. Not needed for ibp_3_3_8 given the 3% ceiling.

**Full-tier baseline for stock ibp_3_3_8 (official pipeline, alone on g3120): RUNNING** — needed to quantify
the headroom that any ibp_3_3_8 rewrite would have at the tier that matters.

## 2026-09-05 (cont. 3) — Goal: replicate the gauge result on more downloaded transformers

**Candidate search (transformers that come with a verification spec and slot into alpha-beta-CROWN):**
VNN-COMP 2023/2024 have only the two `vit` models already covered (2024 `safenlp` is an MLP; no 2025 repo yet).
The GenBaB benchmark suite (Shi et al., TACAS 2025; HF dataset `zhouxingshi/GenBaB`, `cifar/vit_{1_3,1_6,2_3,2_6}`)
ships four PGD-trained CIFAR-10 ViTs with vnnlib specs (ε=1/255, instances pre-filtered to those vanilla CROWN
does not verify and PGD does not falsify) and abcrown configs (`Customized("../models/vit.py", ...)`) — the best
possible fit for the pipeline. Downloaded with `NNs/vit_rewrite/genbab_download.py` (105 MB, git-ignored).
Normalization/disjointness check: spec bounds reproduce `(clip(test[id]±1/255)−mean)/std` to 2e-7, spec id =
CIFAR test index → our CIFAR-train tuning boxes are disjoint.

**Result: the gauge has NO leverage on any GenBaB ViT — their attention is constant over the ε-boxes.**
`NNs/vit_rewrite/genbab_gauge.py` (exact per-head gauge on the nn.Linear q/k/v/out weights, fp64 gate, learner,
export). Probe (vanilla CROWN lse, 8 instances each):

| model | softmax interval width over the box (per layer) | attention at box centre | random gauge (scale 0.3) max Δ min-lb |
|---|---|---|---|
| vit_1_3 | L0: 0.00000 (probs exactly 0.200 = 1/5) | scores std 0.0000 → exactly uniform | 2.9e-6 |
| vit_1_6 | L0: 0.00000 (lower 0.000, upper 1.000) | scores std 9.76 → saturated one-hot | 2.7e-6 |
| vit_2_3 | L0: 0.00000 (uniform); L1: 0.0043 | L0 uniform; L1 mean |p−1/T| 0.06 | 4.9e-5 |
| vit_2_6 | L0: 0.00000; L1: 0.00000 (both uniform) | both exactly uniform | 1.1e-6 |

Mechanism: with 5 tokens and ε=1/255 the PGD-trained ViTs collapsed their attention to a token mean (exactly uniform
softmax; the query/key projections are ~constant across tokens) or to hard one-hot attention that the ε-box cannot
move — so QKᵀ, softmax and AV contribute no CROWN slack and the whole width (0.5–0.6) sits in the ReLU MLPs and
LayerNorms. This is the opposite of the VNN-COMP `pgd_2_3_16` (77% of width in attention). A 3-step debug run of the
learner confirms it: gradient norm on G ≈ 7e-7, held-in eval unchanged to 4 decimals. The GenBaB filtering (only
instances vanilla CROWN fails) makes stock vanilla CROWN 0/72 by construction (mean min-lb −0.0135), which makes no
difference here.

Stock official run (GenBaB config, alone on the L40S 44 GB): 24 instances finished (21 safe, 2 safe-incomplete,
1 unknown) before CUDA OOM at instance 25 inside BaB (`batch_size: 50` was tuned for a bigger GPU). Not rerun — the
rewrite cannot change anything on these models. Negative, mechanistic, closed.

**Next target: DeepT (Bonaert et al., PLDI 2021) pretrained SST transformers** (`eth-sri/DeepT`, 735 MB clone,
git-ignored under `deept_benchmarks/`): BERT-style, hidden 128, 4 heads × 32, 3/6/12 layers, ReLU MLP, 'no_var'
LayerNorm (mean-subtraction: linear), tanh pooler; spec = ℓ∞ ball of radius eps around the embedding of ONE word
of a test sentence (CLS/SEP/word-pieces excluded), property = true-label logit stays larger; DeepT reports the max
certified eps per (sentence, position) on 10 seed-0 test sentences (reference small_3 ℓ∞: mean 0.0327 over 117
positions). Harness `NNs/transformer_rewrite/deept_gauge.py`: clean single-input forward over their modules is
bit-identical to their `forward(embeddings=)` (fp64 diff 0.0); their exact sentence sample is not reproducible
(our seed-0 sample: 155 positions, theirs 117 — different RNG consumption), so we use our own sample and say so.
auto_LiRPA gotcha: default `sparse_intermediate_bounds=True` makes lse-CROWN peak 18.7 GiB at 12 tokens (OOM at 16);
`False` gives the identical bound at 0.24 GiB. In grad mode (needed for gauge learning / alpha-CROWN) the retained
graph costs 21 GiB at 12 tokens and OOMs at 20 → tuning and the full tier are restricted to sentences ≤ 12 tokens.

**Full-tier baseline for stock ibp_3_3_8 (official vit.yaml pipeline, alone on g3120, finished 15:21): 59/100
verified (40 safe by BaB + 19 safe-incomplete by alpha-CROWN), 41 unknown; initial complex-mode CROWN verifies
0/100 (mean min-lb −0.046); alpha-CROWN verifies 8.16/9 specs on average and hit its 30 s cap on 81/100 instances.**
So the ibp_3_3_8 headroom at the tier that matters is 41 instances, but the gauge (3% attention share) cannot reach
it — consistent with the vanilla-tier neutrality above. (`results/official_stock_ibp.json`.)

### DeepT `sst_bert_small_3`: the gauge transfers, weakly (vanilla CROWN tier, out-of-sample, two seeds)

Setup (`NNs/transformer_rewrite/deept_gauge.py learn/eval`): tuning boxes = 119 one-word ℓ∞ boxes on 40 SST **dev**
sentences ≤ 10 tokens (3 random positions each), per-box eps = the box's stock certified radius (so the stock bound sits
at ≈0 on every tuning box); objective = mean CROWN(lse) lb, Adam lr 0.01, cond penalty 1e-4, clip 1.0, 120 steps ×
4-box accumulation (12 min per seed on an L40S); best-by-held-in-lb checkpoint. Exactness gate (fp64, random points in
the boxes): 8.9e-16 for both gauges. Evaluation on SST **test** sentences ≤ 12 tokens (40 sentences, 278 positions),
same bisection grid for stock and gauged:

| | stock | gauge seed 0 | gauge seed 1 |
|---|---|---|---|
| mean certified radius | 0.0331 (median 0.0311) | 0.0340 — larger on 211/278, smaller on 13, equal 54 (+1.7%) | 0.0336 — larger 155, smaller 9, equal 114 (+1.2%) |
| verified at eps 0.01 / 0.02 / 0.03 | 256 / 212 / 147 | 256 / 212 / **149** (2 flips up, 0 down) | 256 / 212 / **150** (3 up, 0 down) |
| lb at eps 0.03: tighter / looser | | 224 / 54, mean Δ +0.025 | 136 / 142, mean Δ +0.035 |

So the effect replicates in verdict and in mean radius across two independently learned gauges with zero reverse
flips, but it is an order of magnitude smaller than on the VNN-COMP ViT. Seed 0 is the clean one (tighter lb on 266/278
at eps 0.01, 224/278 at 0.03); seed 1 is mixed per instance below the radius (tighter on 130 vs looser on 148 at
eps 0.01) while still enlarging more radii than it shrinks. fp32-storage check: the newly verified instances at
eps 0.03 have gauged margins ≥ 1.3e-2 (seed 1: 0.0133/0.0295/0.0764; seed 0: 0.0198/0.0298), while the float32 stock
vs gauged logit discrepancy on random points in those very boxes is ≤ 4.8e-7 — four orders of magnitude apart, so the
flips are not rounding artefacts. Held-in gains were much larger (mean lb on tuning boxes +0.05 → +0.20), so
part of the gap is overfitting to 119 boxes; the rest is the ceiling below:

**Attention-slack attribution** (`attrib`: attention probabilities frozen at their box-centre values, which removes the
QKᵀ-bilinear, softmax and AV-bilinear slack; 24 instances, 12 test sentences ≤ 12 tokens): attention nonlinearities
account for **9.2% of the CROWN width at eps = stock radius and 12.8% at 1.5× the radius** (per sentence 1%–24%).
Compare 77% on `pgd_2_3_16` (where the gauge flipped 7/100 at the full tier) and 3% on `ibp_3_3_8` (neutral). The
one-word embedding perturbation barely moves the attention pattern (softmax interval widths 0.001–0.015), so most of
the slack sits in the ReLU MLPs and the tanh pooler, which the gauge cannot touch. A +1–2% radius gain is consistent with a
~10% attention share (this is a plausibility argument, not a derived relationship). Consistent picture across three model families: **gauge leverage ≈ attention share of the
CROWN width**, and that share is a property of model + spec, not of the rewrite.

**Attention share on the deeper DeepT models:** 39.7% (`small_6`) and 70.0% (`small_12`) — see the table in the next subsection.

**Long-sentence transfer on small_3 (seed-1 gauge, 2026-09-06 03:35, job 39641026 `long1`):** 12 test sentences of 10–32
tokens (251 positions; 236 of them longer than any tuning sentence). Stock mean radius 0.0341 (median 0.0281); gauged
0.0343 (+0.7%): larger on 93, smaller on 2, equal 156; by length 13–20 tokens +1.1%, 21–26 +0.6%, 27–32 +0.6%. Fixed
eps 0.005/0.01/0.02: verified 251/251/227 → unchanged, lb tighter on ~half (132/124/127 of 251), 0 flips. So the
small_3 gauge tuned on ≤ 10-token boxes is essentially neutral (never harmful) at 3× the tuning length — the gauge's
small effect on this model does not grow or reverse with length, it fades. Seed-0 gauge (job 39641025, 05:25): the same picture but uniformly tighter — lb tighter on **251/251** at all three eps (mean
Δ +0.0003 / +0.0013 / +0.0065), radius 0.0341 → 0.0344 (+0.8%; larger on 140, smaller on 0, equal 111), verified counts
unchanged, 0 flips; fp64 gate 8.9e-16. Job 39641025 complete.

**Overfitting test on small_3 (all dev positions), 2026-09-06 00:20 — the ceiling explanation wins.** Learner on ALL
positions of the 53 correctly classified dev sentences ≤ 10 tokens (305 boxes, 2.6× the 119 before), 300 steps × 8-box
accumulation (best step 270; held-in mean lb +0.07 → +0.64, three times the 119-box learner's +0.20). Same 278 test
positions: mean radius 0.0331 → 0.0337 (per-instance +1.3%; larger on 126, smaller on 81, equal 71), eps 0.03 verified
147 → **152** (5 up, 0 down), but the per-instance lb is mixed (eps 0.03: tighter 116 / looser 162, mean Δ +0.031). So
2.6× more tuning data and 3× the held-in gain buy no extra test gain (+1.3% vs +1.7%/+1.2% for the two 119-box gauges):
the small_3 gap is the ~9% attention-share ceiling, not overfitting. Gauge `gauges/deept_small3_alldev_seed0.pt`,
results `results/deept_small3_alldev_eval_short_seed0.json`.

**Alpha-CROWN tier on small_3 (2026-09-06 01:18, `deept_chain.sh alpha`, job 39641025):** paired stock vs seed-0 gauge with
`CROWN-Optimized` (20 iterations, autograd on) on the 44 test sentences ≤ 8 tokens (205 positions; the 44 GB limit for the
3-layer alpha graph). eps 0.03: alpha-CROWN verified 104/205 → 104 (vanilla CROWN 83 → 83), lb tighter on **202/205**,
looser 3, mean Δ +0.015; eps 0.04: 54 → 54 (CROWN 44 → 45), tighter 162/205, looser 43, mean Δ +0.025; 0 flips either
way. So on small_3 the gauge's tightening survives alpha optimisation almost instance-for-instance but is too small to
flip verdicts — the same +1–2% story as the vanilla tier, as the 9% attention share predicts. The analogous alpha-tier
eval for small_6 (share 40%) was tried at ≤ 6 tokens (job 39658740, `alpha6`, 15 sentences / 49 positions): **CUDA OOM at
44 GB** on the first alpha pass — alpha-CROWN (which optimises intermediate bounds too) retains far more than the
plain grad-mode CROWN used by the learner, which fit ≤ 8-token sentences on this model. A ≤ 5-token retry was
queued (job 39662432, `alpha5`) and later cancelled as superseded by the A100 result (see "small_6 alpha tier" below).

**small_12 learner: CUDA OOM (2026-09-06 01:55).** With autograd on, the 12-layer graph does not fit 44 GB even at 7-token
sentences (35 boxes; the first backward OOM'd at 43.9 GiB). Retry at ≤ 6 tokens (job 39659465, `small12b`: 5 dev sentences, 17 boxes) **also OOM'd at 44 GB** (03:59); its automatic
fallback at ≤ 5 tokens (2 sentences, 5 boxes — too few to mean much) did run and its paired eval is job 39659465 (result appended below when it lands). Independently of memory, small_12 is a
poor target for the *radius* metric: the stock lb at the bisection radius is +1.2 to +2.2 on every tuning set, i.e. the
"radius" is where lse-CROWN turns NaN, not where the bound crosses zero (contrast small_6 below: `diagnostics/_nan_cliff_probe.py`, 40 sampled test instances: stock radii are
zero-crossing-limited on 40/40, gauged on 39/40 — one gauged radius is set by the NaN cliff, i.e. the gauged bound is
still positive when lse-CROWN turns NaN, so that instance's +Δ is if anything understated). A small_12 statement would need the NaN-free `complex` softmax mode or an 80 GB card, and is left
out of the goal. Note also that on small_12 the bisection radius is set by the lse NaN
cliff, not by the bound crossing zero (stock lb at the found radius is +1.22 on the tuning boxes, cf. the attribution
run where only 2/24 instances were finite at 1.5× radius) — the same question for small_6 was settled by
`diagnostics/_nan_cliff_probe.py` on a GPU (the CPU attempt was killed as too slow): small_6 radii are zero-crossing-limited (40/40 stock, 39/40 gauged; see below).

### DeepT `sst_bert_small_6`: the gauge result REPLICATES at full strength (vanilla CROWN tier, out-of-sample) — 2026-09-05 ~23:00

Predicted by the leverage rule (attention share 39.7% vs 9.2% on small_3) and confirmed. Same protocol as small_3
(`deept_chain.sh small6`, batch job 39641026): learner on the 23 SST **dev** sentences ≤ 8 tokens (68 boxes, 3 positions
each, per-box eps = stock radius, mean 0.0184), 120 steps × 4-box accumulation, 25 min; best step 90, held-in mean lb
+0.16 → +1.18; fp64 exactness gate 8.9e-16 (learner and eval). Paired eval on 40 SST **test** sentences ≤ 12 tokens
(294 positions; the tuning regime was ≤ 8 tokens, so 267 of these are longer than anything tuned on):

| small_6, 294 test positions | stock | gauge seed 0 |
|---|---|---|
| mean certified radius | 0.0220 | **0.0249 (+13.1%)**; larger on 273/294, smaller on 0, equal 21 (grid resolution); per-instance +10.6% mean, max +36% |
| by length: 5–6 / 7–8 / 9–10 / 11–12 tokens | 0.0146 / 0.0239 / 0.0223 / 0.0218 | +14.5% / +21.5% / +13.3% / +11.7% (gain persists beyond the tuning lengths) |
| verified at eps 0.01 / 0.02 / 0.03 | 275 / 181 / 41 (20 lse-NaN at 0.03) | 275 / **183** / **95** (6 NaN); flips up 0 / 2 / 40 finite + 14 NaN→verified, **0 reverse** |
| lb at eps 0.03: tighter / looser | | 274 / 0, mean Δ +1.62 (eps 0.02: 277 / 17, +0.26) |

**Alpha-CROWN tier on small_6 (2026-09-06 08:16, A100 80 GB, job 39663976, `deept_chain.sh alpha6` with the fresh-module-per-call
`eval_alpha`):** paired stock vs seed-0 gauge with `CROWN-Optimized` (20 it) on the 15 test sentences ≤ 6 tokens (49 positions;
the 6-layer alpha graph peaks at 62 GiB there). **eps 0.02: alpha-CROWN verified 24/49 → 27/49 (3 up, 0 down), lb tighter on
47/47 finite pairs, looser 0, mean Δ +0.33**; NaN 2 → 0; newly verified margins 0.157 / 0.137 / 0.037 (stock −0.019 / −0.038 /
−0.264). Vanilla CROWN on the same instances 11 → 15. eps 0.03 is NaN-dominated at these lengths (stock 47/49 NaN) — the gauge
reduces NaNs to 35 and 4 instances become verified (margins ≥ 1.66), but it is not a meaningful comparison. So on small_6 the
gauge's gain survives alpha optimisation: +3/49 verdicts (+12% of the verified set) and tighter on every finite instance —
the same shape as the VNN-COMP ViT full-tier result (+7/100, 0 reverse). BaB was not run (no abcrown Customized-loader wiring
for DeepT; the alpha graph already needs an 80 GB card).

Newly verified margins ≥ 1.8e-2 (eps 0.02) and ≥ 5.5e-2 (eps 0.03), vs fp32 stock-vs-gauged logit discrepancy 4.8e-7
measured on small_6 itself (32 random points in each newly verified eps-0.03 box) — five orders of magnitude. The learned
gauges are mild (per-head condition numbers 1.3–2.3), and they also reduce lse-CROWN NaNs at eps 0.03 (20 → 6), i.e. keep
the softmax relaxation finite further out. The eps 0.03 count more than doubles (41 → 95 of 294) with zero instances getting worse — this is the
same shape as the VNN-COMP ViT result (initial CROWN tighter on 100/100, 7 unknown→verified, 0 reverse), now on a
second real, downloaded transformer with its authors' spec. Three DeepT depths give a within-family dose-response:

| model | attention share of CROWN width (eps = radius) | radius gain (out-of-sample) | eps-0.03 verified |
|---|---|---|---|
| small_3 | 9.2% | +1.7% / +1.2% (two seeds) | 147 → 149 / 150 of 278 |
| small_6 | 39.7% | **+13.1%** (alpha tier, ≤ 6 tokens, eps 0.02: 24 → 27 of 49, tighter 47/47) | 41 → 95 of 294 |
| small_12 | 70.0% | **+26.6 %** (5-box ≤ 5-token gauge; 120 test positions ≤ 10 tokens; larger 120/120) — but the stock radius is NaN-cliff-limited, so part of this is the cliff moving, not the zero crossing; see footnote | 67 → 96 of 120 at eps 0.01 (NaN 39 → 13), 0 reverse |

Ops note (2026-09-05 20:15): the interactive allocation 39619518 ended (job COMPLETED after 6 h 50, the user's
interactive step cancelled), which killed both long-sentence paired evals ~1.8 h into their gauged half (stock half
done: mean radius 0.0341, median 0.0281 over 251 positions; eps 0.005/0.01/0.02 verified 251/251/227) and made the
queued chains fail instantly ("Slurm job has expired"). Resubmitted as two batch jobs on `gpu-l40s`/`gpu-l40s-amath`
(1 L40S each, 16 h): 39641025 = alpha-tier eval → small_12 learner+eval → long eval seed 0; 39641026 = small_6
learner+eval → all-dev learner+eval → long eval seed 1 (`NNs/transformer_rewrite/deept_chain.sh`). The allocation's end coincides
with the user's login session on klone-login03 closing (20:13); nothing was cancelled by this session. Bookkeeping caveat:
part of this session's context was dropped mid-evening, so the queue notes above were written twice with slightly different
decisions (the all-dev learner was first dropped, then re-queued); the batch-job list in this note is authoritative.

Caveats: the DeepT reference sentence sample is not reproduced (different RNG consumption; our seed-0 sample has 155
positions vs their 117), so DeepT's reported radii (mean 0.0327 for their zonotope on small_3) are on different
instances. The alpha-CROWN / BaB tier is only reachable for ≤ 8-token sentences on 44 GB (alpha-CROWN is +40% tighter
than CROWN there: +0.94 → +1.32 at 6 tokens, +2.39 → +3.49 at 8) — a paired alpha-tier eval on the 44 such test
sentences (205 positions) was the remaining step for the full-tier statement (done: tighter 202/205, no flips, see "alpha tier" below). A long-sentence eval (≤ 32 tokens, 251
positions, 12 sentences) was started to test transfer beyond the ≤ 10-token tuning regime; it was lost with the allocation and rerun as batch jobs (done: +0.7/+0.8 %, no flips, see below).

### Deeper DeepT models have a much larger attention share → they are the better targets (2026-09-05 evening)

Same `attrib` probe (24 instances, 12 test sentences ≤ 12 tokens, attention probabilities frozen at the box centre) on the
6- and 12-layer DeepT SST models, run alongside the long evals:

| model | attention share of CROWN width at eps = stock radius | at 1.5× |
|---|---|---|
| `sst_bert_small_3` | 9.2% | 12.8% |
| `sst_bert_small_6` | **39.7%** (per sentence 0%–86%: 22, 80, 1, 65, 15, 33, 16, 85, 0, 86, 15, 24) | 33.5% (14 finite instances; the rest NaN in lse mode) |
| `sst_bert_small_12` | **70.0%** (24 instances; per sentence 29%–99%: 29, 71, 66, 71, 94, 99, 82, 51, 77, 97, 38) | only 2 finite instances (lse NaN at 1.5× radius on this depth) — not meaningful |

Interpretation: with more layers the one-word perturbation propagates into the attention patterns of later layers, and the
bilinear/softmax slack compounds, so the attention share — and by the leverage rule the gauge's room — grows with depth.
Decision: the queued "all-dev-positions small_3 learner" (overfitting test) was **dropped from the queue before it started**
(its waiting shell was killed; no GPU job was cancelled) in favour of learning + evaluating gauges on `small_6` and then
`small_12` (`run_deept_model_chain.sh`, same protocol as small_3: dev-set tuning boxes, 120 steps × 4-box accumulation, seed
0, paired eval on 40 test sentences ≤ 12 tokens; deeper models tune on ≤ 10 / ≤ 8-token sentences because grad-mode memory
scales with depth, with an automatic fallback to shorter sentences on OOM). The alpha-CROWN paired eval of the small_3 seed-0
gauge (44 test sentences ≤ 8 tokens, eps 0.03/0.04) stayed queued on the other GPU (done, see "alpha tier" below).




**small_12 learner OOM (2026-09-06 01:55, job 39641025 `small12`):** the 12 dev sentences ≤ 7 tokens gave 35 tuning boxes
(stock radii are tiny on the 12-layer model: mean 0.0070, max 0.0094 — it is far less certifiable than small_3/6), radii and
the no-grad sharing check ran, but the first grad-mode CROWN on the 12-layer graph exceeded 44 GB (43.85 GiB allocated by
PyTorch). The chain continued with `long0`. Resubmitted as job `deept_D` (`deept_chain.sh small12b`): tuning sentences
≤ 6 tokens (5 positions each), automatic fallback to ≤ 5 tokens, then the short paired eval at eps 0.005/0.01/0.02 (the
eps list is scaled to small_12's radii). Queued behind the group GPU limit (2 concurrent L40S) with the small_6 alpha eval.

**Probe result (03:50, job 39662749, `diagnostics/_alpha_mem_probe.py`): the weight-gradient hypothesis is rejected.** small_3 at
6 tokens: peak 14.6 GiB with weight gradients vs 16.4 GiB with weights frozen, identical alpha bound (−0.60991); small_6 OOMs
above 43.6 GiB at 6, 7 and 8 tokens with weights frozen. alpha-CROWN on the 6-layer model simply needs more than 44 GB for
≥ 6-token sentences (it retains ~3× what the 3-layer model does, more than the layer count alone would suggest). Options:
the ≤ 5-token retry (`alpha5`, job 39662432, later cancelled) or an 80 GB GPU (taken; result below).
Submitted the ≤ 6-token small_6 alpha eval on an **80 GB A100** via the checkpoint partition (job 39662797, `ckpt-all`/`ckpt-amath`,
`run_on_bigger_gpu.sh alpha6`, which aborts if the card has < 60 GB; preemptible, so it may restart). The ≤ 5-token L40S retry
(job 39662432, submitted by the parallel instance) stayed queued as the fallback and was cancelled once the A100 result landed.

**04:35 — the A100 run OOM'd too** (job 39662797: 78.6 GiB allocated at 6 tokens; the small_12 ≤ 6-token learner likewise at
42.9 GiB on the L40S, now retrying at ≤ 5 tokens, which gives only 5 tuning boxes). alpha-CROWN on the 6-layer graph is not
simply "3× small_3": either the per-instance peak is really > 80 GB or memory accumulates across consecutive instances inside
the reused BoundedModule. Probe 2 (`diagnostics/_alpha_mem_probe2.py`, job 39663682, A100) measures allocated-before/after and
per-call peak over 4 consecutive instances at 5 and 6 tokens, with the module reused (as `eval_alpha` does) and rebuilt per call.

**Probe 2 result (05:05, job 39663682, A100 80 GB): no leak, but retained state + peak overflowed.** small_6 alpha-CROWN per-call
peak is 36.4 GiB at 5 tokens and 61.9 GiB at 6 tokens (stable over 4 consecutive instances, ~72 s each); after each call the
BoundedModule *retains* 4.5 GiB (5 tokens) / 7.9 GiB (6 tokens) of alpha/bound state. `eval_alpha` kept one module per sentence
length alive across the loop and had the weights with `requires_grad`, so the 6-token peak sat on top of ~12 GiB of retained
state and overflowed 80 GB (78.6 GiB at the OOM). Fix in `deept_gauge.py eval_alpha`: a fresh BoundedModule per call (deleted
after, ~1 s overhead) and frozen weights (probe 1: identical bound). Peak then ≈ 62 GiB at 6 tokens → resubmitted on the A100
(job `deept_a100b`, `alpha6`: 15 test sentences ≤ 6 tokens, 49 positions, eps 0.02/0.03, ~49 × 4 × 75 s ≈ 4 h). 7 tokens
would need ≈ 84 GiB, so ≤ 6 tokens is the ceiling for the 6-layer alpha tier on this cluster.

### Wrap-up of the replication goal (2026-09-06 ~08:40)

**Verdict.** The attention-gauge result (learned exact per-head reparametrisation tightens CROWN-family bounds) replicates on
one downloaded transformer that ships with its authors' verification spec: DeepT's `sst_bert_small_6` (6 layers, SST-2 word
substitution, ℓ∞ radius on one word embedding as in the DeepT paper). Tuning used 68 boxes from 23 dev sentences ≤ 8 tokens;
the evaluation is out-of-sample (40 test sentences, 294 positions ≤ 12 tokens, 267 of them longer than any tuning sentence):

| tier | stock → gauged |
|---|---|
| vanilla CROWN, certified radius (294 positions) | +13.1 % mean; larger on 273/294, smaller on 0 |
| vanilla CROWN, verified at eps 0.03 | 41 → 95 of 294, 0 reverse |
| alpha-CROWN (A100, 49 positions ≤ 6 tokens), eps 0.02 | tighter on 47/47 finite pairs, looser on none; 24 → 27 verified, 0 reverse (small sample) |

It transfers weakly on the shallower sibling `small_3` (+1.7 % / +1.2 % radius over two seeds, tighter 202/205 at the alpha
tier, no verdict flips at any tier or sentence length) and is mechanistically null on the GenBaB ViTs (attention constant over
the boxes). All of it is consistent with the leverage rule: the gain tracks the share of the CROWN width attributable to the
attention nonlinearities (9 % small_3 → +1.7 %; 40 % small_6 → +13 %; 77 % pgd ViT → +7/100; 3 % ibp ViT and 0 % GenBaB →
neutral). That is a two-point dose-response inside DeepT plus the ViT points; `small_12` (70 %) is *not* a third point — its
learner OOMs at 44 GB for ≥ 6-token sentences and its bisection radius is set by the lse NaN cliff rather than a zero crossing
(the 5-box ≤ 5-token gauge eval, job 39659465, is appended below as a footnote only).

Not done, and why: the BaB tier for DeepT. The abcrown loader was never wired for these models, but the binding reason is
memory, not plumbing — a single alpha-CROWN call already peaks at 36 GiB (5 tokens) / 62 GiB (6 tokens) on small_6 and only
fits the 80 GB A100 for ≤ 6-token sentences; BaB multiplies that by the number of live domains, so the model where the effect is
large (+13 %) cannot reach BaB on this hardware, and on small_3, where BaB would fit, the effect (+1.7 %) is below what a
time-capped BaB verdict count can resolve. The alpha-CROWN tier is therefore the top tier reported for DeepT (the ViT result has
the full official BaB pipeline). Code provenance of the alpha numbers: small_3's row came from the earlier `eval_alpha` (module per sentence length,
weights with autograd on), small_6's from the current one (fresh BoundedModule per call, frozen weights, needed to fit the
A100). A direct check (job 39666884, L40S, 3 small_3 test sentences = 18 positions, eps 0.03) reran those instances through the
current code: all 18 alpha-CROWN and CROWN bounds, stock and gauged, are **bit-identical** to the earlier file (max |Δ| = 0;
`results/deept_small3_eval_alpha_check.json`). The two alpha rows are therefore directly comparable.

Disclosures for this section: the two long-sentence evals were lost with the user's interactive allocation and rerun as batch
jobs; two duplicate job submissions (39659490, 39662432) were cancelled by me; the CPU NaN-cliff probe was killed as too slow
and rerun on a GPU (first GPU attempt failed on a device mismatch, fixed); a duplicate instance of this session ran concurrently
for several hours (its A100 submissions and the `eval_alpha` fix produced the small_6 alpha-tier result; it also consumed L40S
and ckpt-amath hours); the stock GenBaB official run noted earlier was not rerun; `git push` is blocked by the permission
classifier, so `main` is ahead of origin locally until the user pushes.

### Prior-art check: Huang, Wei, Isac, Wu, Wu, Barrett, "Parameterized Abstract Interpretation for Transformer Verification" (AAAI-26) — 2026-09-06 ~11:00

User asked whether the gauge rewrite is still novel against https://ojs.aaai.org/index.php/AAAI/article/view/40860 (code:
https://github.com/huangdiudiu/PBVerification-for-Transformers, a fork of Shi et al. 2020's `main.py` verifier — the same
codebase DeepT's `sst_bert_small_*` checkpoints live in, so it slots into our pipeline).

What they do: keep the network fixed and parameterise the affine relaxation of every scalar product xy inside QKᵀ and
V·softmax — either tangent planes to a mean-gap-optimal quadratic bound (PBverifierT) or a convex combination of Shi et al.'s
two McCormick planes (PBverifierI, their eq. 20) — and optimise the parameters per verification query by gradient descent on
the final bound (their eq. 26; optionally per-layer widths, eq. 25). Baseline = Shi et al. (18a,b). Models: Shi-style
SST/Yelp BERTs, N ≤ 3 layers, 4 heads, hidden 256; one perturbed word embedding; radius by bisection on 50/60 instances.
Their gains (ratio of mean radii): SST ℓ∞ 1 layer 0.0346 → 0.0348 (tie), 2 layers +2.0 %, 3 layers 0.0204 → 0.0210
(+2.9 %, wins 44/50); Yelp ℓ∞ 3 layers +8.7 %; interval widths "advantage becomes more pronounced for deeper layers". Cost ≈ 5×
verification time (100 s vs 21 s per query at 3 layers).

Relation to the gauge rewrite, checked in code: auto_LiRPA's `MulHelper.interpolated_relaxation` (used by `BoundMul` and by
`BoundLinear.bound_backward_with_weight`, i.e. the QKᵀ and V·P matmuls) is exactly Shi's (18a,b) in plain mode and exactly
the PBverifierI family (r_l, r_u ∈ [0,1] per product, learned in `opt_stage`) under `CROWN-Optimized`. So our "vanilla CROWN"
tier IS their Baseline, and our alpha-CROWN tier already runs their interpolated per-query optimisation (final-bound objective
only, intermediate boxes fixed at plain CROWN; plus softmax/ReLU alphas). The alpha-tier pairings are therefore the
composition test: on top of the PBverifierI-family relaxation the gauge still tightens 47/47 (small_6), 202/205 (small_3),
100/100 initial + 41 → 52 alpha verified (pgd ViT). The two methods drain the same slack from different sides — theirs moves
the plane inside the McCormick hull of the *original* per-coordinate boxes; the gauge changes which quantities get
box-concretised (the CROWN box on Gᵀq is not derivable from the box on q), so neither subsumes the other (their tangent family
reaches off-centre planes the gauge never picks). Like-for-like: their SST 3-layer ℓ∞ +2.9 % vs our DeepT small_3 seed 0
+2.7 % (ratio of means; the diary's +1.7 % is the per-instance mean ratio) on the same model family / threat model; our small_6
+13.1 % (273/294, 0 smaller) has no counterpart (they stop at 3 layers). Their depth trend independently corroborates the
leverage rule. The gauge symmetry itself is known (Wang & Wang, NeurReps 2025, characterise the full per-head GL(d_k)×GL(d_v)
group; no verification use); a brief search found no prior use of a learned gauge to move bound-propagation boxes.
Honest deltas: theirs needs no tuning data and is never worse than Baseline in principle; ours costs a one-off GPU-hour of
learning, adds zero verification-time overhead, and has no out-of-sample guarantee (0 reverse on small_6; neutral on ibp ViT).
Full write-up of this and the other related work (verifier-side transformer relaxations, symmetry work) lives in
`NNs/transformer_rewrite/RELATED_WORK.md`. Follow-up launched at the user's request (11:17): their verifier (`origin` = Baseline,
`originPlus` = PBverifierI, `bilinear` = PBverifierT) on stock vs gauged DeepT small_3/small_6 checkpoints exported by
`NNs/transformer_rewrite/export_gauged_ckpt.py` — jobs 39672839/40 (L40S) and 39672841/42 (A100 ckpt); their `model_sst_{1,2,3}`
checkpoints are not in their data archive (only data + BERT base), so learning a gauge on their models would need retraining.

**small_12 footnote eval cancelled and resubmitted (2026-09-06 11:24).** Job 39659465 (`deept_chain.sh small12b`) had spent
5 h 20 on the *stock* half of the 290-instance eval without printing its summary (small_6's stock half took 66 min; the
12-layer graph is far slower per CROWN call, ~4 s/call at ≤ 7 tokens from the learner's box-radius timing). The gauged half
needs the same again, so the job could not finish before its 13:40 hard limit (`scontrol update TimeLimit` is denied to
users), and `cmd_eval` only wrote its JSON at the very end — so I cancelled it (`scancel 39659465`; ~7.7 GPU-hours lost,
the learner's gauge `gauges/deept_small12_seed0.pt` survives). Fixes: `cmd_eval` now dumps a partial JSON after each half;
resubmitted as job 39672903 `deept_s12e` (L40S, 14 h) with a smaller footnote eval: 20 test sentences ≤ 10 tokens, 8
bisection iters, eps 0.005/0.01/0.02, log `_scratch/deept_eval_short_small12.log` (the timed-out log is kept as
`deept_eval_short_small12_timedout_290inst.log`). An intermediate resubmission (job 39672899) was cancelled after 19 s
because it raced the code patch. The job queues behind the two L40S PBVerification jobs (account limit 2 concurrent).


**PBVerification composition runs, first pair (12:00).** small_3 under their Baseline (`origin`, 15 sentences × positions 1–3
= 35 instances, ℓ∞, 8 bisection steps): stock 0.03667 → gauged 0.03617 (−1.3 %; larger 10 / smaller 18 / equal 7). The same
35 instances under auto_LiRPA CROWN: 0.03605 → 0.03663 (+1.6 %; 30 / 3 / 2). So the small_3 gauge's gain is
verifier-specific — it was learned against auto_LiRPA's relaxation and does not carry to Shi's (different softmax relaxation;
their stock radii differ from auto_LiRPA's per instance by up to ±40 %). Details and table in
`NNs/transformer_rewrite/RELATED_WORK.md`. small_6 pair (+13 % in auto_LiRPA) pending (job 39672839, gauged half running).

**PBVerification composition runs, small_6 Baseline pair (12:35, job 39672839):** stock 0.01686 → gauged 0.01912 = **+13.4 %**
mean certified radius in *their* verifier (larger on 29/35, smaller on 6, median +14.6 %, range −4.7 … +30.4 %) — the same
size as the +13.1 % under auto_LiRPA. The small_6 gauge is therefore not an artefact of auto_LiRPA's relaxation: it
tightens Shi et al.'s independently implemented Baseline as well. Cross-check on the same 35 instances under auto_LiRPA (job 39673979,
13:13): +17.5 % (larger 35/35). Their Baseline is much looser than auto_LiRPA on the 6-layer model (stock radii ≈ half),
yet the gauge lifts both by a similar fraction. PBverifierI/T pairs still running.

**Their own model (13:41–14:05).** Their archive ships no trained models, so `model_sst_3` was retrained with their script (5 min on
an A100, test acc 0.836; their attention is bias-free, harness loads it with zero biases and `pbv_harness_check.py` verifies the
logits). Running: their Baseline/PBverifierI/PBverifierT on it with the paper's protocol (jobs 39676401/2) = a direct reproduction
attempt of their SST 3-layer row; and the full gauge pipeline on it (job 39680163: attrib → learn → auto_LiRPA eval → export →
their verifier on stock vs gauged). Also landed: small_3 PBverifierI pair −1.0 % (21/12/2, neutral like its Baseline pair). See
`NNs/transformer_rewrite/RELATED_WORK.md`.

**Gauge on THEIR model (16:27, job 39686146).** `model_sst_3` (their code, hidden 256, bias-free attention): attention share
26.7 %; learner margin +0.107 → +1.002 on the tuning boxes (fp64 gate 8.9e-16); out-of-sample (276 test positions ≤ 12 tokens)
certified radius **0.0199 → 0.0218 (+9.5 %), larger on 244, smaller on 0**; eps 0.03 verified 36 → 79, eps 0.02 131 → 153, 0
reverse; tighter 276/276 at every eps. Three times the +2.9 % their AAAI-26 method reports on this configuration. Their verifier
on stock vs gauged exports is running. Details: `NNs/transformer_rewrite/RELATED_WORK.md`.

**17:30 — alpha tier on their model.** alpha-CROWN OOMs at 80 GB for ≤ 8 and ≤ 6-token sentences on `model_sst_3` (hidden 256);
at ≤ 5 tokens (29 instances) the gauge is tighter on 29/29 at both eps (0.02: +0.35; 0.03: +1.08), no flips because both eps
exceed these sentences' radii (rerun at 0.01/0.015 queued). Ops: the paper-protocol reproduction jobs on their model were
restructured — 39676401 was preempted at 2 h 50 with nothing saved and 39676402 could not finish 60 PBverifierT positions in
9 h; both cancelled, replaced by five single-run jobs (Baseline gauged 20 sentences; PBverifierI/T stock+gauged 8 sentences).

**18:10 — small_6 under their PBverifierT (job 39673560):** stock 0.01486 → gauged 0.01535 = **+3.3 %, larger on 35/35, smaller
on 0** (range +1.2 … +6.9 %). Smaller than the +13.4 % under their Baseline but one-sided. Their PBverifierT itself is −11.8 %
against their Baseline on stock small_6 (14 larger / 21 smaller). PBverifierI pair for small_6 due ~18:45.

**18:20 — small_12 footnote (job 39672903, the other instance's rerun; 2 × 2 h 27).** Gauge learned on only 5 boxes (2 dev sentences
≤ 5 tokens, the largest the 12-layer learner fits in 44 GB), evaluated on 20 test sentences ≤ 10 tokens = 120 positions:
certified radius 0.0095 → 0.0121 (**+26.6 %**, larger on 120/120); eps 0.005 tighter 120/120 (all verified both ways); eps 0.01
verified 67 → 96 (14 flips up, 0 down; NaN 39 → 13); eps 0.02 all NaN both ways. Caveat as before: on small_12 the bisection
radius is where lse-CROWN turns NaN rather than where the bound crosses zero, so the radius gain conflates a real tightening
(the eps-0.005 and finite eps-0.01 deltas are genuine, +1.17 mean) with the gauge pushing the NaN cliff outward (NaN 39 → 13).
It is consistent with the leverage rule's ordering (70 % attention share > small_6's 40 %) but is not a clean third point.
Decomposition from the saved JSON (`results/deept_small12_eval_short_seed0.json`): on the 81 positions whose stock bound is
finite at eps 0.01 the gauged bound is tighter on 81/81 (0 gauged NaNs where stock was finite); of the 29 newly verified
positions at eps 0.01, 14 are zero crossings and 15 are NaN rescues (11 further NaNs become finite but stay unverified, 13 stay
NaN). The only subgroup whose stock radius is provably a zero crossing rather than a cliff — the 14 positions with a finite
non-positive stock bound at eps 0.01 — gains **+16.2 %** (median +14.2 %, range +12 … +44 %), clearly less than the +26.6 %
overall mean, so the cliff shift does inflate the headline radius number; the zero-crossing gain is the fairer figure and sits
between small_6's +13 % and the headline. Per-instance relative radius gain overall: min +5.6 %, median +24.1 %, max +51.9 %.

## More transformers from the DeepT release: Yelp models and SST width / LayerNorm variants (2026-09-06 18:40)

The stop-hook review judged the replication scope too narrow (one family, three depths). The downloaded DeepT release
(`deept_benchmarks/DeepT/Robustness-Verification-for-Transformers`) ships further *separately trained* checkpoints under the
same published one-word ℓ∞ verification protocol, all loadable by `deept_gauge.py` unchanged apart from a data path:
`yelp_bert_small_{3,6,12}` (Yelp Review Polarity, own 45 944-piece vocab, hidden 128, 4 heads), `sst_bert_big_3` (hidden 256,
MLP 512), `sst_bert_smaller_3` (hidden 64), `sst_bert_standard_layer_norm_3` (full LayerNorm instead of the linear
mean-subtraction 'no_var' norm). Yelp is also the second dataset in Huang et al. AAAI-26 (their Yelp ℓ∞ 3-layer: +8.7 %).

Harness: `--data sst|yelp|auto` (auto from the model-name prefix) and `load_yelp` — Yelp has no dev split, so tuning boxes come
from `train.csv` and evaluation from `test.csv` (disjoint); only reviews ≤ 14 words are read (856 of 38 000 test reviews;
12 084 of 560 000 train). Short Yelp reviews are plentiful: ≤ 8 word pieces 212 test reviews (195 / 194 correctly classified by
small_3 / small_6), ≤ 10: 334 (306 / 304), ≤ 12: 492 (450 / 450) — `diagnostics/_count_short_yelp.py`. Word splitting is a regex
stand-in for DeepT's nltk (not in the venv); BERT's basic tokenizer re-splits punctuation so word pieces are unchanged.

New chain `deept_yelp_chain.sh` (separate file so neither instance edits a running script): attribution on all five models
first, then learn (train boxes ≤ 8 tokens, 40 reviews × 3 positions, 120 steps) + paired test eval (40 reviews ≤ 12 tokens) for
yelp small_3 and small_6. The eval eps grid is derived from each model's attribution stock radii (median r → 0.5 r, r, 1.5 r) —
on SST small_6 this rule reproduces the 0.01/0.02/0.03 grid used before (0.0109/0.0219/0.0328). Job `deept_yelp` (L40S, 12 h).

**Preregistered prediction (written before any Yelp number exists):** by the leverage rule, the Yelp gain ordering follows the
attention share — yelp small_6 ≫ yelp small_3; if the Yelp shares match SST's (≈ 9 % and ≈ 40 %) the radius gains should be
≈ +2 % and ≈ +13 % with 0 reverse; stdln3 should have a *smaller* share than small_3 (the variance nonlinearity adds slack
outside attention) and hence a smaller gain; big3 / smaller3 test whether width moves the share at fixed depth 3. The share
numbers will be recorded here as soon as the attribution step lands, before the learners finish.

**18:41 — small_6 under their PBverifierI (job 39672839):** stock 0.01422 → gauged 0.01463 = +2.9 % (larger 18 / smaller 17;
median +4.5 %, range −2.9 … +13.7 %). PBverifierI itself is −15.6 % vs their Baseline on stock small_6 (9/26). All six DeepT
stock-vs-gauged pairs in their verifier are now done; table in `NNs/transformer_rewrite/RELATED_WORK.md`. Their-model runs
(Baseline 20 sentences; PBverifierI/T 8 sentences; stock and gauged) still running.

**18:45 — Yelp attribution landed (before any learner ran); prediction sharpened.** Attention nonlinearities' share of the
CROWN width at eps = stock radius (12 test reviews × 2 positions, seed 3): **yelp small_3 59.2 %** (24/24 finite; 74.5 % at
1.5× radius), **yelp small_6 80.6 %** (23/24 finite; 93.9 % at 1.5×, only 13/24 finite → the NaN cliff is close, as on SST
small_12). Same depth as SST small_3 (9.2 %) and small_6 (39.7 %), so the Yelp models are a much sharper test of the leverage
rule than another SST depth: at *equal depth* the share is 6× higher. Stock radius scale: yelp small_3 median 0.0160 (eval grid
0.008/0.016/0.024), yelp small_6 median 0.0059 (grid 0.0029/0.0059/0.0088); SST small_3 0.0266, small_6 0.0219.
Prediction on record: yelp small_3 gains ≥ SST small_6's +13 % (its share is above small_6's 40 %), yelp small_6 gains more
still but with the NaN-cliff caveat (report the zero-crossing subgroup as for SST small_12); both 0 reverse. Falsifier: a yelp
small_3 gain in the +2 % class would break the rule.

**18:50 — SST width / LayerNorm variants, attribution (depth 3 throughout):** `sst_bert_big_3` (hidden 256) **33.9 %** (24/24
finite; 50.4 % at 1.5×), `sst_bert_smaller_3` (hidden 64) **13.6 %** (12.2 % at 1.5×), `sst_bert_small_3` (hidden 128) 9.2 %.
So at fixed depth the share is not monotone in width (64 → 128 → 256 gives 14 → 9 → 34 %) but big_3 sits in the same class as
SST small_6 (40 %), which predicts a ≈ +10 % gain; smaller_3 predicts the +2–3 % class. Queued as job `deept_width`
(`deept_yelp_chain.sh full:sst_bert_big_3:big3 full:sst_bert_smaller_3:smaller3`, runs when an L40S slot frees) — a width
dose-response at fixed depth to go with the depth one. `sst_bert_standard_layer_norm_3` **cannot be bounded by auto_LiRPA as
built**: the full LayerNorm's `sqrt(var + eps)` hits `convex_concave.py:138 assert x.lower.min() >= 0` — CROWN's linear
relaxation of the squared deviations yields a negative lower bound on the variance, so the sqrt relaxation refuses. This is a
verifier limitation independent of the gauge (the stock model fails the same way before any gauge is applied); the DeepT
release ships this variant because their zonotope verifier handles the full norm, auto_LiRPA's CROWN path as built does not.
Dropped, recorded as such (a variance node bounded by IBP instead of CROWN would be the fix; not pursued).

**19:30 — gauge learned against THEIR verifier (user: "do it").** `pbv_learn.py` differentiates Huang et al.'s bound (patched copy
`PBVerification_grad/`, detach removed) w.r.t. the folded gauge; smoke test OK (small_3: +1.62 → +1.88 in 6 steps, logits
unchanged). Launched: small_6 trained against their Baseline (`origin`) and against their midpoint tangent planes (`inner`), and
their `model_sst_3` against `origin`; each followed by their verifier variants and a paired auto_LiRPA eval (jobs 39699642–53).
Also noticed two new L40S jobs not mine (`deept_width`, `deept_yelp`) — see the note below once identified.

**19:35 — Yelp small_3 RESULT (job 39697769, `results/deept_yelp3_eval_short_seed0.json`): the gauge transfers to a second
dataset.** Gauge learned in 400 s on 113 train-review boxes (40 reviews ≤ 8 tokens × 3 positions; tuning-box mean lb +0.25 →
+1.78; max cond(G) 6.0; fp64 gate 2.7e-15). Held-out test: 40 reviews ≤ 12 tokens = 277 positions, vanilla lse-CROWN, no NaN
anywhere (clean zero-crossing radii, unlike SST small_12):

| | stock | gauged | paired |
|---|---|---|---|
| certified radius (mean) | 0.0223 | 0.0245 | **+9.5 % mean per-instance (+10.1 % ratio of means)**, larger on 263/277, smaller on 1 (−0.5 %, one bisection step), equal on 13 (all at radii < 0.021, no cap hits) |
| eps 0.008 | 255/277 | 256/277 | tighter 275/277 (looser 2: −0.013, −0.001), flips +1 / −0 |
| eps 0.016 | 209/277 | 219/277 | tighter 270/277 (looser 7, worst −1.24), flips +10 / −0 |
| eps 0.024 | 116/277 | **152/277** | tighter 268/277 (looser 9, worst −0.27), **flips +36 / −0** |

Per-instance relative gain: median +9.0 %, 10–90 % quantiles +2.2 … +18.0 %, max +20.9 %; flat across sentence lengths 4–12
(+5.6 … +11.3 %). fp64 gate on 24 test boxes 2.2e-15. **Against the preregistered prediction:** the qualitative part holds —
Yelp small_3 (share 59 %) gains 5.6× SST small_3 (share 9 %) at the same depth and with 0 verified→unverified flips, and the
attention share predicted this before any learner ran. The quantitative part does not: +9.5 % is *below* SST small_6's +13 %
despite a higher share (59 % vs 40 %), so the share orders gains within a dataset but is not a cross-dataset calibration
(the falsifier — a +2 %-class gain — did not occur). Also the first DeepT model with a handful of looser fixed-eps bounds (2/7/9
of 277, none changing a verdict); the learner objective is the mean lb at the tuning radius, and Yelp's train/test split is
looser than SST's dev/test. For scale: Huang et al.'s PBverifier gains +8.7 % on their Yelp ℓ∞ 3-layer model over their baseline
— a different model and verifier, but the same order of magnitude as this rewrite-only gain on the stock verifier.

**20:40 — SST big_3 RESULT (hidden 256, depth 3; job 39697876, `results/deept_big3_eval_short_seed0.json`): the prediction
from the width attribution holds.** Attribution said 33.9 % attention share → "≈ +10 %" (small_6 class). Gauge learned in 715 s
on 70 dev boxes (24 sentences ≤ 8 tokens; tuning mean lb +0.15 → +1.18, max cond(G) 8.7, fp64 gate 8.9e-16). Held-out test,
40 sentences ≤ 12 tokens = 288 positions, vanilla lse-CROWN:

| | stock | gauged | paired |
|---|---|---|---|
| certified radius (mean) | 0.0166 | 0.0186 | **+9.3 % mean per-instance (+11.9 % ratio of means)**, larger on 274/288, smaller on **0**, equal on 14 (all with stock radius 0.0003, the bisection floor) |
| eps 0.0096 | 256/288 | 258/288 | tighter 288/288, flips +2 / −0 |
| eps 0.0191 | 116/288 | 129/288 | tighter 288/288, flips +13 / −0 |
| eps 0.0287 | 4/288 | **48/288** | tighter 282/288 (looser 2: −0.006, −0.083; NaN 4 → 3), **flips +44 / −0** |

Per-instance gain median +8.6 %, 10–90 % +1.8 … +18.2 %, max +21.6 %; all sentence lengths 5–12 positive (+4.5 … +19.1 %).
fp64 gate on 24 test boxes 8.9e-16. So at fixed depth 3, doubling the hidden size moved the attention share 9 % → 34 % and the
gauge gain +1.7 % → +9.3 %, i.e. the leverage rule holds along the width axis too, within the SST family where its calibration
was set (small_6: 40 % → +13 %). Two SST points (small_3, big_3) and one Yelp point now sit at depth 3, with gains ordered
exactly by share (9 % → +1.7 %, 34 % → +9.3 %, 59 % → +9.5 %) — monotone, saturating across the dataset boundary.
Remaining in this batch: smaller_3 (share 14 %, prediction +2–3 % class; learner started 20:38) and yelp small_6 (share 81 %,
eval started 20:16, due ≈ 22:30).

**20:41 — their verifier × their protocol × their retrained model (jobs 39686146 / 39696050).** Baseline (`origin`), 20 sentences,
positions 1–3, 10 bisection steps, ≤ 32 tokens, ℓ∞: stock 0.02455 (paper Table 1: 0.0204) → CROWN-gauged **0.02661 = +8.4 %,
larger on 52/52, smaller on 0** (median +5.9 %). Their own method's number on this row is +2.9 % with 44/50 wins. PBverifierI/T
stock-vs-gauged on their model (8 sentences each) and the verifier-trained gauges still running.

**20:58 — PBverifierI on their model (jobs 39696051/52; 8 sentences = 20 instances).** Their method reproduces on their own model:
Baseline → PBverifierI on stock +2.4 % (20/20; paper +2.9 %). Gauge under PBverifierI: +6.1 % (20/20). Both together (Baseline
stock → PBverifierI gauged): +8.6 % (20/20) — near-additive. Recorded in `NNs/transformer_rewrite/RELATED_WORK.md`.

**21:11 — PBverifierT on their model (jobs 39696053/54):** their PBverifierT vs Baseline on stock −1.0 % (3/17; matches their Table 1
tie); gauge under PBverifierT +6.0 % (20/20). Their-model summary with the CROWN-trained gauge in their verifier: Baseline +8.4 %
(52/52), PBverifierI +6.1 % (20/20), PBverifierT +6.0 % (20/20); their methods on stock: PBverifierI +2.4 %, PBverifierT −1.0 %.

**21:35 — SST smaller_3 RESULT (hidden 64, depth 3, share 13.6 %; job 39697876, `results/deept_smaller3_eval_short_seed0.json`):
NEUTRAL and mixed — the low-share prediction holds, and the first DeepT model with reverse flips.** Gauge learned in 546 s on 73
dev boxes (tuning mean lb +0.06 → +0.14, max cond(G) 2.1, fp64 gate 8.9e-16). Test (40 sentences ≤ 12 tokens, 277 positions):
mean radius 0.0384 → 0.0389 (+1.0 % per-instance, +1.5 % of means; median +0.4 %, 10–90 % −3.8 … +7.8 %), larger on 140 /
smaller on **115** / equal 22. Fixed-eps bounds are a coin flip (tighter 148 / 139 / 128 of 277 at the three eps, mean |Δ| 0.01 /
0.15 / 0.52), verified 246 → 246, 167 → 169 (3 up, **1 down**), 88 → **85** (0 up, **3 down**; the lost margins were +0.030,
+0.016, +0.028 → −0.28, −0.25, −0.72). So at a 14 % attention share the dev-tuned gauge does not transfer as a one-sided
improvement: it moves the test bounds by ±0.5 either way and the net is ≈ 0 — the SST small_3 result (+1.7 %, 0 reverse) was the
benign end of the same regime. Rule confirmed in the direction that matters for practice: a low attention share means *don't
gauge* (nothing to gain, small risk of loss), a share ≥ 30 % means a one-sided ≈ +10 % gain with 0 reverse (big_3, small_6,
yelp small_3, and the ViT). Width at depth 3 on SST is therefore 64 → 128 → 256 hidden: share 14 / 9 / 34 %,
gain +1.0 (mixed) / +1.7 / +9.3 % — the two low-share models are both neutral and their ordering is noise; only the jump to
big_3 is a real, share-predicted effect.

**22:15 — verifier-trained gauge on their model, evaluated under auto_LiRPA (job 39700391):** the gauge learned through THEIR
Baseline bound on only 5 boxes gives +7.5 % certified radius under auto_LiRPA (240/276 larger, 0 smaller; eps 0.03 verified
36 → 63) — close to the CROWN-trained gauge's +9.5 %. Its evaluation under their own Baseline (paper protocol) runs until ~02:40.

**22:25 — Yelp small_6 RESULT (share 80.6 %; job 39697769, `results/deept_yelp6_eval_short_seed0.json`): one-sided but
cliff-dominated, and SMALLER than yelp small_3 — the magnitude prediction fails again.** Gauge learned in 1498 s on 115 train
boxes (tuning mean lb +1.61 → +2.79, max cond(G) 4.7, fp64 gate 1.8e-15). Test: 40 reviews ≤ 12 tokens = 277 positions, eps grid
0.0029 / 0.0059 / 0.0088 (this model's stock radii are 4× smaller than yelp small_3's).
- Mean radius 0.0061 → 0.0066: **+4.9 % per-instance (+7.2 % of means)**, larger on 176, smaller on 1, **equal on 100**.
- The NaN cliff dominates and the gauge does *not* move it here: stock NaN 64 / 97 / 127 of 277 at the three eps, gauged 63 / 96
  / 128 (1 rescued). 59 of the 100 equal-radius positions are NaN already at the smallest eps — both bisections hit the same
  cliff, so those radii are numerics, not bounds. On the 213 positions with a finite stock bound at the smallest eps the gain is
  **+6.2 %**; on the 177 positions whose radius moved at all +7.7 % (median +8.7 %, 10–90 % +2.2 … +11.7 %); on the provable
  zero-crossing subgroup at the top eps (71 positions with a finite non-positive stock bound) **+6.4 %** (median +6.8 %).
- Fixed-eps bounds where both are finite: tighter **212/213**, 174/180, **147/148** (looser 1 / 6 / 1, worst −1.8); verified
  191 → 192, 135 → 140, 79 → **90** (+1 / +5 / +11, **0 reverse**). fp64 gate on test boxes 1.8e-15.
Against the prediction: one-sidedness and 0 reverse hold; ">yelp small_3" does not (+6 % on the clean subset vs +9.5 %). Two
reasons, the first visible in the data and the second a hypothesis: (i) the cliff removes the positions where the radius could grow most (the NaN rate at the stock
radius is already 23 %, vs 0 % on yelp small_3 and big_3); (ii) as on SST small_12 vs small_6, once the attention share is
above ≈ 60 % the lse-CROWN bound is numerically fragile and the gauge — which rotates operands but cannot shrink the softmax
input range — has less clean slack to remove. Net reading of the whole batch: the share is a good *screen* (≤ 14 % neutral,
≥ 30 % one-sided gain ≈ +6–13 %, 0 reverse on 5 of 5 such models) but not a dose-response calibration beyond ≈ 40 %.

**Batch wrap-up (2026-09-06 22:30, jobs 39697769 + 39697876, 6 h 31 GPU total).** Four more separately trained transformers from
the DeepT download were run through the unchanged pipeline (attribution → gauge learned on train/dev boxes → paired held-out
eval), with the gain predicted from the attention share before each learner ran. Direction and one-sidedness were predicted
correctly on all four (three one-sided gains with 0 reverse flips, one neutral/mixed at low share); magnitude was predicted
correctly for big_3 (≈ +10 %) and smaller_3 (+2–3 % class) but over-predicted for both Yelp models. Together with the earlier
runs the gauge result now stands on: VNN-COMP ViT (full official BaB pipeline, +7/100), DeepT SST small_6 (+13 %, alpha tier
too), SST big_3 (+9 %), Yelp small_3 (+9.5 %), Yelp small_6 (+5–6 %, cliff-limited), SST small_12 (+16–27 %, cliff-limited); and
is neutral on SST small_3, SST smaller_3, ibp ViT, GenBaB ViTs — every neutral case has attention share ≤ 14 %, every positive
case ≥ 30 %. Correction to the 20:40 reading ("gains ordered exactly by share"): that did not survive smaller_3 (14 % < small_3's
gain) and Yelp small_6 (81 % < Yelp small_3's gain) — the share is a screen with a threshold between 14 % and 30 %, not a
calibration. Unchanged: BaB tier for DeepT out of memory reach (see above); the standard-LayerNorm variant not boundable.

**00:04 — small_6 gauge learned through THEIR Baseline (13 boxes ≤ 6 tokens; job 39703629/31):** under their Baseline +6.9 % vs
stock (18/17, range −10 … +42 %) — worse than the CROWN-trained gauge's +13.4 % (29/6) on the same instances. The thin tuning
set the memory limit forces (13 vs 68 boxes) overfits; verifier-matched training is not automatically better than transfer.

**00:50 — small_6 gauge learned through their midpoint-tangent bound (13 boxes; job 39703635/37):** under their Baseline +13.2 %
vs stock, **larger on 35/35** — matches the CROWN-trained gauge (+13.4 %, 29/6) with a cleaner per-instance picture, while the
Baseline-trained twin from the same 13 boxes overfit (+6.9 %, 18/17). `bilinear` and auto_LiRPA evaluations pending.

**03:19 — midpoint-tangent-trained small_6 gauge under auto_LiRPA (job 39703638):** +9.5 % radius (266/294 larger, 0 smaller) vs
+13.1 % for the CROWN-trained gauge; transfer works in both directions on small_6, the native gauge is best in its own verifier.

**05:00 — Baseline-trained small_6 gauge (13 boxes, cond ≈ 35) under auto_LiRPA (job 39710860): −16.3 % radius, smaller on
289/294.** The overfit verifier-trained gauge that was +6.9 % (mixed) in their Baseline is strongly harmful under auto_LiRPA —
thin tuning sets can produce gauges that help one relaxation and hurt another. (Contrast the tangent-trained twin: +9.5 %.)

**05:04 — Baseline-trained small_6 gauge under their PBverifierT (job 39703633): −6.1 % (9/26)** vs +3.3 % (35/0) for the
CROWN-trained gauge. The overfit gauge helps only the relaxation it was tuned on, and only on average.

**05:47 — their-model gauge learned through THEIR Baseline (5 boxes), evaluated in their Baseline at the paper protocol
(job 39710859, L40S after two ckpt preemptions): +6.0 % (52/52 larger)** vs +8.4 % for the CROWN-trained gauge. Remaining:
PBverifierI on the overfit small_6 gauge and PBverifierT on the tangent-trained small_6 gauge (~11:00).

### Wrap-up of the prior-art / composition thread (2026-09-07 ~06:30; two evaluation jobs still to land)

**Verdict.** Huang et al. (AAAI-26) tighten the *verifier's* product relaxations per query; the gauge is an exact *network*
rewrite. Their PBverifierI family is what auto_LiRPA's CROWN-Optimized already optimises, their Baseline is plain CROWN on
Shi's relaxation. Novelty holds: no prior work found learns a reparametrisation to tighten bound propagation. The fair
comparison — same model (their SST 3-layer, retrained with their script), same verifier (theirs), same protocol — reads:
their PBverifierI +2.4 % over their Baseline (20/20; paper +2.9 %), the CROWN-trained gauge under that Baseline **+8.4 %**
(52/52), under PBverifierI +6.1 % and under PBverifierT +6.0 % (20/20 each; the gauge's Baseline gain on those 20 instances
is +7.5 %, so their optimisation absorbs ≈ 1.5 points and the rest survives). Composability is empirical and conditional
(`NNs/transformer_rewrite/RELATED_WORK.md`, grids 1–2): the 68–120-box CROWN-trained gauges transferred to every relaxation
tested with no reversals except the neutral small_3 cases (small_6: auto_LiRPA +17.5 % on their 35 instances, their Baseline
+13.4 %, PBverifierI +2.9 %, PBverifierT +3.3 %); their own optimised variants fall below their Baseline on stock DeepT
weights (−2.7 … −15.6 %) because of their optimiser's initialisation/best-tracking, not the relaxation family (they reproduce
on their own model).

**Answer to "is that row a gauge trained on PBverifierT?" → it was not; now done** (`pbv_learn.py`, their bound
differentiated; PBverifierT-trained approximated by its unoptimised midpoint-tangent centre). Their model, 5 boxes: +7.5 %
auto_LiRPA (240/0), +6.0 % their Baseline (52/52) — vs +9.5 % / +8.4 % for the 120-box CROWN-trained gauge. small_6, 13 boxes:
tangent-trained generalised (+13.2 % their Baseline 35/35, +9.5 % auto_LiRPA 266/0, cond 2.9/4.5); Baseline-trained overfit
(+6.9 % mixed 18/17, **−16.3 %** under auto_LiRPA 0/289, −6.1 % under PBverifierT; cond 28.3/15.5). The CROWN-trained gauge
remained best or equal in every verifier. The last two cells (PBverifierI on the Baseline-trained small_6 gauge, job
39703632; PBverifierT on the tangent-trained one, 39713376) landed 07:15 and 10:16 and are recorded below.

**Method caveats.** Differentiating their backward bounds is memory-hungry (OOM 80 GB at ≤ 8 tokens small_6, ≤ 6 their model),
hence the 5–13-box tuning sets. Their pooler-tanh slope 1/cosh² overflowed to NaN gradients; the gradient copy uses 1 − tanh²
(`diagnostics/pbv_grad.patch`). After that fix the `origin` training masked no gradient entries, but `inner` training still had
8 192 of 49 152 non-finite entries zeroed at every logged step (source not chased).

**Disclosures (ops).** ckpt-partition preemptions cost repeats (39676401 lost 2 h 50; 39700390 twice; 39703632/36 restarted once,
39703636 then cancelled and its step re-run on L40S as 39713376). Cancelled by me: 39672841 (bilinear chain superseded),
39676401/02 (Table-1 reproduction chains, re-run inside the model chain), 39682385, 39682534, 39686146 (after its `origin_stock`
half saved; a duplicate their-model Baseline half was truncated when two chains briefly overlapped — results unaffected, JSON
written at the end), 39699643–50, 39699652/53, 39700047/48 (dependency chains restructured for the 9 h cap and OOM/NaN failures),
39701472–80, 39701896–99, 39701946–50 (OOM and NaN chains before the ≤ 6 / ≤ 5-token and dtanh fixes), 39703634. The other
Claude session (user's own) committed 07dac29 in between; PROGRESS.md carries both sessions' entries.

**07:15 — Baseline-trained (overfit) small_6 gauge under their PBverifierI (job 39703632, ckpt, restarted once by preemption):
−1.8 % (18/17)** vs +2.9 % (18/17) for the CROWN-trained gauge; head-to-head −4.5 % (6/29). Grid 2 in RELATED_WORK.md updated;
one cell left (PBverifierT on the tangent-trained gauge, job 39713376).

**07:55 — experiment record artifact.** All transformer gauge experiments (ViTs, GenBaB, DeepT SST/Yelp variants, Huang et al.'s
model, composition grids) with metric / baseline / model / gauge-training data stated on every table, and a share-vs-gain chart:
https://claude.ai/code/artifact/6eeacba6-2cea-4410-bae7-10ca1f929c05 (source `gauge_experiments.html` in the session scratchpad;
radius gains recomputed from the eval JSONs as ratio of means, with the per-instance convention alongside).

**10:16 — tangent-trained small_6 gauge under their PBverifierT (job 39713376, L40S): +3.3 % (35/35)**, identical to the
CROWN-trained gauge's +3.3 % (35/0); head-to-head 0.0 % (14/21). Grid 2 is complete: the transferred 68-box CROWN gauge is
best or tied in all four verifiers on small_6 and in all four on their model; training through their bound on the thin sets
memory allows never beat it (tangent-trained tied it under PBverifierT and their Baseline, lagged under auto_LiRPA;
Baseline-trained overfit and was negative in three of four). All jobs of this thread are finished; nothing left in the queue.

**11:05 — out-of-distribution tuning and two-word perturbation on sst_bert_small_6 (user: "is the gauge derivable rather than
learned?" and "do we do multi-token perturbations?").** `deept_gauge.py` gains `--data random` (sequences of 3–6 random whole-word
vocabulary entries, the model's own prediction as label) and cross-dataset use of `--data yelp` with an SST model (Yelp text through
the SST tokenizer, Yelp true labels, correctly-classified filter as usual), plus `--k_words 2` (two embedding rows widened at once;
tuning boxes = 3 random position pairs per dev sentence, eval = up to 7 pairs per test sentence) and a multi-gauge eval
(`--gauge a.pt,b.pt` → tags gauged / gauged2). Jobs: 39722892 `deept_ood_chain.sh yelp` and 39722893 `deept_ood_chain.sh random`
(L40S, 8 h each: learner on 60 sentences ≤ 8 tokens × 3 positions, then the standard paired one-word eval on the same 40 SST test
sentences / 294 positions as the S6 gauge → `results/deept_small6_ood{yelp,random}_eval_short_seed0.json`); 39722894
`deept_2w_chain.sh` (ckpt A100, 9 h: two-word gauge learned on dev pairs ≤ 8 tokens, then a two-word paired eval of stock vs the
two-word-trained gauge vs the one-word S6 gauge, eps grid 0.5/1/1.5 × the median stock two-word tuning radius →
`results/deept_small6_2w_eval_short_seed0.json`). Preregistered reading: if the Yelp- or random-tuned gauge recovers most of the
one-word S6 gain (+13.1 %, 273/0), the gauge is a property of the weights and worth deriving; if the two-word-trained gauge beats
the one-word gauge on two-word boxes by a clear margin, gauges are spec-specific.

**12:28 — split attention-slack attribution on yelp_bert_small_6 (user: why did its +7 % fall short of its 81 % share?; job 39730086,
A100, 28 min; `deept_gauge.py attrib --split_attrib 1`, `results/deept_yelp6_attrib_split.json`).** New diagnostic: besides freezing
the attention probabilities, linearise ONE nonlinearity at the box centre (QKᵀ → q₀kᵀ + qk₀ᵀ − q₀k₀ᵀ; softmax → its Jacobian at
the centre; P·V → p₀v + pv₀ − p₀v₀) or all three, and report the width-weighted share of the CROWN width removed (12 test reviews
× 2 positions, seed 3, same instances as the 80.6 % figure). At ε = stock radius (23 finite instances, mean width 4.48): frozen
attention 80.6 % (reproduces), **QK-only 56.2 %, softmax-only 71.5 %, AV-only 66.8 %, all three 82.3 %**. At 0.5 × radius (width
0.55): frozen 26.1 %, QK 8.5 %, softmax 20.8 %, AV 14.7 %, all 32.7 %. At 1.5 × (13 finite, width 29): 93.9 / 77.0 / 70.9 / 88.3 /
94.5 %. Reading: the three shares overlap almost completely (each alone removes more than half at ε = radius, together 82 %), so
the slack is interactive through the six layers rather than owned by any one nonlinearity; the softmax is the largest single
lever at every ε, but linearising either product alone also collapses most of the width, because the product boxes feed the
softmax inputs and outputs. This does not by itself say whether the gauge's +6–7 % is a ceiling (softmax-owned slack) or an
under-performance (product slack the gauge could reach), because linearising a product removes all of its slack, not only the
basis-dependent part the gauge can move. Control queued: the same split on sst_bert_small_6 (share 40 %, gauge +13 %) and
yelp_bert_small_3 (59 %, +10 %) — if their product shares are similar to Yelp small_6's, the small_6 shortfall is not explained by
where the slack sits; if Yelp small_6 has a much larger softmax-only share relative to its product shares, it is.

**13:05 — split attribution controls (jobs 39741703 sst_bert_small_6, 39741704 yelp_bert_small_3; A100, 22 + 7 min;
`results/deept_{small6,yelp3}_attrib_split.json`).** Width-weighted share of the CROWN width removed at ε = stock radius (24 / 24 /
23 finite instances), listed as frozen attention / QK-only / softmax-only / AV-only / all three:
sst small_6 (gauge +13.1 %): 39.7 / 32.6 / 31.5 / 28.8 / 40.0 %; yelp small_3 (+10.1 %): 59.2 / 35.8 / 34.0 / 44.3 / 61.1 %;
yelp small_6 (+7.2 %, cliff-limited): 80.6 / 56.2 / 71.5 / 66.8 / 82.3 %. Softmax-only as a fraction of the all-three share:
0.79 / 0.56 / 0.87; QK-only: 0.82 / 0.59 / 0.68; AV-only: 0.72 / 0.73 / 0.81. **Verdict: the split does not explain the Yelp small_6
shortfall.** The model that gains most (SST small_6) has a softmax fraction (0.79) close to Yelp small_6's (0.87), and Yelp small_3
gains less than SST small_6 with the *smallest* softmax fraction (0.56); no single-nonlinearity share, absolute or relative, orders
the three gains. On every model the three shares overlap heavily (their sum is 2.1–2.4 × the joint share), i.e. the slack is
interactive across layers rather than owned by one nonlinearity. The remaining explanation for Yelp small_6 is the one visible in
the data: the lse NaN cliff (stock NaN on 23 % of positions at the stock radius, 100 equal-radius positions) and the tiny radius
scale, not a softmax-owned slack the gauge cannot reach. Diagnostic caveat: at 1.5 × radius the softmax-only share collapses
(small_6 4.8 %, yelp small_3 −2.5 %) — the centre-Jacobian linearisation of the softmax over wide score boxes is itself a
loose linear map, so the single-mode shares are only meaningful near the certified radius.

**14:00 — RANDOM-VOCABULARY-TUNED gauge on sst_bert_small_6 (job 39722893; no dataset: 180 boxes on random 3–6-word sequences
of vocabulary entries, the model's own prediction as label; `results/deept_small6_oodrandom_eval_short_seed0.json`), same 294 SST
test positions as the SST-tuned gauge:** mean certified radius 0.0220 → 0.0245, **+11.6 % ratio of means (+9.4 % per-instance);
larger on 271, smaller on 0, equal 23**; eps 0.03 verified 41 → 91 (36 finite flips up + NaN 20 → 6, 0 reverse), eps 0.02 181 → 183;
lb tighter 279 / 286 / 274 of 294 at eps 0.01 / 0.02 / 0.03 (looser 15 / 8 / 0, all below the radius); fp64 gate 8.9e-16. Head-to-head
with the SST-dev-tuned gauge on the same instances: 0.0245 vs 0.0249 (−1.3 %; larger on 2, smaller on 182, equal 110). So a gauge
tuned on boxes around *random token strings*, without any SST data or labels, recovers ~89 % of the in-distribution gauge's gain
with the same one-sidedness. Preregistered reading applies: the gauge is substantially a property of the weights (and the generic
scale of the input boxes), not of the tuning distribution — the strongest indicator so far that it is derivable rather than
learned. Remaining: the Yelp-text-tuned twin (job 39722892, eval running) and the two-word run.

**15:58 — YELP-TEXT-TUNED gauge on sst_bert_small_6 (job 39722892; 165 boxes on Yelp reviews pushed through the SST tokenizer, Yelp
true labels; `results/deept_small6_oodyelp_eval_short_seed0.json`), same 294 SST test positions:** 0.0220 → 0.0250, **+13.6 % ratio of
means (+11.0 % per-instance); larger on 271, smaller on 0, equal 23**; eps 0.03 verified 41 → **103** (SST-tuned gauge: 95; 48 finite
flips up + NaN 20 → 6, 0 reverse), eps 0.02 181 → 183; lb tighter 274/294 at 0.03 (looser 0), 281 at 0.02, but only 197 (looser 97,
all tiny: mean Δ +0.01) at 0.01. Head-to-head with the SST-dev-tuned gauge: +0.4 % (larger 142, smaller 134, equal 18) — a tie;
vs the random-vocabulary gauge +1.8 %. **Both out-of-distribution tests pass:** text from another dataset gives an identical gauge in
effect, and random vocabulary strings give 89 % of it. The three gauges (SST dev / Yelp text / random tokens) land within 2 % of
each other on 294 held-out instances with 0 reverse flips each, so the learned gauge is a property of the weights plus the box
scale, which is the precondition for deriving it without data. Both OOD runs complete; the two-word run is still evaluating.

**16:20 — rigorous certificate transfer (user: "bound the difference between the two networks"; option 2 = interval weights) and
per-query gauges (route A) — smoke tests passed, full runs launched.** `deept_gauge.py eval --weight_intervals 1` re-verifies with the
folded attention weights declared as auto_LiRPA `BoundedParameter`s spanning their two fp32 neighbours (contains the exact real
product; largest fp32 rounding actually incurred 7.3e-9), so the certificate covers the exact rewrite = the original function.
Two auto_LiRPA obstacles on the way: nn.Linear traces to MatMul(x, Transpose(W)) and the weight-perturbation path cannot push A
through the Transpose → the pass swaps each attention Linear for an `IntervalLinear` (x @ Wt + b, parameter feeds the MatMul directly);
and concretisation takes dim 0 of a perturbed root as the batch → parameters get a leading size-1 dim. Smoke (job 39768955, 7
instances ≤ 6 tokens): gauged-with-intervals vs gauged: radii identical 7/7, lb looser by ≤ 2e-4 (mean −0.0000 / −0.0000 /
−0.0002 at eps 0.01 / 0.02 / 0.03); stock-with-intervals vs stock: identical radii, lb looser ≤ 1e-4. So the rigorous version of the
certificate costs ≈ 1e-4 in margin and nothing in radius at the 1e-4 grid. Full 294-position run: job 39771549 (L40S, 4 weight
sets, `results/deept_small6_wint_eval_short_seed0.json`). Per-query gauges (`eval_pq`: Adam on the gauge per instance from the
learned S6 init, best-of iterates, sound for every iterate): smoke (job 39770485, 4 instances, 3 steps) ran end to end; one
eps-0.03 instance stopped after 1 step on a non-finite gradient near the NaN cliff → gradient entries are now masked as in
`pbv_learn.py`. Full runs: job 39771547 (verified counts at eps 0.02 / 0.03, 20 sentences ≤ 12 tokens, 20 steps) and 39771548
(certified radius: optimise at the fixed gauge's radius, bisect upward with the optimised gauge frozen), ckpt A100, progressive
save + resume. Goal set by the user meanwhile: **find the formula for the gauges** (work interleaved with landing results).

**16:25 — per-query eps run (job 39771547) died of CUDA OOM after 9 instances; patched and resubmitted.** Grad-mode CROWN keeps
every A matrix alive, and on small_6 the retained graph grows ≈ n³ in the token count: the radius run measured 74.5 of 80 GB at
11 tokens, and the first 12-token instance of the eps run asked for ≈ 30 % more. The first nine instances needed no optimisation
(the fixed gauge already verified both eps), so the crash hit the first instance that actually optimised. `cmd_eval_pq` now wraps
the optimisation in `pq_safe`: on `torch.OutOfMemoryError` the instance is recorded with per-query := fixed gauge and an `oom` flag
(stock and fixed values are still measured), the grads are cleared and the cache emptied, and the summary reports how many
instances were skipped and at which lengths. eps run resubmitted as 39772157 (resumes from the 9 saved instances); the radius run
39771548 will hit the same wall at its instance 9, so its resume 39772158 is queued `afterany` it (progressive JSON keeps 0–8).
Consequence to report with the results: per-query optimisation on this model is measured on the ≤ 11-token instances only.
Second failure of the resubmitted eps run (39772157): the guard caught the OOM on the eight 12-token instances, but the failed
grad-mode pass leaves its A matrices and node bounds attached to the BoundedModule, so the next NO-grad call (a 10-token instance)
also ran out of memory. Fix: `pq_safe` now skips the optimisation outright above `--pq_max_tokens` (default 11, the measured
limit) and, if an OOM still happens, calls the module's own `_clear_and_set_new(None)` (the reset compute_bounds runs at start)
plus gc before emptying the cache. Resubmitted as 39772595 (eps, 17 instances saved) and 39772599 (radius, 9 saved; the resume
39772158 still had the old guard and was cancelled). Per-query numbers on this model are therefore for ≤ 11-token instances.
Also noticed: the two-word job 39722894 was pre-empted at ≈ 14:35 and requeued from the top; it re-learned the gauge (bit-identical
to the committed `deept_small6_2w_seed0.pt`, deterministic) and restarted the three-weight-set eval at 15:34.

**16:30 — formula search, step 1 (the gate).** Model of the cost: CROWN's McCormick planes on a product pay ∝ w(x)·w(y), the widths
of the paired coordinates q'_c = (GᵀW_q x)_c and k'_c = (G⁻¹W_k x)_c over the box. With the layer input varying as centre + M z
(z in the unit ℓ∞ ball) the width of a linear functional is 2‖aᵀM‖₁, giving the surrogate Σ_c ‖(GᵀW_q)_c M‖_p ‖(G⁻¹W_k)_c M‖_p
(+ the (W_v, W_o) analogue); it is invariant under diagonal G exactly as CROWN is, p = 2 is minimised in closed form by SVD
balancing (nuclear norm of (W_qM)ᵀ(W_kM)), p = 1 by a 32×32 weight-only optimisation per head. `gauge_formula.py validate`
scores gauges of KNOWN CROWN quality (identity; SST-, Yelp-, random-token-, two-word-trained; the two verifier-trained ones incl.
the cond-28 overfit gauge) under the four surrogates {ℓ1, ℓ2} × {M = I, M = box-shape Jacobian at random-token boxes (exact at
layer 1 because the no_var LayerNorm is linear)}, next to the held-in CROWN metric (mean lb at the stock radius on 24 random-token
boxes and on the learner's own 48 SST-dev boxes), ablates the SST gauge (QK-only, AV-only, one layer at a time), and builds the
four weight-only candidates svd_iso / svd_jac / l1_iso / l1_jac. A surrogate earns the right to be optimised only if it orders
the known gauges like CROWN does. First smoke/full pair (39772195/6) died on argparse: the repo path contains spaces and the chain script word-splits its argument string → gauge paths are now relative and the script exits with python's code so `afterok` means what it says. Smoke 39772615, full 39772616, `results/formula_small6_validate.json`.

**16:50 — formula search: the gate passed on the smoke (job 39772615: 2 random-token boxes for M, 4 SST-dev boxes), and the
closed form already lands in the learned gauge's region.** Surrogate (Jacobian box shape, ℓ1 or ℓ2) relative to identity: learned
SST gauge 0.61, cond-28 overfit gauge 1.53 — the right order; the isotropic surrogates (M = I) give 1.10 vs 1.11 and cannot tell
the two apart, so the box shape is what makes the width model work. Ablations of the learned gauge on its own SST boxes (mean lb;
identity +0.081, full gauge +0.841): AV side alone +0.794, QK side alone +0.344; per layer: layer 0 +0.171, 1 +0.077, 2 +0.074,
3 +0.250, 4 +0.455, 5 +0.408 — the gain lives on the value/output side and in the deeper layers, and the surrogate's AV term
tracks it (0.78 → 0.45 of the identity total). Candidates, no CROWN training: **svd_jac** (per head, SVD balancing of
(W_q M_l, W_k M_l) and (W_v M_l, W_oᵀ) with M_l = stacked Jacobians of the layer input w.r.t. the perturbed row at 2 random-token
centres, scaled by their stock radii) scores **+0.954** on the SST boxes and +0.032 on the random boxes — above the learned gauge
(+0.841 / +0.025) on both, and those SST boxes are held-IN for the learned gauge and held-OUT for the candidate. svd_iso (M = I,
the ViT-R45 construction) +0.578; l1_iso +0.239; l1_jac = svd_jac bit for bit (Adam found nothing below the SVD init in 20 steps).
Caveats: 4 + 2 boxes; max cond 29.5 (QK layer 4 head 2; AV cond grows 2 → 12 with depth) — watch the fp64 gate. The candidate is
NOT the learned matrix: diag-ness of learned⁻¹·candidate is 0.03 (random-like) on every head, so the formula reaches a flat
optimum rather than reconstructing the learned gauge — which is also why the uniqueness comparison of learned gauges was inconclusive.
Model limitation seen in the per-layer view: the learned layer-0 gauge raises the layer-0 AV surrogate (1.25×) while CROWN says
layer 0 helps — M_l is treated as gauge-independent, but a tighter early layer shrinks the downstream widths. Launched: paired
294-position eval of the 2-box candidates vs the learned gauge (job 39773109, L40S: stock / svd_jac_2box / learned S6 / svd_iso_2box,
`results/deept_formula_small6_2box_eval_short_seed0.json`); full small_6 validate (39772616: 24 + 48 boxes, adds the CROWN-width
surrogate that reads the q'/k'/v' widths auto_LiRPA actually derives); replication validates on big_3 (39773174) and Yelp small_3
(39773175), which also carry two verifier-free variants (eps := 1 on every box, and a second seed's random centres) to test
whether the CROWN radii in M matter at all.

**17:00 — ops: third per-query failure, and the two-word chain job replaced by a resumable eval.** The eps run (39772595) got past the
capped 12-token instances and then ran out of memory on an 11-token one at the edge (79.2 GB) — raised inside auto_LiRPA's
TorchScript'd ops as a plain `RuntimeError`, which the `OutOfMemoryError` guard did not catch. The guard now catches both, empties
the CUDA cache before each optimisation, and keeps the 11-token cap; resubmitted as 39773887 (36 instances saved). The radius run
39772599 is healthy at 10 tokens (per instance: fixed gauge +8.5 % radius, per-query a further +3.6 % on top). The two-word chain job
39722894 was pre-empted a second time and restarted from its learn step again, so it was cancelled; `cmd_eval` now resumes weight
sets already stored in its JSON (same instance list), and `deept_2w_eval_resume.sh` (job 39773950) runs only the eval with the eps
grid stored in the JSON (0.0047 / 0.0094 / 0.0141 = 0.5 / 1 / 1.5 × the median stock two-word tuning radius): stock is resumed from
the first attempt, the two-word-trained and one-word-trained gauges remain (~3.6 h).

**17:20 — replication validates (v1) on big_3 (job 39773174) and Yelp small_3 (39773175): the candidate transfers, the surrogate
does not fully explain the learned gauges, and the value/output side is where the model is incomplete.** Held-in mean lb
(random boxes / learner's own boxes); identity → learned → svd_jac → svd_iso:
big_3: +0.111 / +0.146 → +0.473 / +1.186 → +0.322 / +0.805 → +0.262 / +0.609 — the closed form keeps 58 % / 63 % of the learned
gain (svd_iso 42 % / 45 %), l1_jac 64 % / 66 %; the eps ratios and the probe seed do not matter (svd_jac_u, _u2 within 1 %).
Yelp small_3: +0.076 / +0.246 → +0.275 / +1.785 → +0.144 / +1.373 → +0.065 / +1.157 — 34 % / 73 %; but here the second-seed
candidate svd_jac_u2 gets +0.224 / +0.400 (better on random boxes, far worse on the learner's) with the SAME surrogate value as
svd_jac (0.933), and the ℓ1-refined l1_jac (surrogate 0.912 < 0.933) is worse than svd_jac on CROWN (+1.16 vs +1.37). Two
facts break the pure width-product model: (i) on Yelp small_3 the learned gauge scores 0.998 (ℓ1) / 1.094 (ℓ2) on the Jacobian
surrogate and 0.94–0.99 on the CROWN-width surrogate — i.e. ≈ identity — while gaining +1.54 on its boxes; (ii) on all three models
the learned gauge RAISES the layer-0 value-side width product (small_6 1.25×, big_3 1.04×, Yelp 1.14× — also in the widths CROWN
actually derives), yet the layer-0 gauge alone is worth a large share of the gain, and the AV-only ablation is the bigger half on
every model. So the value/output gauge is not (only) shrinking widths. Hypothesis: the relaxation slack of coordinate c of the
context reaches the final bound through W_o's column c and the downstream backward functional λ, i.e. with weight |λᵀ W_o'_{:,c}|,
not ‖W_o'_{:,c}‖₁ — and G_a can ROTATE the value coordinates so that the wide ones are the ones downstream barely reads. Same
balancing form with B = W_oᵀ N instead of W_oᵀ, N = the downstream functionals: added to `gauge_formula.py` as N_out (margin
gradients w.r.t. the attention output at the random centres), N_ffn ((W₁Γ₁P)ᵀ, the same layer's ReLU inputs — weight-only, the
no_var LayerNorm being linear), N_next (next layer's projections through the residual, weight-only) and N_all (the three with equal
Frobenius weight); surrogate column l1_jacN, CROWN-width column "with N", candidates svd_jacN_{out,ffn,next,all}. v2 runs:
Yelp small_3 39776031, big_3 39776034 (v1 outputs kept as *_v1). Also from the full small_6 table so far: the SST-, Yelp-,
random-token- and two-word-tuned gauges and the verifier-trained "inner" gauge all sit at surrogate 0.72–0.73 / CROWN-width
0.53–0.57 with held-in +0.43–0.47 / +1.08–1.18 (identity +0.09 / +0.16) — one flat optimum reached from five different tuning sets.

**17:30 — per-query eps run, fourth failure, root cause found.** The resumed run (39773887) hit OOM on an 8-token instance: a grad-mode
CROWN pass leaves its node bounds and A matrices — hence the whole retained graph — on the BoundedModule of that sentence length,
and with five lengths in play the leftovers of up to five graphs sit on the card at once; once the first OOM happened the caught
exception did not give the memory back either, so every later instance (len 8, 10) was "skipped" too. Fix: `pq_safe` now calls the
module's `_clear_and_set_new(None)` + gc after EVERY optimisation, and after a caught OOM the instance is saved as fixed-gauge and
the process exits with code 3; the new `deept_pq_chain2.sh` restarts it (fresh CUDA context, resume from the JSON) up to 12 times.
The 17 instances that were skipped only because of the leak (25–27 and 57–70, lengths 5–10) were removed from the JSON so they are recomputed;
the 12-token ones stay capped. Resubmitted as 39776320. The radius run (39772599, old code) is still healthy at instance ~25.

**17:45 — v2 Yelp small_3 (job 39776031): the output-side functionals explain the value gauge, and the closed form with them reaches
82–86 % of the learned gain on the learner's boxes.** With N in the cost, the learned gauge now REDUCES the width-product cost
CROWN actually incurs (with-N column 0.87 random / 0.83 learner boxes; per layer AV·N 0.93 / 0.91 / 0.89, where the plain AV column
went 1.14 / 1.00 / 1.08) — the value gauge works by rotating the wide value coordinates away from the directions downstream reads,
not by shrinking widths. Candidates (identity +0.076 / +0.246, learned +0.275 / +1.785): svd_jacN_out **+0.202 / +1.566** (63 % /
86 % of the learned gain; margin-gradient functionals at the 24 random centres), svd_jacN_all +0.192 / +1.510 (58 % / 82 %; cond 6.5),
svd_jacN_ffn +0.166 / +1.431, svd_jacN_next +0.163 / +1.371, vs svd_jac +0.144 / +1.373 (34 % / 73 %). svd_jacN_out came out with
max cond 9e17 (a near-zero singular value of (W_v M)ᵀ(W_oᵀ N_out) makes G singular): the diagonal freedom Λ in G = R_a⁻¹ U Λ is
CROWN-neutral, so `svd_balance` now floors the balancing scale at sqrt(1e-3 · s_max) — unchanged where conditioning was fine.
The surrogate columns themselves are weak discriminators on this model (l1_jacN 0.95–0.98 for everything); the CROWN-width-with-N
column and, above all, the held-in CROWN metric are what ranks candidates. Paired eval on the 277-instance Yelp small_3 protocol
(stock / svd_jacN_all / svd_jac / learned; eps grid 0.00801 / 0.016 / 0.024 as in the learned gauge's eval): job 39778637.

**18:00 — full small_6 validate v1 (job 39772616: 24 random-token boxes for M, 48 SST-dev learner boxes).** Held-in mean lb
(random / learner boxes): identity +0.092 / +0.156; the five learned gauges (SST, Yelp, random-token, two-word, verifier-"inner")
+0.43–0.47 / +1.08–1.18 with surrogate 0.72–0.73 and CROWN-width 0.53–0.57 — one flat optimum; overfit gauge −1.24 / −2.77 with
surrogate 1.07 (ℓ1) / 1.32 (ℓ2) and CROWN-width 2.4 / 3.6. Ablations of the SST gauge: AV-only +0.41 / +1.04, QK-only +0.32 / +0.73;
layers 0…5 alone: +0.11 / +0.22, +0.10 / +0.17, +0.16 / +0.34, +0.23 / +0.56, +0.28 / +0.68, +0.23 / +0.45. Candidates:
**svd_jac +0.426 / +1.087 = 88 % / 91 % of the learned gain** (svd_iso +0.272 / +0.716 = 48 % / 55 %); l1_jac (400 Adam steps from
the SVD init, surrogate 0.697 < svd_jac's 0.703) +0.325 / +0.814 — a lower surrogate but a worse bound, so the ℓ1 refinement is
dropped: the closed form is the formula. Per layer the learned gauge cuts the QK width product most at layer 4 (0.46 / CROWN 0.35)
and the AV product at layers 2–5 (0.5–0.6); layer 0 AV rises (1.25) as on the other models. Max cond of svd_jac 34.6 (pre-floor).
Outputs archived as *_v1; v2 validate with the output-side functionals launched (39779153).

**18:20 — big_3 v2 validate (job 39780094, floored balancing): the output-side functionals lift the closed form from 58–63 % to
75–77 % of the learned gain on the held-in metric.** Identity +0.111 / +0.146, learned +0.473 / +1.186; svd_jacN_all **+0.384 /
+0.946**, svd_jacN_out +0.382 / +0.934, svd_jacN_ffn +0.363 / +0.899, svd_jacN_next +0.329 / +0.820, svd_jac +0.322 / +0.805,
svd_iso +0.262 / +0.609, l1_jac +0.326 / +0.827. As on Yelp small_3, the learned gauge lowers the with-N cost at layer 0 (AV·N
0.76) where the plain width product rises (1.04). Paired eval on the 288-instance big_3 protocol (stock / svd_jacN_all / svd_jac /
learned; eps 0.00957 / 0.0191 / 0.0287): job 39782433.

**18:35 — per-query eps run, first complete pass (job 39776320, four restarts): on the instances it could optimise, per-query
beats the fixed gauge every time but rarely flips a verdict.** 153 instances (20 test sentences ≤ 12 tokens). eps 0.02: stock /
fixed / per-query verified 125 / 125 / 125 — only 3 instances were unverified under the fixed gauge and short enough to optimise;
all 3 improved (mean +0.36) without flipping. eps 0.03: verified 28 / 68 / 69; 22 instances optimised (lengths 5–8), all 22
improved (fixed mean lb −2.13 → −1.39, mean gain +0.73, max +1.18), one flipped to verified. 63 instances were skipped: the 12-token
ones by design, and lengths 9–11 after genuine OOMs in eps mode (a length-10 and a length-11 instance died at eps 0.03 with the
cache emptied and bounds cleared — memory at eps 0.03 is higher than at the radius run's r_f ≈ 0.03 for the same lengths, still
unexplained), plus a cascade: three 5-token instances sit on the NaN cliff at eps 0.03 (stock and fixed lb NaN, no finite
iterate), the record-keeping mistook "no finite iterate" for an OOM, and the adaptive cap fell to 4 tokens. Fixed: NaN-cliff
outcomes are recorded as fixed with step −2 (not oom), each instance line now logs peak GPU memory, and the 11 cap-cascade
records (lengths 5, 9, 10) are being recomputed with a 10-token cap (job 39782573). The radius run (39772599) is at instance 67
of 153 with no failures.

**18:55 — per-query memory: measured, not guessed.** With peak-memory logging: a 5-token optimisation peaks at 8.4 GiB, a 10-token one
at 61.7 GiB (n³ scaling; 11 tokens ≈ 82 GiB > the A100, so 10 is the real cap in eps mode), and after ONE 10-token optimisation a
9-token one (≈ 45 GiB) no longer fits — i.e. ≈ 35 GiB stay allocated after a pass even with the cache emptied and the module's known
bound attributes cleared. `pq_safe` now also purges every graph-attached tensor (anything with a grad_fn) from the BoundedModule and its
nodes after each optimisation, and `cmd_eval_pq` exits 3 for a clean restart after any optimisation on an instance of ≥ 10 tokens
(`--pq_restart_len`); `deept_pq_chain3.sh` allows 80 restarts. The 7 nine-token records that had been cap-skipped were dropped for
recomputation; resubmitted as 39784132 (146 of 153 instances already final).

**19:00 — per-query eps run complete (job 39784132; the purge works: seven consecutive 9-token optimisations at a flat 40.6 GiB
peak, no restart needed).** 153 test instances (20 SST test sentences ≤ 12 tokens), 20 Adam steps from the learned S6 gauge, early
stop once verified. eps 0.02: verified stock 125 / fixed gauge 125 / per-query 125 (3 instances optimised, all improved by ≈ +0.4,
none flipped). eps 0.03: verified 28 / 68 / 69; 30 instances optimised (lengths 8–11): all 30 improved, fixed mean lb −2.41 →
per-query −1.74 (mean gain +0.66, max +1.18), one flipped to verified; 3 five-token instances sit on the NaN cliff (no finite
bound for stock, fixed or any iterate); 52 instances of 11–12 tokens were not optimised (grad-mode memory). Reading: per-query
optimisation is a reliable but small refinement on top of the fixed gauge (+0.66 margin vs the fixed gauge's own +2.0 over stock
at eps 0.03), and it rarely crosses zero — the fixed gauge already takes the instances that are close. `results/deept_small6_pq_eps_seed0.json`.

**19:10 — small_6 v2 validate (job 39779153).** The output-side functionals change little on this model: svd_jacN_all +0.430 / +1.097
(90 % / 92 % of the learned gain), svd_jacN_next +0.430 / +1.096, svd_jacN_out +0.428 / +1.091, svd_jacN_ffn +0.426 / +1.087, svd_jac
+0.426 / +1.087; the verifier-free variants are identical to svd_jac (uniform eps +0.425 / +1.085, second probe seed +0.425 / +1.085):
on small_6 the construction needs only the weights and a few random probe sequences. l1_jac again worse (+0.309 / +0.777). The
learned gauge's layer-0 value effect is alignment here too (CROWN-width AV·N 0.84 at layer 0 where the plain AV product is 1.25);
the QK product falls most at layers 4–5 (0.35 / 0.41). Paired 294-position eval of the 24-box candidates launched (job 39787014:
stock / svd_jacN_all / svd_jac / learned), alongside the running 2-box eval (39773109).

**18:50 — Yelp small_3 v3 (job 39787037): the CROWN radii are not needed, but the probe set still matters on this model.**
svd_jacN_all_u (eps := 1 everywhere) +0.190 / +1.517 vs svd_jacN_all +0.192 / +1.510 — verifier-free confirmed. But
svd_jacN_all_u2 (second probe seed for M and the margin gradients) +0.261 / +0.625: better than seed 0 on the random boxes
(near the learned gauge's +0.275) and far worse on the learner's Yelp-dev boxes, the same pattern the plain svd_jac showed
(+0.224 / +0.400). So N does not remove the probe dependence here; the gauge a probe set produces fits that probe set's own
box shapes, and 24 random-token boxes are too few to average over on this model (on small_6 and big_3 the two seeds agree
within 1 %). First paired number on the real protocol (job 39778637, first gauge set): svd_jacN_all +6.3 % certified radius
(0.0223 → 0.0237), eps-0.024 verified 116 → 145 of 277; the learned gauge gave +9.5 % on the same instances (≈ 2/3 of it).
Test of the remedy: the same validate with 96 probe boxes per seed (job 39789896, `gauge_formula_chain.sh yelp3_big`).

**18:58 — big_3 v3 (job 39787038): verifier-free and probe-insensitive.** svd_jacN_all +0.384 / +0.946; with eps := 1 everywhere
+0.384 / +0.947; with the second probe seed (M and margin gradients) +0.382 / +0.941. Together with small_6 (all three within
0.5 %), the construction on the two SST models needs the weights and any two dozen random-token probes; Yelp small_3 is the
one model where the probe set matters (96-box test running).

**19:20 — first paired eval of the formula gauges: Yelp small_3, 277 instances (job 39778637).** Certified radius (ratio of means):
stock 0.02226 → svd_jacN_all 0.02370 (**+6.5 %**; larger on 234, smaller on 25, equal 18), svd_jac 0.02335 (+4.9 %; 196 / 62 / 19),
learned 0.02450 (+10.1 %; 263 / 1 / 13). So the formula gauge takes 64 % of the learned radius gain here (svd_jac 45 %), below the
80 % bar. Fixed-eps verified 255 / 209 / 116 (stock) → 256 / 219 / 145 (formula) vs 256 / 219 / 152 (learned); flips
unverified→verified 1 / 10 / 29 (formula) vs 1 / 10 / 36 (learned), verified→unverified 0 for both. Head to head, learned − formula:
radius larger on 225, smaller on 18; eps-0.024 lb tighter on 267/277 (+0.75). fp64 gates 2–4e-15. Consistent with the held-in
screen (82 % on the learner's boxes, 58 % on random boxes → 64 % on test). User's goal updated: **improve the manual procedure to
beat the learned gauges.**

**19:42 — Yelp small_3 with 4× the probes (96 random-token boxes, job 39789896): more probes help a little on random boxes and
not at all against the seed dependence.** svd_jacN_all +0.327 / +1.521 (random / learner boxes; learned +0.464 / +1.785, identity
+0.076 / +0.246): 65 % / 83 % of the learned gain vs 58 % / 82 % with 24 probes. The second probe seed, also with 96 probes, still
gives +0.354 / +0.625 — better on random boxes, collapsed on Yelp text. The layer-0 gauges of the two seeds are identical (M_0 is
sentence-independent under the linear LayerNorm); the layer-1 and layer-2 gauges are unrelated matrices (relative difference
1.2–1.4). So the deeper Jacobian shapes of random-token sequences do not average into anything Yelp-like, and seed 0 was lucky.
Advisor's plan for the new goal, in order: (1) localise the gap with side/layer swaps between the learned and the closed-form
gauge, (2) screen on a held-out certified-radius metric (24 dev sentences the learner never saw; the learner-box column mispredicted
Yelp), (3) diagnostic ceiling: warm-start the CROWN learner from the closed-form gauge for 40 low-lr steps and paired-eval it — if
it beats the 120-step learned gauge, headroom exists in the formula's basin; (4) then CROWN-consistent M (coupling), sensitivity-
weighted token blocks, text probes. Implemented (1)–(3) plus unlabeled dev-text probes as a candidate (`--dev_probes`,
`cand:svd_jacN_all_dev`, `_u+dev`); `gauge_formula.py` got `--hybrids --cross --radius_names --n_dev --dev_probes --tag`, the learner
got `--gauge` as a warm start, new `deept_init_chain.sh`. Jobs (all ckpt, 7 h limits because a 24 h cluster maintenance starts
2026-09-08 04:00): smoke 39796636, hybrid validates Yelp 39796637 / small_6 39796639 / big_3 39796640, warm-start Yelp learn+eval
39796638. The 24-probe Yelp candidate gauges used by the paired eval are kept as `*_24box.pt` (the 96-probe run overwrote the
originals).

**20:01 — warm-start learner on Yelp small_3 (job 39796638, first half): 40 Adam steps at lr 0.005 from the closed-form gauge
(24-probe `svd_jacN_all`) on the learner's own boxes.** Held-in mean lb +1.510 → +1.714 (step 39, still rising; fraction verified
0.958 → 1.000, cond 6.1 → 4.5, fp64 gate 6e-15), vs +1.785 for the 120-step learner from identity (its best step was 100). So 40
cheap steps recover 89 % of the gap between the closed form and the learned gauge held-in; the paired eval against the learned
gauge runs now (`results/deept_formula_yelp3_init_eval_short_seed0.json`). This is the hybrid reference, not the manual procedure.

**20:05 — localisation on big_3 (job 39796640, held-in): the closed form falls short at LAYER 0, and the reason is the norm.**
Side/layer swaps between the learned gauge and `svd_jacN_all` (share of the learned gain, random / learner boxes; candidate
alone 0.75 / 0.77): learned QK + candidate AV 0.92 / 0.94, candidate QK + learned AV 0.86 / 0.86; candidate with the learned
layer 0 **0.95 / 0.93**, with the learned layer 1 0.76 / 0.78 (no change), layer 2 0.81 / 0.84; learned with the candidate's
layer 0 0.82 / 0.86, layer 1 0.99 / 0.99, layer 2 0.96 / 0.95. So the candidate's layers 1–2 are as good as the learned ones and
the gap is layer 0 — the one layer where M_0 is exact. CROWN-measured width products at layer 0 (learner boxes, relative to
identity): learned QK 0.87, candidate 0.97 (barely moves), ℓ1-refined `l1_jac` 0.79; N-weighted AV: learned 0.76, candidate
0.82, l1_jac 0.86. Same pattern on small_6 (QK 0.89 / 0.98 / 0.78) and Yelp (0.91 / 1.00 / 0.87). Reading: at layer 0 there is
a single token block, so the box width of a functional is its ℓ1 row norm, and the ℓ2 closed form (nuclear norm) is the wrong
norm there; with many token blocks at deeper layers the ℓ2 proxy is close. `l1_jac` lost overall because its AV side ignores N.
Round 3 (jobs 39801668–71, smoke + three models): ℓ1 surrogate WITH N minimised by Adam from the closed form (`cand:l1N`), its
layer-0 splice into the closed form (`l1N@L0`), QK-only splices, an inflated-M variant, and `l1jac_qk+jacN_av`; held-out radius
screen on SST *test* sentences for the SST models (SST dev has no short sentences beyond the learner's 40; there is no train.txt). Round 2 (sensitivity
weights, inflation; jobs 39799465–67) still running; big_3's inflation: CROWN widths 1.33× / 2.22× the Jacobian shape at layers
1 / 2, fixed point in one round.

**20:18 — round 2 on big_3 (job 39799466): sensitivity-weighted token blocks and CROWN-inflation rescaling of M do nothing.**
Held-in (random / learner boxes): `svd_jacN_all` +0.384 / +0.946; sensitivity-weighted (query/key blocks scaled by the rank-1
factors of |∂margin/∂score|, value blocks by the first-order softmax widths) +0.385 / +0.951 with cond 46 (was 10.6); QK-only /
AV-only / sqrt weights the same; inflation rescaling (CROWN width / Jacobian width per token, 1.34× at layer 1, 2.23× at layer 2,
fixed point in one round) +0.384 / +0.945 for 1, 2 or 3 rounds; both together +0.385 / +0.952. Consistent with the localisation:
the deeper layers were already matching the learned gauge, and neither idea touches layer 0 (a single token block, ρ ≡ 1). Both
directions dropped for big_3; round 3 (ℓ1 with N at layer 0) is the live hypothesis.

**20:38 — round 3 on big_3 (job 39802258, held-in): the ℓ1 surrogate WITH N closes most of the gap, and it is still a manual
procedure.** Random / learner boxes, share of the learned gain: closed form `svd_jacN_all` 0.75 / 0.77 → **`l1N` 0.96 / 0.95**
(+0.458 / +1.130 vs learned +0.473 / +1.186; cond 9.8): the ℓ1 (box-width) version of the same cost model, with the downstream
functionals N on the value side, minimised by Adam for 400 steps from the closed form — weights and random-token probes only, no
verifier. Splices: only layer 0 taken from `l1N` 0.94 / 0.93 (the layer-0 reading was right: that is where the norm matters);
only the QK side 0.84 / 0.86; only layer-0 QK 0.83 / 0.84 — the layer-0 AV·N side is the other half. `l1jac_qk+jacN_av` (ℓ1 QK
without N + closed-form AV) 0.84 / 0.86; inflated-M ℓ1N 0.95 / 0.95 (inflation irrelevant again). Held-out radius screen on 48
SST-test boxes running for these rows; the paired eval of `l1N` follows as soon as the gauge file is saved.

**20:42 — two jobs from the earlier queue landed.** (1) Two-word perturbation eval on small_6 (job 39773950, 269 two-word boxes,
7 position pairs per sentence, eps 0.0047 / 0.0094 / 0.0141): the two-word-tuned gauge +13.7 % radius (243 larger / 0 smaller /
26 equal) and the ordinary one-word-tuned gauge +13.8 % (same 243 / 0 / 26); eps-0.0141 verified 75 → 107 vs 110; 0 reverse
flips at every eps; fp64 gates 1e-15. The one-word gauge transfers unchanged to the two-word specification, and two-word tuning
adds nothing — the gauge is a property of the weights, not of the perturbation set. (2) Per-query gauge, radius mode (job
39772599, old chain, 153 test instances of 20 sentences ≤ 12 tokens, 20 Adam steps per instance from the fixed gauge): radius
stock 0.0249 / fixed gauge 0.0287 (+15.4 %) / per-query 0.0296 (+18.9 % vs stock, +3.1 % vs fixed); per-query larger than fixed
on 99, equal on 54 (the 44 twelve-token instances could not be optimised on 80 GB and count as fixed), never smaller; at eps
0.02 verified 125 / 125 / 125. So per-instance re-optimisation buys ≈ 4 % on the instances it can touch, at ~90 s per instance —
the fixed gauge carries almost all of the value.

**20:48 — round 2 on Yelp small_3 (job 39799465, held-in): also null.** Sensitivity weights +0.318 / +1.512 (cond 83) vs the
closed form +0.327 / +1.521; inflation rescaling (1.14× / 1.67× at layers 1 / 2) +0.326 / +1.519 for 1–3 rounds. Token weighting
and CROWN-consistent M are dropped on all models where they were tried (big_3, Yelp); the held-out radius screens of these rows
are still running but cannot rescue a held-in tie with a 10× worse condition number.

**20:57 — round 3 on Yelp small_3 (job 39801670, held-in): the ℓ1-with-N refinement replicates.** Random / learner boxes, share
of the learned gain: closed form 0.64 / 0.83 → **`l1N` 0.85 / 0.94** (+0.407 / +1.687 vs learned +0.465 / +1.785, cond 9.6);
layer-0 splice only 0.73 / 0.88 (on Yelp the gap is spread over the layers, as the swaps said); QK side only 0.70 / 0.87;
inflated-M variant 0.87 / 0.92. Held-out radius screen on 46 Yelp-dev boxes running for these rows. Held-out screen facts so
far (Yelp hybrid run): the closed form takes 0.66 of the learned radius gain there (paired eval: 0.64 — the screen predicts the
paired protocol); the seed-1 candidate that "collapsed" on the learner's boxes takes 0.51 with 0 smaller radii, so the learner-box
column exaggerated the seed dependence; unlabeled Yelp-text probes give 0.52 with 19 larger / 18 smaller (two-sided — worse than
random tokens), random + text probes 0.68.

**21:05 — round 2 on small_6 (job 39799467, held-in): null, three for three.** All sensitivity / inflation variants +0.428–0.431 /
+1.090–1.098 vs the closed form +0.430 / +1.097 (learned +0.468 / +1.184). Round 2 closed on every model.

**21:07 — localisation on small_6 (job 39796639, held-in): the whole gap is the QK side, spread over the layers.** Candidate
`svd_jacN_all` 0.90 / 0.92 of the learned gain; learned QK + candidate AV **1.01 / 1.00** (the candidate's value gauges are as
good as the learned ones on this model), candidate QK + learned AV 0.88 / 0.91 (no gain). Per-layer swaps move the candidate by
0.00–0.05 each (largest at layer 5: 0.95 / 0.96; layers 3–4: 0.92 / 0.94), and the learned gauge loses ≤ 0.02 from any single
candidate layer except layer 5 (0.95 / 0.96) — no single layer carries it. Fits the ℓ1-vs-ℓ2 reading on the QK side (the
`l1N` result for small_6, job 39802259, is next). No held-out screen in this run (SST dev has no held-out short sentences).

**21:08 — paired eval of the closed-form gauges on big_3 (job 39782433, 288 instances, 3 h 10).** Certified radius (ratio of
means): stock 0.01662 → `svd_jacN_all` 0.01782 (**+7.2 %**; larger on 273, smaller on 0, equal 15), `svd_jac` 0.01757 (+5.7 %;
268 / 0 / 20), learned 0.01859 (+11.9 %; 274 / 0 / 14) — the closed form takes 61 % of the learned radius gain (svd_jac 48 %),
one-sided here (on Yelp it had 25 smaller). Fixed-eps verified 256 / 116 / 4 (stock) → 257 / 124 / 32 (closed form) vs
258 / 129 / 48 (learned); flips unverified → verified 1 / 8 / 28 vs 2 / 13 / 44; verified → unverified 0 for both; fp64 gates
1–2e-15. The held-out SST-test screen predicted 0.56 for this gauge (paired: 0.61) and gives 0.84 for `l1N`, so its paired eval
is the next number to get.

**21:24 — round 3 on small_6 (job 39802259, held-in): the ℓ1 refinement helps on the QK side and HURTS on the value side here.**
Closed form 0.90 / 0.92 → `l1N` (both sides) **0.83 / 0.84**, but `l1N_qk` (ℓ1 QK at every layer, closed-form N-weighted AV)
**0.96 / 0.96** (cond 19); layer-0-only splices 0.90–0.91 / 0.92–0.93 (on this model the QK gain is spread over the layers, as
the swaps said). Across models: the QK-side ℓ1 refinement helps everywhere (big_3 +0.10, Yelp +0.06, small_6 +0.06 held-in
random); the AV-side ℓ1 refinement helps at layer 0 on big_3 (+0.10, all of its AV gain) and at all layers on Yelp (+0.15), and
costs 0.13 on small_6 at the deeper layers. A verifier-free rule that never loses: ℓ1-refine QK at every layer and AV at layer 0
only (`l1N_qk+av0`: predicted 0.95 big_3, 0.97 small_6, 0.77 / 0.91 Yelp held-in) — built from the saved parts for big_3 and
submitted for a paired eval (job 39817354, with `l1N_qk`; the learned set comes from the finished job 39782433, same instances).
Yelp keeps the full `l1N`. Also finished: the Yelp localisation held-out screen (job 39796637) — closed form 0.66; learned QK +
candidate AV 0.88; candidate QK + learned AV 0.77; learned layer 0 / 1 / 2 in the candidate 0.76 / 0.80 / 0.75; the seed-1
layer 2 in the seed-0 gauge 0.56 with 0 smaller (a 0.10 loss, not the collapse the learner-box column showed).

**21:26 — warm-start ceiling test, 40 steps (job 39796638, paired, 277 Yelp instances):** +9.0 % radius (264 larger / 3 smaller /
10 equal) vs +10.1 % for the learned-from-identity gauge (263 / 1 / 13); head to head the learned gauge is larger on 196, smaller
on 15. Forty low-lr CROWN steps from the closed form do not reach the 120-step learner; the 100-step run (job 39802260) decides
whether the learned gauge is the ceiling of this objective.

**21:25 — round 4 (1500 annealed steps) converged: Yelp `l1N` 0.86 / 0.94 held-in, 0.88 on the held-out screen — identical to the
400-step run; big_3 0.86 vs 0.84.** The surrogate's optimum is reached; nothing more comes from the optimiser. Paired evals of the
refined gauges are running: big_3 `l1N` + layer-0 splice + learned (job 39816017), big_3 unified rule + `l1N_qk` (39817354), Yelp
`l1N` + layer-0 splice + learned (39818135); small_6 follows its round-3 save. Round 5 (jobs just submitted): held-out screens of
the unified rule on all three models, and a second probe seed for the whole refined procedure on Yelp.

**21:58 — ceiling test answered: the learned gauge is the optimum of its own objective (job 39802260, Yelp small_3, paired).**
The CROWN learner warm-started from the closed form, 100 steps at lr 0.005, reaches held-in +1.781 (learned from identity, 120
steps: +1.785) and on the paired protocol +10.0 % radius (264 larger / 2 smaller) vs +10.1 % (263 / 1); head to head 94 larger,
94 smaller, 89 equal; eps-0.024 verified 153 vs 152; fp64 gate 2e-15. Two unrelated starting points converge to the same
performance, so no gauge in this family beats the learned one *on this objective* (mean CROWN lb on the learner's 120 boxes at
their stock radii) by more than noise. "Better than the learned gauges" therefore cannot come from a better formula for the same
target; it needs a target the learner is not optimising (more or different boxes, larger radii, worst-case), or it is a tie at
best. The manual procedure's job is to reach parity: currently 0.84–0.88 of the learned gain on the held-out screens (paired
evals of `l1N` running: big_3 39816017, Yelp 39818135; small_6 to follow), up from 0.56–0.66 for the closed form.

**22:28 — held-out screens, rounds 3–5 (48 SST-test boxes for the SST models, 46–47 Yelp-dev boxes), share of the learned
radius gain, larger / smaller vs stock.** small_6 (learned +14.1 %): closed form **1.06** (41 / 0; head to head 18 / 16), its
layer-0 ℓ1N splice **1.08** (41 / 0; 18 / 15), second probe seed 1.06, inflation 1.03, both-sided `l1N` **0.54** (36 / 0; 0 / 42 —
the deeper-layer value refinement is harmful here on held-out text, worse than held-in suggested). big_3 (learned +12.7 %): closed
form 0.56, `l1N` 0.83–0.84, unified rule `l1N_qk+av0` 0.83 (46 / 0), QK-only 0.71. Yelp seed 0 (learned +10.7 %): closed form
0.66, `l1N` 0.88, unified rule 0.82, QK-only 0.76; Yelp seed 1 (own screen, learned +12.8 %): closed form 0.28, `l1N` 0.59 (46 / 0),
unified rule 0.56 — the probe draw still matters on Yelp after the refinement. So the value-side ℓ1 refinement is model-dependent
(helps big_3 and Yelp, hurts small_6 beyond layer 0) and the QK-side one helps everywhere; on small_6 the closed form is already at
or above the learned gauge on held-out text (paired eval of it: job 39787014, last set running). Paired eval of the unified rule
and the QK-only refinement on small_6 submitted (job 39825991 at 22:27, 3 sets; learned set from the earlier paired eval).

**23:15 — paired eval of the refined gauges on Yelp small_3 (job 39818135, 277 instances).** Certified radius (ratio of means):
stock 0.02226 → `l1N` 0.02415 (**+8.5 %**; larger on 255, smaller on 10, equal 12), layer-0 splice `l1N@L0` 0.02394 (+7.6 %;
247 / 15 / 15), learned 0.02450 (+10.1 %; 263 / 1 / 13). So the refined manual gauge takes **84 %** of the learned radius gain on
the headline protocol (closed form: 64 %, with 25 smaller radii; now 10); the held-out screen said 0.88. Head to head the learned
gauge is larger on 174 instances, `l1N` on 62. Fixed-eps verified 255 / 209 / 116 (stock) → 256 / 219 / 147 (`l1N`) vs
256 / 219 / 152 (learned); flips unverified → verified 1 / 10 / 31 vs 1 / 10 / 36, verified → unverified 0; fp64 gates 3e-15.
Not parity, but the largest single step of the evening on this model.

**23:16 — small_6 paired evals are slow (≈ 2.3 h per weight set at 294 instances), so the closed-form run (job 39787014, 2 of
4 sets done) and the 2-box run (39773109, 3 of 4) will not reach their learned set before the 04:00 maintenance; the learned
set exists in the earlier small_6 paired eval on the same instances (`results/deept_small6_eval_short_seed0.json`) and is joined
by `paired_summary.py <json> <tag> <ref.json> <ref_tag>`. The refined-gauge run (39825991, stock set 47 min in) was cancelled and
resubmitted as job 39828671 with a stock-only resume file built from the finished stock set (same instances), so its two gauges
(unified rule, QK-only ℓ1N) get ≈ 4.6 h instead of 2.3 h. small_6 held-out screen, round 3 complete: closed form 1.06, layer-0
splice 1.08, QK-only 1.06 (42 / 0; 17 / 12 head to head), `l1jac_qk+jacN_av` 1.07, both-sided `l1N` and its inflated variant 0.52.

**23:45 — paired eval of the unified rule on big_3 (job 39817354, 288 instances; learned set joined from job 39782433, same
instances).** Certified radius: stock 0.01662 → unified rule `l1N_qk+av0` 0.01842 (**+10.8 %**; larger on 276, smaller on 0,
equal 12), QK-only `l1N_qk` 0.01817 (+9.4 %; 275 / 0 / 13), learned 0.01859 (+11.9 %; 274 / 0 / 14). The unified rule takes
**91 %** of the learned radius gain (closed form: 61 %) and is one-sided; head to head against the learned gauge it is larger on
100 instances and smaller on 97 (equal 91) — a per-instance tie, with the learned gauge ahead in the mean through the largest
eps: verified 256 / 116 / 4 (stock) → 258 / 125 / 34 (unified) vs 258 / 129 / 48 (learned); flips unverified → verified
2 / 9 / 30 vs 2 / 13 / 44, verified → unverified 0 for both; fp64 gates 1–2e-15. The full `l1N` (job 39816017) has the same
+10.8 % mean on its finished set; its per-instance lines are pending. Small_6 round-5 screen complete: unified rule **1.08**
(42 / 0; 18 / 11 head to head), QK-only 1.06, closed form 1.06, both-sided `l1N` 0.56.

**23:54 — small_6 closed form on the paired protocol (job 39787014, first gauge set done; learned set joined from the earlier
paired eval on the same 294 instances).** Certified radius: stock 0.02199 → `svd_jacN_all` (24 boxes) 0.02465 (**+12.1 %**;
larger on 276, smaller on 0, equal 18), learned 0.02486 (+13.1 %; 273 / 0 / 21): the closed form alone takes **92 %** of the
learned gain on small_6 and is larger than the learned gauge on 142 instances, smaller on 133. The 2-box `svd_jac` (job 39773109)
scores 0.90 (+11.8 %; 254 / 2 / 38; 142 / 138 vs learned), so the probe count barely matters here. Fixed-eps verified
275 / 181 / 41 (stock) → 275 / 182 / 92 (closed form) vs 275 / 183 / 95 (learned); no reverse flips. Split check: `cmd_eval`
hard-codes the test split, so every paired number is on sentences the learners never saw (they tuned on dev). The screen's
1.06 for this gauge was slightly optimistic (paired 0.92) — small_6 is at parity per instance, not above it.
User asked to finish the search for a better formula with sub-agents; three launched: paired-gap bucketing + gauge-matrix
comparison, surrogate-cost-at-the-learned-gauge diagnostic (`--cost_only`), and a theory note on CROWN's bilinear slack vs the ℓ1
surrogate. Maintenance at 04:00 caps any GPU round at ≈ 2.5 h.

**00:12 (2026-09-08) — two diagnostics on the residual gap (sub-agent reports in the job tmp dir).**
(1) *Where the learned gauge's edge is:* not sentence length, position, radius quartile or eps — it is a **label split**.
On big_3 the unified rule is larger than the learned gauge on every label-0 instance it does not tie (100 / 0; share 1.09) and
smaller on every label-1 instance (0 / 97; share 0.69); the whole eps-0.0287 verified deficit (34 vs 48) is label 1 (4 vs 18).
small_6 closed form: label 0 share 0.57 (0 / 133), label 1 share 1.42 (142 / 0). Yelp `l1N`: label 0 0.64, label 1 0.97.
The manual construction is sign-blind (M and N enter only through norms); the learner maximises the *signed* margin bound and
trades one class against the other, in opposite directions on the two SST models (structural, not class imbalance).
(2) *Gauge matrices:* learned ≈ closed form ∘ near-orthogonal rotation (to-orthogonal 0.13–0.35) ∘ mild stretch (singular values
0.79–1.34 at big_3 layer 0); the ℓ1 refinement instead moves the closed form by a large stretch (up to 2.9; cond 30 on small_6).
The ℓ2 closed form has a flat O(d_h) valley per head (Gᵀ A = G⁻¹ B = √S Uᵀ for every G·Q), so its rotation is a numerical
accident — and it is exactly what the Yelp probe seed changes: seeds 0 and 1 give identical metrics G Gᵀ (distance 0.00–0.05)
with held-out shares 0.66 vs 0.28.
(3) *Objective vs optimiser:* the ℓ1-with-N surrogate at the learned gauge is HIGHER than at the manual optimum on every model,
layer and side (big_3 total 0.679 vs 0.576 relative to identity; Yelp 0.971 vs 0.947; small_6 0.744 vs 0.731; at big_3 layer 0
QK: closed form 0.970 → `l1N` 0.795, learned 0.865). The surrogate orders gauged < identity correctly but ranks manual below
learned — the remaining gap is an objective mismatch, not an optimisation failure; more optimisation cannot close it.
Follow-ups launched for the last GPU window before the 04:00 maintenance: rotation-only ℓ1 refinement `l1N_rot` (closed-form
metric kept, Cayley-parametrised rotation optimised on the ℓ1 surrogate; unified variant too) screened on all three models +
Yelp seed 1 (round 6), and per-class learners on big_3 (label-0-only and label-1-only boxes) to bound what a sign-aware rule
could gain. Interval-weight run (job 39771549, small_6) hit its 8 h limit after the stock and gauged sets (identical to the
paired eval, +13.1 %); the fp32-interval-weight sets did not finish — resubmit after the maintenance with a longer limit.

**00:20 — small_6 2-box paired eval complete (job 39773109, all four sets).** `svd_jac` from 2 probe boxes +11.8 % (254 / 2;
share 0.90), learned +13.1 %, and the isotropic closed form `svd_iso` (same 2 boxes, no Jacobian shaping) only **+3.9 %** with
59 smaller radii (212 / 59 / 23; eps-0.03 verified 78 vs 92–95). So on small_6 the Jacobian box shape is what carries the closed
form (0.90 with it, 0.30 without), while the probe count (2 vs 24 boxes) is nearly irrelevant.

**2026-09-09 08:42 — results that landed after the sub-agents died (all three hit the model's usage limit at ≈00:25 on
09-08; the round-6 screens and the per-class screen were never submitted — resubmitted this morning after the maintenance).**
- **small_6, refined rule on the paired protocol (job 39828671, 294 test instances; learned set joined from the earlier paired
  eval):** stock 0.02199 → unified rule `l1N_qk+av0` 0.02487 (**+13.1 %**; larger on 277, smaller on 0, equal 17), QK-only `l1N_qk`
  0.02481 (+12.9 %; 277 / 0 / 17), learned 0.02486 (+13.1 %; 273 / 0 / 21). Share of the learned gain **1.00** (QK-only 0.98);
  head to head the unified rule is larger on 142 instances, smaller on 122. Fixed-eps verified 275 / 181 / 41 (stock) →
  275 / 183 / **98** (unified) vs 275 / 183 / 95 (learned); mean lb at eps 0.03 +2.08 vs +1.62; reverse flips 0; fp64 gates
  1–2e-15. The radius search bisects 10 times from 0.1, so radii are quantised at ≈1e-4 and the 1e-5 difference of the means is a
  **tie**; the resolvable edges are head to head (142 vs 122) and eps-0.03 verified (98 vs 95), neither of which is the
  pre-registered metric (mean above the learned gauge's on two models) — so the bar is not met, but small_6 is at parity.
  By label the tie is the same class trade as the closed form, evened out: label 0 share 0.79 (0 / 122 head to head; closed
  form 0.57), label 1 share 1.30 (142 / 0; closed form 1.42).
- **big_3, full `l1N` paired (job 39816017):** 0.01841 (+10.8 %; 276 / 0; vs learned 101 / 97) = share 0.91, same as the unified
  rule; layer-0 splice `l1N@L0` 0.01835 (+10.4 %; share 0.87; vs learned 87 / 98). Verified 258 / 125 / 34 vs learned 258 / 129 / 48.
- **Proxy-cost diagnostic, final report:** (a) the learned gauge is above the manual optimum on the ℓ1-with-N surrogate in *every*
  (layer, side) cell of every model (two ties within 0.01); (b) Adam started AT the learned gauge walks it down to the manual
  optimum's surrogate value and away from the learned gauge (drift ‖G_l⁻¹G′ − I‖_F/√d_h 0.85–1.23 per layer; unrelated rotations
  ≈ 1.41) — the learned gauge is not a stationary point of the surrogate; (c) Yelp cross-draw: the seed-0 and seed-1 gauges have
  identical surrogate values to 3–4 decimals under either probe draw while their CROWN shares differ 2× — the surrogate is flat
  along the direction the probe draw moves the gauge (layers 1–2 drift 1.3 between draws; layer 0 identical because the no_var
  LayerNorm Jacobian is data-free). No verifier-free selection rule exists in this surrogate.
- Per-class learners on big_3 finished (label 0: 53 boxes, label 1: 17 of the 70; 20 / 14 min); their screen failed in 23 s
  (absolute gauge paths word-split on the spaces in the repo path) — resubmitted with relative paths (job 39881644).
- Resubmitted: round-6 rotation-only screens r6_big3 / r6_yelp3 / r6_yelp3_s1 / r6_small6 (jobs 39881639–39881642; smoke passed
  on 09-08, rc=0), interval-weight run with a 30 h limit (job 39881645, resumes the finished stock / gauged sets).

**09:08 — interval-weight rerun (job 39881645) cancelled at the user's request after 26 min** (the fp32-interval tier turns the
24 attention projections into bilinear nodes, so each CROWN call is ≥ 5× slower and the full 294-instance protocol needs > 11 h
per set). The stock / gauged sets in `results/deept_small6_wint_eval_short_seed0.json` stay valid; if the rigorous-transfer check
is wanted later, run it on the 24-instance fp64-gate subset or at the certified radius only (one call per instance).

**09:25 — exactness measurement (CPU, small_6, 6 test sentences × all positions × 32 random points per box at eps 0.02, fp64
arithmetic, |logit| ≤ 2.3):** max |original − gauged| = 8.9e-16 with the folded attention weights kept in fp64, **3.9e-9** with
them rounded to fp32 (the network the verifier bounds). The fp32-interval tier's bound loss (2e-4 in the smoke run) is therefore
~5 orders looser than the real difference — McCormick slack on 128×128 weight intervals; fp64-ulp intervals would bring the
certified loss to ~1e-12. User's standing requirement: every certified instance must carry an exact-equivalence guarantee, not a
sampled check. Proposed design (not yet built): bisect with the rounded network, then certify the found radius and each fixed eps
with ONE fp64 interval-weight CROWN call (step down a bisection level if it fails) — ≈ 1.5× today's runtime instead of 5×.

**09:25 — round 6, big_3 held-out screen (job 39881639, 42 min):** rotation-only ℓ1 refinement of the closed form (Cayley, no
cond penalty) `l1N_rot` share **0.87** (46 / 0; vs learned 5 / 19), its unified variant `l1N_rot_qk+av0` 0.83 (3 / 21); unified
rule 0.83 (4 / 21), closed form 0.56, learned 1.00. So restricting the ℓ1 step to the closed form's flat rotation valley gives the
same screen number as the full stretch (+0.04 on both sides, within the 24-sentence screen's noise), with the closed form's
conditioning instead of cond 11–30 — a cleaner rule, not a better one. Yelp (both draws), small_6 and the per-class screen pending.

**09:35 — round 6, Yelp small_3 held-out screen, probe seed 0 (job 39881640, 51 min):** `l1N_rot` 0.83 (42 / 2; vs learned 4 / 27),
`l1N_rot_qk+av0` 0.76, unified rule 0.82, closed form 0.66, full `l1N` was 0.88 in round 5. Rotation-only matches the unified
rule here and stays below the full ℓ1 step; as predicted from the cross-draw table, the rotation valley is where the surrogate
is flat, so it cannot pick a better point than the stretch does. Seed 1, small_6 and the per-class screen pending.

**09:45 — per-class learned gauges on big_3 (learners: jobs 39832933 / 39832938, label-0-only 53 boxes and label-1-only 17 boxes
of the 70 tuning boxes, same settings as the single learner; held-out screen job 39881644, 24 test sentences / 48 boxes, split by
label with `screen_label_split.py`).** Radius gain vs stock (share of the single learned gauge; head to head vs it):
- label 0 (22 boxes): single learned +9.5 %; label-0 learner +9.9 % (1.04; 3 / 0); **manual unified rule +10.0 % (1.05; 4 / 0)**;
  closed form +8.0 %; label-1 learner +3.7 %.
- label 1 (26 boxes): single learned +15.1 %; **label-1 learner +18.5 % (1.23; 19 / 0)** from only 17 tuning boxes; manual
  unified rule +11.1 % (0.73; 0 / 21); closed form +6.5 %; label-0 learner +7.6 %.
- combined "label-0 gauge on label-0 queries, label-1 gauge on label-1 queries": mean radius 0.01731 = **+14.9 %** vs +12.7 % for
  the single learned gauge (1.17×) and +10.6 % for the manual rule.
Reading: the class trade is real and worth ≈ 2 points on this model — a gauge chosen by the label being verified (legitimate: the
query knows its label, both are exact rewrites) beats the single learned gauge. The single learner sacrifices label 0 for label 1;
the sign-blind manual rule sits exactly at the label-0 ceiling (it cannot see the sign, so it lands on whichever class the
weights favour) and at 0.59 of the label-1 ceiling. This is where a sign-aware manual surrogate would earn its keep: up to
+7 points on label 1 here. Caveat: per-class is an axis available to the learner too, so the manual target becomes the
per-class learned gauge, not the single one. Queued: the same per-class test on small_6 (there the manual rule loses label 0).

**09:51 — round 6, Yelp small_3, probe seed 1 (job 39881641):** closed form 0.28, unified rule 0.56, `l1N_rot` 0.49,
`l1N_rot_qk+av0` 0.45 (learned 1.00 on this draw's own 24-sentence screen). Rotation-only does not rescue the bad probe draw —
as the cross-draw table predicted, the surrogate is flat along the direction the draw moves the gauge, so optimising inside
the rotation valley cannot find the seed-0 point. Round 6 verdict so far (3 of 4 screens): rotation-only ℓ1 = unified rule
within noise (big_3 +0.04, Yelp seed 0 +0.01, seed 1 −0.07), with better conditioning; not adopted as the procedure of record.

**11:06 — round 6, small_6 held-out screen (job 39881642, 2 h 04; done, sacct lagging):** `l1N_rot_qk+av0` 1.09 (42 / 0; vs
learned 19 / 11), unified rule 1.08 (18 / 11), closed form 1.06, both-sided `l1N_rot` 1.00 (12 / 13). Round 6 complete: the
rotation-only refinement equals the unified rule on every screen (big_3 0.87 vs 0.83, Yelp 0.83 vs 0.82, Yelp seed 1 0.49 vs
0.56, small_6 1.09 vs 1.08) while keeping the closed form's conditioning; the both-sided rotation variant is harmless on small_6
(1.00) where the both-sided stretch was harmful (0.52). Not adopted: same numbers, one more knob. Committed as a documented
option (`--l1rot 1`, `cand:l1N_rot*`).

**11:37 — PBverifierI rerun on DeepT with the plane parameter started at the Baseline plane (user request).** Their
`originPlus` initialises the pre-sigmoid variable at v = −4 (`Verifiers/Edge.py` rebuild_ori), i.e. X0 = −1 + 2σ(−4) ≈ −0.96:
almost the lower McCormick corner plane (worst-case slack 4·q_eps·k_eps) instead of Shi's centre plane X0 = 0 (slack
2·q_eps·k_eps); 20 Adam steps at lr 0.2 per layer on the layer's own width can at best walk back to the Baseline, and the
per-layer bounds are frozen before the last-layer margin step. Patch (their clone, our fork): `--init_v` flag in `Parser.py`
(default −4, so the published runs are unchanged), used in `rebuild_ori`; chain `run_pbv_chain_v0.sh <3|6> 15 8 16` runs
originPlus with `--init_v 0` on stock then gauged checkpoints (same 15 samples / 8 iters / max length 16 / ℓ∞ / seed 0 as the
published-default runs, 35 instances). Jobs 39885882 (small_3, L40S, ≈ 1.7 h) and 39885883 (small_6, ≈ 6 h). Results →
`results/pbv_s{3,6}_originPlus_v0_{stock,gauged}.json`; compare with `pbv_compare.py` against the origin / originPlus files.

**13:10 — small_6 per-class learned gauges (learners 39882898 / 39882899: label 0 = 53 boxes, label 1 = 15 boxes; screen
39882900, 24 test sentences / 48 boxes, 24 per label).** Radius gain vs stock (share of the single learned gauge; head to head vs it):
- label 0: single learned +15.8 %; label-0 learner +17.2 % (1.09; 7 / 0); manual unified rule +13.5 % (0.85; 0 / 11); closed form +10.7 %.
- label 1: single learned +12.8 %; **label-1 learner +23.1 % (1.81; 21 / 0)** from 15 tuning boxes; manual unified rule +16.5 %
  (1.29; 18 / 0); closed form +17.9 % (1.40).
- combined per-class ("lab0 on label-0 queries, lab1 on label-1"): mean radius 0.02022 = **+20.6 %** vs +14.1 % single learned
  (1.46×) and +15.2 % manual. Same picture as big_3 with the classes swapped: the single learner sacrifices label 1 here; the
  sign-blind manual rule lands on label 1 (above the single learned gauge there) and at 0.85 on label 0. Per-class gauges are the
  largest single improvement over the learned gauge found in this project (+6.5 points on small_6, +2.2 on big_3), and they are
  exact rewrites chosen by the label of the query.

**13:10 — PBverifierI started at the Baseline plane, small_3 (job 39885882, `--init_v 0`, 35 instances, same protocol).**
Stock: v = 0 gives 0.03544 vs their Baseline 0.03667 (**−3.3 %**, 9 larger / 19 smaller) — slightly WORSE than the published
default v = −4 (0.03569, −2.7 %). So the deficit of their optimised variant on DeepT weights is NOT the initial plane: even
starting exactly at the Baseline plane, the per-layer width objective and the per-layer freezing walk the bound below the
Baseline, and nothing tracks the margin against the Baseline value. **Correction to the 09-06 note in RELATED_WORK.md, which
attributed the deficit to the initialisation.** Gauge under PBverifierI(v = 0): 0.03544 → 0.03599, **+1.6 %, larger on 32 /
smaller on 0 / equal 3** (default init: −1.0 %, 21 / 12) — small, as everything is on small_3, but one-sided. small_6 pending
(job 39885883, ≈ 17:40).

**13:14 — write-up: per-class attention gauges (consolidating the 09-08 00:12 diagnostic and the 09-09 09:45 / 13:10 results).**

*Finding.* A single learned gauge trades one class against the other. Bucketing the paired test results by label: big_3
unified manual rule vs learned gauge is 100 / 0 on label-0 instances (share 1.09) and 0 / 97 on label 1 (0.69); small_6 the
reverse (0 / 122 on label 0, 0.79; 142 / 0 on label 1, 1.30); Yelp l1N 0.64 / 0.97. Length, position, radius and eps buckets are
flat. The manual construction is sign-blind (M and N enter through norms); the learner maximises the *signed* margin bound
logit[y] − logit[1−y], whose relaxation picks lower/upper planes by the sign of every backward coefficient, so it can favour a
class. The favoured class is model-structural (label 1 on big_3, label 0 on small_6; same 60 dev sentences for both learners).

*Method.* `deept_gauge.py learn --label {0,1}` keeps only the tuning boxes of one label; everything else as the gauges of
record (dev split ≤ 8 tokens, 3 positions/sentence, eps = stock radius, CROWN margin objective, Adam 0.01 × 120, cond penalty,
best step by held-in mean lb). Boxes: big_3 53 / 17, small_6 53 / 15; 14–55 min per learner on an A100. Evaluation: the
formula work's held-out screen (24 test sentences, 2 positions, 48 boxes, certified radius by bisection), split by label with
`screen_label_split.py` (reconstructs the screen's draws; validated on the r5 big_3 screen).

*Results (radius gain vs stock; head to head vs the single learned gauge).*
| gauge | small_6 label 0 | small_6 label 1 | small_6 combined | big_3 label 0 | big_3 label 1 | big_3 combined |
|---|---|---|---|---|---|---|
| single learned | +15.8 % | +12.8 % | +14.1 % | +9.5 % | +15.1 % | +12.7 % |
| label-0 learner | +17.2 % (7 / 0) | +5.8 % (0 / 22) | — | +9.9 % (3 / 0) | +7.6 % (0 / 23) | — |
| label-1 learner | +5.4 % (0 / 20) | +23.1 % (21 / 0) | — | +3.7 % (0 / 19) | +18.5 % (19 / 0) | — |
| **per-class (chosen by label)** | +17.2 % | +23.1 % | **+20.6 %** | +9.9 % | +18.5 % | **+14.9 %** |
| manual unified rule | +13.5 % (0 / 11) | +16.5 % (18 / 0) | +15.2 % | +10.0 % (4 / 0) | +11.1 % (0 / 21) | +10.6 % |
| closed form | +10.7 % | +17.9 % | +14.9 % | +8.0 % | +6.5 % | +7.1 % |

*Why it is legitimate.* Both per-class gauges are exact G G⁻¹ rewrites (fp64 gate ≤ 6e-15); the query certifies label y and
already uses y in the margin, so choosing the rewrite by y adds no information; verification-time cost is zero (two weight
files); training cost doubles once per model. On its own class the per-class gauge is larger than the single learned gauge on
21 / 0, 7 / 0, 19 / 0, 3 / 0 boxes and smaller on none. The label-1 learners used 15–17 boxes with normal conditioning (the
cond-28 overfit regime of the 5–13-box verifier-trained gauges did not appear), but the paired-protocol confirmation on the
full 288–294 test instances is still to be run.

*Implication for the manual formula.* The sign-blind rule lands on the class the weights favour (big_3 label 0: 1.05 of the
label-0 learner; small_6 label 1: 0.71 of the label-1 learner's +23.1 %) and well below the ceiling on the other. A sign-aware
surrogate — the downstream margin functional with its sign, per class — is the one open piece; its target is now the per-class
learner, not the single one. Round 6 (rotation-only ℓ1) is closed: equal to the unified rule on all four screens, better
conditioning, not adopted.

*Next.* (1) Paired eval of the per-class gauges on both models (per-label join with the existing stock sets). (2) Per-class on
Yelp small_3 and the ViT. (3) Sign-aware manual surrogate. (4) The fp64 + error-budget verification mode for unconditional
certificates (fold error 1e-15, forward error ≈ 1e-13, budget 1e-9; ≈ 2× runtime) — replaces the interval-weight tier.
The artifact page published earlier this afternoon duplicates this entry; PROGRESS.md and RELATED_WORK.md are the record.

**13:19 — round of learners warm-started from the manual unified rule (user request): single and per-label, small_6 and
big_3.** `deept_gauge.py learn --gauge <unified rule .pt> --label {-1,0,1}`, all other settings as the gauges of record
(120 steps, lr 0.01, cond penalty, best step by held-in mean lb; the init itself is the step-−1 candidate). Starts:
`formula_sst_bert_small_6_l1N_qk_av0_r4.pt`, `formula_sst_bert_big_3_l1N_qk_av0_r3.pt`. Jobs 39888760–2 (small_6: all /
label 0 / label 1), 39888764–6 (big_3), each ≤ 1.5 h on ckpt A100s; dependent held-out screens 39888763 (small_6, ≤ 4 h) and
39888767 (big_3) with nine names: identity, learned, lab0, lab1, init, initlab0, initlab1, closed form, unified rule →
`results/formula_{small6,big3}_init.json`, split by label with `screen_label_split.py`. Questions: does the warm start reach or
pass the identity-initialised learner (single: the Yelp ceiling test said tie), and does it reach the per-class learners on each
label (the unified rule is already at the label-0 ceiling on big_3 and above the single gauge on label 1 of small_6)?

**13:20 — label-conditioned manual unified rule (user request: a manual rule that produces a gauge for a specific label).**
Cheapest honest version of the sign-aware idea: the same construction (closed form `svd_jacN_all` + ℓ1 QK at every layer + ℓ1
AV at layer 0) with its random-token probes restricted to sequences the model predicts as the target label
(`gauge_formula.py --probe_label {0,1}`; random tokens carry the model's prediction as their label, so the filter is
verifier-free and label-free), so the box shapes M, the downstream margin functionals N_out and every cand:* row are built
for that class. n_sent 24 → ≈ 12 probes per class (the 2-vs-24-box test says the count is not the limiting factor). Screens
`plab{0,1}_{big3,small6}` (chain modes; smoke 39888861, then jobs 39888862 / 39888866 / 39888867 / 39888868) on the usual 24
test sentences with identity, the single learned gauge, the per-class learner of that label, the closed form and the unified
rule from those probes → `results/formula_<model>_plab<y>.json`, split by label afterwards. Read-out: on label y, does the
label-y manual rule move from the all-probe rule (big_3 label 1: +11.1 %, small_6 label 0: +13.5 %) toward the per-class
learner (+18.5 % / +17.2 %)? It cannot express the plane-sign trade itself (the surrogate is still sign-blind); it tests whether
class-conditioned box shapes and sensitivities carry part of the per-class gain.

**13:38 — alternating alpha-CROWN / gauge learner built and launched on small_6 (user request).** `deept_gauge.py learn
--alpha_iters K`: per tuning box, (1) inner loop with the weights frozen — CROWN-Optimized, K Adam iterations on the relaxation
parameters α at the current gauge (best α left in the module's nodes); (2) outer step with α frozen (`opt_reuse` on the
optimisable nodes) — one plain backward pass whose graph reaches the effective weights → gradient to G. Danskin: d/dG max_α
lb(G, α) = ∂lb/∂G at the optimal α, so this is the exact envelope gradient without differentiating through the α loop. A fresh
BoundedModule per call (each retains 4.5–8 GiB of α state); the held-in selection score is the α-optimised bound on 16 boxes
every 20 steps, and the best gauge is checkpointed at every eval (ckpt is preemptible). **Correction to my earlier estimate:**
the measured α-CROWN peaks on small_6 are 36 GiB at 5 tokens and 62 GiB at 6 (cmd_eval_alpha's probes), not ≈ 25 GiB at
8 tokens, so the tuning boxes are restricted to ≤ 6-token dev sentences on the 80 GB A100, and a call costs ≈ 70 s at 20
iterations. Run: warm start from the plain-learned gauge `deept_small6_seed0.pt`, lr 0.005, 60 steps × 4 boxes, K = 10,
≈ 3.5 h, then `eval_alpha --skip_stock 1` on the same 15 test sentences ≤ 6 tokens / 49 instances / eps 0.02, 0.03 as the
existing alpha-tier eval (`results/deept_small6_eval_alpha_seed0.json`: stock 24 / 0 verified, plain-learned gauge 27 / 4,
tighter on 47 / 49), ≈ 2.7 h. Smoke 39889885 (40 min), full 39889886 (afterok, 9 h limit) →
`gauges/deept_small6_alt_seed0.pt`, `results/deept_small6_alt_eval_alpha_seed0.json`. Read-out: does the α-trained gauge beat the
plain-trained one on the α tier (per instance, same 49 boxes), and by how much relative to the +0.33 mean-lb gap between the
plain-trained gauge and stock?

**14:16 — alternating learner smoke passed (job 39889885, 27 min; full run 39889886 released).** Sanity line at step 0: α-optimised
lb +0.93 / +3.57 vs the 'reuse' pass that carries the weight gradient +0.79 / +3.41 — the reuse pass recomputes the intermediate
bounds with the reused α instead of the fixed ones the α loop optimised against, so the outer objective is a slightly looser
α-tier bound (≈ 0.15 below the optimised one). Acceptable for the gradient; if the full run's gauge underperforms, pass the α
loop's intermediate bounds into the reuse pass. fp64 gate 7e-16; α-tier smoke eval 7 / 7 tighter than stock.

**14:41 — label-conditioned manual rule, big_3 (jobs 39888862 / 39888866; 12 probes of the target class each; same 22 + 26
held-out boxes).** Radius gain vs stock on the target class / on the other class:
- rule from label-1 probes: label 1 **+12.0 %** (all-probe unified rule +11.1 %; label-1 learner +18.5 %; single learned +15.1 %);
  label 0 +8.7 % (all-probe +10.0 %).
- rule from label-0 probes: label 0 +9.8 % (all-probe +10.0 %; label-0 learner +9.9 % — already at that ceiling); label 1 +9.8 %
  (all-probe +11.1 %).
Reading: restricting the probes to the target class moves the manual rule by about ± 1 point in the expected directions, i.e.
class-conditioned box shapes and margin sensitivities carry only a small part of the per-class gain; the bulk of the +7 points
between the manual rule and the label-1 learner is the plane-sign trade that a norm-based surrogate cannot express. A per-class
manual rule needs a sign-aware surrogate (linear-bound structure: invariant part ± the gauge-dependent width term, with the
downstream coefficients' signs). small_6 label screens pending (jobs 39888867 / 39888868).

**15:42 — learners warm-started from the manual unified rule, big_3 (jobs 39888764–6; screen 39888767; 22 + 26 held-out
boxes).** Radius gain vs stock (head to head vs the identity-initialised learned gauge):
| start → learner | all | label 0 | label 1 |
|---|---|---|---|
| identity → single (gauge of record) | +12.7 % | +9.5 % | +15.1 % |
| **unified rule → single** | **+13.7 %** (15 / 0) | +10.0 % (4 / 0) | +16.4 % (11 / 0) |
| identity → label-0 / label-1 learners, chosen by label | +14.9 % | +9.9 % | +18.5 % |
| **unified rule → label-0 / label-1 learners, chosen by label** | **+16.1 %** | +10.7 % (7 / 0) | **+20.0 %** (21 / 0) |
| unified rule (manual, no learner) | +10.6 % | +10.0 % | +11.1 % |
The warm start from the manual rule ends ABOVE the identity-initialised learner on this screen, one-sidedly (single: larger on
15 boxes, smaller on 0; per-label: 7 / 0 and 21 / 0). Best steps 110 / 119 / 119 for the warm starts and the same 110 / 119 / 119
for the identity starts, so both families were still moving at the 120-step cap; conditioning 10.6 (warm single) vs 11.6, and
3–4 for all per-label gauges. This differs from the Yelp ceiling test (warm start from the CLOSED form, lr 0.005: tie); here the
start is the refined rule and lr 0.01. Caveat: 48-box screen; the paired protocol on 288 instances is
the confirmation. small_6 counterpart pending (job 39888763, waiting on the all-label learner 39888760).

**16:17 — label-conditioned manual rule, small_6, label-0 probes (job 39888867; 15 probes; 24 + 24 held-out boxes).** Label 0:
rule from label-0 probes **+13.3 %** vs the all-probe unified rule +13.5 % (label-0 learner +17.2 %, single learned +15.8 %) —
no movement at all; label 1: +14.3 % vs +16.5 % all-probe. Same verdict as big_3: conditioning the probe distribution on the
target class does not carry the per-class gain (≤ 1 point, here 0); the trade lives in the plane-sign structure of the bound.
The label-conditioned rule is not adopted. Label-1 screen (39888868) pending, expected to confirm.

**16:37 — label-conditioned manual rule, small_6, label-1 probes (job 39888868; 9 probes; same 24 + 24 boxes).** Label 1: rule
from label-1 probes **+18.9 %** vs the all-probe unified rule +16.5 % (closed form from label-1 probes +19.7 % vs +17.9 %;
label-1 learner +23.1 %; single learned +12.8 %) — larger than the single learned gauge on 21 / 0; label 0: +11.5 % vs +13.5 %.
Summary of the four label-conditioned screens (target-class gain, conditioned vs all-probe rule → per-class learner):
big_3 label 0: 9.8 vs 10.0 → 9.9; big_3 label 1: 12.0 vs 11.1 → 18.5; small_6 label 0: 13.3 vs 13.5 → 17.2; small_6 label 1:
18.9 vs 16.5 → 23.1. Conditioning the probes helps only on the class the sign-blind rule already favours (small_6 label 1,
+2.4 points, reaching 0.82 of the per-class learner's gain) and does nothing on the unfavoured class (≤ 1 point) — which is
exactly where the per-class learners win big. Verdict stands: the per-class gain is the plane-sign trade; a per-class manual
rule needs a sign-aware surrogate. Not adopted as a procedure; `--probe_label` stays as a documented option.
