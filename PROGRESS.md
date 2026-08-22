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
