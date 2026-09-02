# TASO — codebase map & spec (the whole fork, as our code)

Companion to `TENSAT_SUMMARY.md`. TASO (`taso/`, fork `xvade/TASO` @ branch
`klone-cpu-gpu-build`) is **part of our codebase**, not a black-box dependency —
this file documents the *entirety* of it at architecture altitude, and points to
where each piece is specified and tested. The fork *delta* is indexed separately in
`taso/MODIFICATIONS.md`; per-bug rationale is in `BUGS.md`; open gaps are in
`PROBLEMATIC.md`.

Upstream TASO (OSDI'19, *Optimizing Deep Learning Computation with Automatic
Generation of Graph Substitutions*) is a superoptimizer that rewrites a tensor
graph to a faster equivalent, **verifying** each candidate rewrite against a set of
operator specifications. We use two of its capabilities and **ignore its runtime
optimizer entirely**:

1. **Model ingest/export** — ONNX ⇄ a TASO dataflow graph (`.taso`), the interchange
   `tensat` reads and `reconstruct_generic.py` writes.
2. **Rule generation** — enumerate + verify candidate graph substitutions, emitted as
   `graph_subst.pb`, which `NNs/pb2egg.py` lowers into the tensat rule corpus.

TASO's own cost-driven search (`Graph::optimize`) is **never called** in our
pipeline; extraction is tensat's job, driven by verifiability (VerifCost), so op
**cost measurement is force-zeroed** (see §5, `taso/MODIFICATIONS.md`).

## 1. Repository map

| Path | Role |
|---|---|
| `include/taso/ops.h` | The one core header: `OpType` enum, `Tensor`/`TensorHandle`, `OpBase` & `Op`, **`Graph`** (dataflow graph + builders + `optimize`), **`Model`** (op factory + cost + backend dispatch), `SplitInfo`. |
| `include/taso/substitution.h` | `GraphXfer` — the source→target pattern rewrite engine. |
| `src/core/*.cc` (29) | Per-op **shape inference + `collect_costs`** (`conv2d.cc`, `matmul.cc`, `element.cc`, `pool2d.cc`, `mul.cc`, …), plus `ops.cc` (Graph/Model glue, attr accessors), `substitution.cc` (`GraphXfer::run`, `Graph::optimize`, `Graph::export_to_file`). Backend-independent. **Spec + per-op shape table: [`taso/src/core/README.md`](taso/src/core/README.md).** |
| `src/cudnn/*.cu` (26) | GPU backend: real cuDNN/cuBLAS **kernels + `measure_*_cost`** (executes and times each op). Compiled only when `USE_CUDA=ON`. |
| `src/cpu/*.cc` (3) | CPU backend (`USE_CUDA=OFF`): `execution_stubs_cpu.cc` (no real exec), `ops_cpu.cc`, **`measure_cost_cpu.cc`** (analytic cost — now force-zeroed). |
| `src/generator/` | The rule generator (`generator.cc`) + build scripts (`compile*.sh`) + the produced corpus `graph_subst.pb`. Own docs in `src/generator/README.md`. |
| `python/taso/_cython/{core.pyx,CCore.pxd}` | Cython bindings: `PyGraph`/`PyTensor` wrap `Graph`/`TensorHandle`; the module-level `op_table` (`int → ONNX-ish name`); `get_padding_mode`/`get_activation_mode`. Compiles to `core.*.so`. |
| `python/taso/__init__.py` | Pure-python ONNX layer: **`load_onnx`** (importer, dispatched through `xf_operators`), **`export_onnx`**, and the `operator_attrs` / `input_weight_names` / `_onnx_op_name_overrides` tables the exporter uses. |
| `CMakeLists.txt`, `config.cmake` | Backend selection (`USE_CUDA`, `USE_DNNL`) and the CPU/GPU source globbing. |

Upstream also ships `src/dnnl/` (a oneDNN CPU backend, 21 commits); **our fork does
not include it** — deliberately, since it only provides real op cost, which we don't
use (see the DNNL analysis in the session record / `PROBLEMATIC.md`).

## 2. Core data model (`ops.h`)

- **`Tensor` / `TensorHandle`** — a tensor value: `numDim`, `dim[]`, `stride[]`, an
  owning `Op`, an output index. `TensorHandle` is `Tensor*`, the currency the Graph
  builders pass around.
- **`OpBase` / `Op`** — an operator instance. `OpBase` holds inputs, `numInputs`,
  computed output `Tensor`s, and a `runtime` (its measured/estimated cost). Each op
  *type* subclasses `OpBase` (e.g. `Conv2D`, `Matmul`, `Mul`, `Element`) in its
  `src/core/*.cc`, where the constructor does **shape inference**.
- **`Graph`** — a DAG of `Op`s (`inEdges`/`outEdges` maps). Exposes the **builder API**
  (`conv2d`, `matmul`, `element`, `pool2d_*`, `transpose`, `mul`, `concat`, `split`,
  `new_input`, `new_weight`, …), each of which calls a `Model::get_or_create_*`
  factory. Also `total_cost`, `optimize` (unused), `export_to_file`.
- **`Model`** — the op **factory + cache + backend handle**. `get_or_create_X(...)`
  builds-or-dedups an `Op`, then calls **`measure_X_cost(op)`** (§5). Holds the cuDNN
  handles / workspace in the GPU build.
- **`OpType` enum** — the op vocabulary. Upstream set + our appended
  `OP_EW_SUB`/`OP_EW_MAX`/`OP_EW_MIN` (26/27/28) and the const family
  (`OP_CONSTANT_*`). This enum's *order* is load-bearing downstream: `pb2egg.py` and
  `reconstruct_generic.py` derive `OP_MUL`'s int from `OP_MATMUL`'s position (it has
  no `op_table` entry) — see `reconstruct_op_names`.

## 3. The op lifecycle (one invariant to hold in your head)

`graph.X(...)` → `Model::get_or_create_X` → **`X::X(...)` ctor computes output shape**
→ **`measure_X_cost(op)` sets `op->runtime`** → the `Op` is cached and returned. So
*building* a graph both infers shapes (which tensat's `TensorAnalysis` depends on) and,
in the stock code, *measures cost*. We keep the shape inference and neuter the cost.

## 4. Substitution engine & the rule generator

- **`GraphXfer`** (`substitution.{h,cc}`) matches a *source* op pattern in a Graph and
  rewrites it to a *target* pattern, subject to the substitution's conditions. This is
  the shared machinery behind both TASO's optimizer and the generator.
- **`Graph::optimize`** — TASO's backtracking cost-driven search over `GraphXfer`s.
  **Unused by us.** (Documented here so nobody wires it in by accident; extraction is
  tensat's, and cost is zeroed so it would be meaningless anyway.)
- **The generator** (`src/generator/generator.cc`) enumerates candidate source/target
  op pairs up to a depth, **verifies** each against the op specs, and serializes the
  survivors to `graph_subst.pb` (a `RuleCollection` protobuf). Our fork's knobs —
  min/max/sub ops, `PWL_FOCUS`, `GEN_MAX_DEPTH`, the `RELAX_*` quotient relaxations,
  `GEN_COMMUTE` — and the presence-check gotcha and 2 GB protobuf trap are in
  `src/generator/README.md`.

## 5. Cost & backends — and why cost is off

Every op subclass computes `flops`/`mem_acc` in its `collect_costs`; the backend's
`measure_*_cost` turns that into `op->runtime` (cuDNN times a real kernel launch; the
CPU stub estimated `(flops+mem_acc)/UNITS_PER_MS`). `Model::get_or_create_*` calls it
at build time; the only reader is tensat's runtime-driven `CostModel::get_self_cost`.

**This project never uses runtime cost** (it is not about efficiency; extraction is
verifiability-driven). So `estimate_runtime` in `measure_cost_cpu.cc` is **force-zeroed**
(`op->runtime = 0` for the whole CPU build), guarded by an opt-in
`TASO_ENABLE_COST_MEASUREMENT`. Consequence: greedy/ILP extraction reports `Best cost:
0.0` (proved in `run_tests.sh` Test 13) and is degenerate — fine, because we extract
with tensat's VerifCost, not CostModel. **Gap:** the 20 cuDNN `measure_*_cost` functions
(`USE_CUDA=ON` build) are *not* guarded, so a GPU build still measures cost and can hit
the small-N cuBLAS SGEMM abort — see `PROBLEMATIC.md`.

## 6. Python layer — the pipeline's actual surface

- **`load_onnx(path)`** walks the ONNX graph, dispatching each node through
  `xf_operators[op.op_type]` to a `_<op>` builder that calls the Graph API. An op
  **absent from `xf_operators` is silently skipped**, which orphans everything
  downstream (the failure mode behind the MatMul-casing and Sigmoid bugs). Coverage
  and the builders are tested env-independently by `NNs/tests/test_taso_importer.py`.
- **`export_onnx(graph)`** walks the TASO Graph back to ONNX, using `op_table` (names),
  `operator_attrs` (which attrs each op emits), `input_weight_names`, and
  `_onnx_op_name_overrides`. This is the path `reconstruct_generic.py` relies on.
- **`op_table`** (in `core.pyx`) is the `int→name` map; note it is *incomplete*
  (`OP_MUL` has no entry) — consumers derive missing ints positionally.

## 7. How TASO plugs into the verifiability pipeline

```
ONNX model ─load_onnx→ TASO Graph ─export_to_file→ model.taso ─→ tensat (ingest/rewrite/extract)
generator ─→ graph_subst.pb ─pb2egg.py→ egg rules ─→ tensat (-r rules)
tensat extraction ─→ .model ─reconstruct_generic.py→ TASO Graph ─export_onnx→ ONNX ─→ ab-CROWN
```

## 8. Documentation / spec / test map

| Area | Doc | Test |
|---|---|---|
| Fork delta (index) | `taso/MODIFICATIONS.md` | — |
| ONNX importer registrations + Gemm bias + MatMul casing + Sigmoid/Tanh | this file §6; `BUGS.md` | `NNs/tests/test_taso_importer.py` (env-independent, mocks `taso.core`/`onnx`) |
| Exporter (`export_onnx`, attrs) + `export_op` min/max/sub | `BUGS.md`; `taso/MODIFICATIONS.md` | end-to-end via `reconstruct_generic.py` in the reconstruct env; `NNs/tests/test_reconstruct_arms.py` for the reconstruct-side dispatch |
| Cost zeroed / CPU backend | this file §5; `taso/MODIFICATIONS.md` | `run_tests.sh` **Test 13** (`Best cost: 0.0`, greedy still extracts) |
| Rule generator flags | `src/generator/README.md` | `src/generator/tests/test_flags_probe.sh` |
| `OpType` / `op_table` ordering invariant | this file §2; `reconstruct_generic.py` header | `test_reconstruct_arms.py` (enum-shift guard), `test_taso_importer.py` |

**Known gaps (→ `PROBLEMATIC.md`):** taso+onnx can't co-run in-repo (compiled core is
py-3.14/container-only; that env's onnx is broken), so importer/exporter end-to-end
tests are deferred to the GPU env and stood in for by the mock tests above; the cuDNN
`measure_*_cost` guard is unimplemented; and `Graph::optimize` / the CUDA-only kernels
remain upstream-only, undocumented here beyond their role.
