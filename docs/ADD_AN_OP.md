# Adding a new operator to the whole pipeline

A tensor op (say `gelu`, or a new elementwise/structural op) has to be taught to
**every** stage a rule travels through, or it fails — often silently, or at a
stage far from where you added it. This is the end-to-end contract, the file to
edit at each stage, and the **authoritative check** that proves that stage is
done. Skipping a stage is the single most common way to ship a broken op (see the
"application gap" story in `../PROBLEMATIC.md` #8: transpose/const passed emission,
parse, and Z3 verification but panicked tensat on *application*, because two
tensat stages were skipped).

## The pipeline, and where an op must be taught

```
 taso generator ──► graph_subst.pb ──► pb2egg.py ──► egg rules ──► tensat ──► extract ──► reconstruct ──► ONNX
   (C++)                                (Python)                    (Rust)                  (Python)
        │                                   │                    ┌───┴────┐
        │                                   │                    parse  apply
        └── taso core (compute/export)      └── Z3 verify (z3_verify_egg.py + tensor_axioms.py)
```

An op is only "done" when it survives all of: **generate → emit → parse → verify
→ apply → extract → reconstruct**. The two that are easy to forget are tensat
**apply** and tensat **make()** — parse success does not imply either.

## Stage-by-stage checklist

Legend for the **oracle** column: the concrete command/test that proves the stage
works — trust it over reading code.

### 1. taso generator — can the generator *build* the op?  (C++, only if the op should be enumerated into rules)
- **Edit:** `taso/src/generator/generator.cc` op set, and the `xflow/ops.h` shim
  (`taso/include/xflow/ops.h`) if the op is new to the shim. For a PWL/elementwise
  op, add it alongside the existing `EW_*` handling.
- **Build:** `taso/src/generator/compile_pwl.sh` (in-container). See
  `taso/src/generator/README.md` for the `-DPWL_FOCUS` / `GEN_MAX_DEPTH` /
  `RELAX_*`/`GEN_COMMUTE` flags (note: flags are **presence-checked**, `X=0` is ON).
- **Oracle:** the generator runs and `graph_subst.pb` contains rules using the op.
- Skip this stage if the op only needs to be *ingested* from existing pb corpora.

### 2. taso core — op semantics + serialization  (C++)
- **Edit:** `taso/src/core/<op>.cc` (compute + `get_int_parameter` for its params),
  `taso/src/core/ops.cc` `export_op` (so an extracted graph with the op exports to
  ONNX — a missing case is `Assertion 'false'`; see `../BUGS.md`), and the op enum
  in `taso/include/xflow/ops.h`.
- **Oracle:** reconstruct a graph containing the op to ONNX without crashing
  (`NNs/reconstruct_generic.py`), and `onnxruntime` runs it.

### 3. pb2egg.py — pb op → egg s-expression  (Python)
- **Edit:** `NNs/pb2egg.py` `operator_data`: `OP_X: ('eggname', [PM_PARAM, ...],
  'pi'|'ip')`. Params-first (`'pi'`) or input-first (`'ip'`) must match tensat's
  child order. If a param needs decoding (e.g. transpose's `PM_PERM` → a `dim_dim`
  Name leaf), special-case it in `build()` like `OP_TRANSPOSE`/`OP_CONCAT`.
- **Apply-safety:** add the egg name to `APPLY_SAFE_EGG_OPS` **only after** tensat
  stages 5+6 are done. Until then the op is emitted only under
  `--emit-unapplicable` (so it can't panic a real saturation). This gate is the fix
  for the application-gap class of bug.
- **Oracle:** `pb2egg` emits the op with `0` non-clean drops; then stage 4.

### 4. tensat egg language — the op exists as a term  (Rust)
- **Edit:** `tensat/src/model.rs` `define_language!` (`Mdl`):
  `"eggname" = Variant([Id; N])`. Choose `N` = #params + #tensor-inputs; params
  are `DataKind::Scalar`/`Name` leaf children, tensors are `DataKind::Tnsr`.
- **Oracle — `parse_check` (authoritative for parsing):**
  `tensat -m parse_check -r <rules>` → `0 FAIL`. The `Mdl` doc-comments and
  `converted_full*.txt` are **stale**; only `parse_check` is truth. Hand-write a
  candidate `(eggname ...)` rule and confirm it parses.

### 5. tensat make() — bottom-up metadata for the op  (Rust) — REQUIRED
- **Edit:** `tensat/src/model.rs` `TensorAnalysis::make()`, add an `Mdl::Variant(...)
  => { ... }` arm that computes the enode's `Data` (dtype, dims, `meta` pointer via a
  taso `g.<op>(...)` call, weight-name provenance). **Without this arm the op falls
  to `other => { println!(...); todo!() }` and panics the moment such an enode is
  created.**
- **Design note:** the op must have well-defined bottom-up metadata. The const ops
  (`Cpool`/`Iconv`/`Imatmul`/`Iewmul`) are `MagicConst` — shape-polymorphic, channels
  filled by the *consuming* op — so a standalone `Cpool(3,3)` has no sound metadata.
  That is why they were left `todo!()`. **Resolved for the identity consts** via a
  `DataKind::Const` marker + consumer resolution: `make(const)` emits a marker (no
  tensor), and the approved consumer (e.g. `ewmul` for `Iewmul`, since
  `ewmul(x,ones)==x`) returns the *other* operand's data. A central applier guard
  declines any non-approved parent of a Const child. See `Iewmul` in
  `../tensat/MODIFICATIONS.md` for the worked pattern (`Imatmul`/`Iconv` mirror it;
  `Cpool` still needs real materialization since it's `==poolavg`, not `==x`).

### 6. tensat apply — build the op during rewrite application  (Rust) — REQUIRED
- **Edit:** `tensat/src/rewrites.rs` — the tensor-building match (the one ending in
  `other => { println!(...); todo!() }`, ~line 897) needs an arm for the op, so a
  rule whose RHS produces the op can materialize it. Missing arm = panic on apply,
  *not* on parse.
- **Oracle — the apply-smoke test (authoritative for application):** a
  guaranteed-fire rule through a real saturation. Pattern (see `run_tests.sh` Test
  12): `(relu ?input_1)=>(<op> (relu ?input_1) ...)` run via
  `tensat -r rule.txt -s none --model_file NNs/mnist_tiny_mlp.taso --n_iter 2` — it
  must reach `Stopped:` / `Number of iterations`, **not** `todo!()`/`panicked`.
  Only once this passes may the op join `APPLY_SAFE_EGG_OPS` (stage 3).

### 7. tensat cost — extraction cost  (Rust)
- **Edit:** `tensat/src/optimize.rs` `CostModel::get_self_cost()` — add an arm (a
  real taso runtime, or a synthetic cost). See `tensat/MODIFICATIONS.md`. Verify
  it does not break `--verif_cost`.

### 8. tensat parse for special leaves — if the op has Name/Scalar params  (Rust)
- **Edit:** `tensat/src/parse.rs` / `model.rs` — a param encoded as a `Name` leaf
  (transpose `perm_name = "1_0"`, reshape `shape_name`) needs the parser to accept
  the `dim_dim...` token and `make()` to decode it. `parse_check` (stage 4) covers
  this once the language + parse handle the leaf.

### 9. Z3 verification — prove/reject rewrites using the op  (Python)
- **Lane 1** (`NNs/z3_verify_egg.py` `Builder.build`): interpret the op exactly if
  it's PWL (like `ewmax → If(a>=b,a,b)`), else route it to the uninterpreted-op
  branch (congruence). A new op that hits neither raises `ValueError` and the rule
  is counted `parse-errored` — add it explicitly.
- **Lane 2** (`NNs/tensor_axioms.py`): if the op has algebraic axioms in
  `taso/verify/verify.py`, declare its `Function` in `_OPS`, port the axioms into
  `AXIOMS`, and map the egg name in `build()`. Run with the `taso_py` env (z3 is
  not in the container python — `../PROBLEMATIC.md` #5).
- **Oracle:** `NNs/tests/test_z3_axioms.sh` — add a flip (a valid rewrite that
  should VERIFY) and a **negative canary** (a false rewrite that must stay
  unproven; with quantified axioms an inconsistent set proves everything, so the
  canary is the soundness guard).

### 10. reconstruct — op → real-weight ONNX  (Python)
- **Edit:** `NNs/reconstruct_generic.py`'s op dispatch, so an extracted `.model`
  DAG containing the op rebuilds to ONNX (via a graph op, or a numpy fold for
  weight-only cases — see `reconstruct_optimized.py` for the transpose numpy fold).
- **Oracle:** `NNs/verify_reconstruction*.py` — the rebuilt ONNX matches the
  seed-0 reference to atol 1e-4.

### 11. Tests — pin every stage you touched  (`NNs/tests/`)
Add to `run_tests.sh` / `test_z3_axioms.sh` following the existing pattern:
emission count + `0` non-clean, `parse_check` `0 FAIL`, **apply-smoke** (Test 12
pattern — the stage most often skipped), a Z3 flip + canary, and a reconstruction
round-trip if applicable. Fixtures must be **tracked** artifacts (carve a tiny
`.pb` from a corpus like `transpose_fixture.pb`); never depend on the multi-GB
uncommitted `reassoc_results` outputs.

## One-line summary of the traps this contract encodes
- **`parse_check` ≠ applicable.** Parsing proves stage 4; it says nothing about
  make() (5) or apply (6). An op can parse, Z3-verify, and still panic tensat.
- **make() and apply are two separate arms** (`model.rs` and `rewrites.rs`), both
  ending in `todo!()`. Both are required.
- **Only gate an op `APPLY_SAFE`** after the apply-smoke test passes.
- **Some ops can't be given sound bottom-up metadata** (the `MagicConst` const
  family) — those need a taso API, not a match arm; leave them gated and say so.
- **Generator flags are presence-checked**; **stale rule text files lie** — trust
  `parse_check`, the apply-smoke saturation, and the Z3 lanes, not comments.
