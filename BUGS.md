# Bugs found while building the TENSAT → alpha-beta-CROWN pipeline

Found 2026-08-22 while getting a real trained model (`mnist_tiny_mlp`, from
alpha-beta-CROWN's own examples) through: PyTorch → ONNX → `taso.load_onnx()`
→ tensat optimizer → real weights reconstructed → ONNX → verified numerically
against the original PyTorch model. All code is `Copyright 2019 Stanford` in
both repos, so these likely predate both `yycdavid/taso` and `xvade/tensat`
diverging from upstream `jiazhihao/TASO` — not confirmed against upstream,
but worth checking before filing there specifically.

Repos or their upstreams to consider filing against:
- `taso` fork used here: https://github.com/yycdavid/taso
- `tensat` fork used here: https://github.com/xvade/tensat
- Original: https://github.com/jiazhihao/TASO

---

## 1. `tensat`: `-f`/`--model_file` never calls the parser built for it

**Where:** `tensat/src/main.rs`, the `None =>` arm of the model-selection
`match` (originally line ~270-276, now fixed in our fork).

**Symptom:** Pointing `-f` at a file written by `taso.export_to_file()` (the
exact format `tensat/tests/parse.rs` tests against, and the format `-x`
itself writes) silently produced a degenerate one-node, zero-edge graph.
No error, no crash — `optimize()` ran to completion and reported "Nodes: 1,
Classes: 1" as if that were a valid input.

**Root cause:** `-f`'s actual code path was:
```rust
input_graph.parse().unwrap()
```
i.e. generic `RecExpr::from_str` — egg's own Lisp-style S-expression parser
(`(matmul (input ...) (weight ...))`), intended for small hand-written test
graphs. `tensat/src/parse.rs::parse_model()` is a *separate* function that
correctly parses the TASO-exported numbered-op-list format (op-id /
op-type-code / dependency-refs / shape-params) — it's tested directly in
`tests/parse.rs`, whose own comment describes the intended pipeline
(load_onnx → export_to_file → parse_model), but nothing in `main.rs` ever
called it. Two working, tested pieces of the same crate, never wired to
each other.

**Status:** Fixed in this fork — commit `e349c73`, "Fix -f/--model_file to
actually parse TASO-exported models." `-f` now calls `parse_model()`
directly.

---

## 2. `taso`: ONNX `Gemm` importer silently drops the bias input

**Where:** `taso/python/taso/__init__.py`, `_gemm()` (originally lines
254-262).

**Symptom:** Loading any ONNX model with a `Gemm` node that has a bias
(the common case — every `nn.Linear` layer exports as `Gemm(A, B, C)`)
produced a graph with the bias weight tensor present but never referenced
by anything — a dangling, unused node. No error. The resulting graph is
structurally different from the real network (ReLU ends up applied
directly to the raw matmul output instead of matmul+bias).

**Root cause:**
```python
def _gemm(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)  # fetches A, B, C
    ...
    outputs = graph.matmul(inputs[0], inputs[1])           # C never used
    return outputs
```
`_get_inputs` correctly fetches all three ONNX inputs per the
[Gemm spec](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm),
but only `inputs[0]`/`inputs[1]` (A, B) are ever used.

**Related, not fixed:** `_conv2d()` (same file, ~line 203) has the identical
pattern — it also fetches but never uses Conv's optional bias input
(`inputs[2]`). Not hit in this session (no conv layers involved), but the
same class of bug.

**Status:** Fixed for `Gemm` in this fork — commit `bd8ba5d`, "Fix ONNX Gemm
importer silently dropping the bias input." Now calls `graph.add(outputs,
inputs[2])` when a third input is present (TASO's `Model::broadcastable`,
`element.cc:19`, handles broadcasting a `[out_features]` bias against a
`[..., out_features]` matmul result with no reshape needed). `_conv2d`'s
identical gap is **not** fixed.

---

## 3. `taso`: `Graph::preprocess_weights()` collapses the graph

**Where:** `taso/src/core/ops.cc`, `Graph::preprocess_weights()` (lines
499-586), specifically the "Remove isolated nodes" phase (~lines 549-586).

**Symptom:** Calling `graph.preprocess_weights()` (exposed to Python in this
fork, see below — not exposed upstream) on a graph with 8 real ops
collapsed it down to a single stray `Reshape` node with no useful output.
Every `Matmul`/`Add`/`Relu`/`Transpose` vanished.

**Intended behavior:** `Graph::optimize()` calls this internally
(`ops.cc:477`) right before reporting final costs. Any op whose *entire*
input set is `OP_WEIGHT`-typed (e.g. a rewrite's `Transpose(weight)` or
`Concat(weight, weight)`) gets actually executed once (`op.ptr->map()` +
`forward()`) and replaced with a literal `Weight` node holding the real
computed result — constant-folding so the same transpose/concat doesn't
get redundantly re-run on every forward pass.

**Root cause (Phase 2 specifically):** the forward-reachability check used
to decide which nodes are prunable is:
```cpp
if (it2->srcOp.guid != GUID_WEIGHT) cnt++;
```
`GUID_WEIGHT` is `11` — a sentinel constant (`ops.h`: `GUID_INPUT=10,
GUID_WEIGHT=11, GUID_PRESERVED=19`) used elsewhere purely as a placeholder
in the *text export format* ("no real producer, this is a leaf"). It is
not a property any actually-constructed op's `.guid` field carries — real
ops get unique guids from an incrementing counter starting well above 19.
So this check is comparing a real guid (~100+) against the literal integer
11 — true for nearly every real edge regardless of the source op's actual
type. Two blocks earlier, in Phase 1 (line 522), the *correct* form of
this exact check is right there: `it->srcOp.ptr->type != OP_WEIGHT`. Looks
like a copy/paste that substituted the wrong comparison. The practical
effect: the reachability walk treats almost the entire graph as
"prunable," and the removal loop cascades deletion backward from the
graph's actual output (zero out-edges by definition) until almost nothing
is left.

Traced with reasonable confidence down to this specific line; not
confirmed further with a debugger (e.g. exactly why one `Reshape` node
specifically survived the cascade).

**Status:** Not fixed — the underlying bug in `ops.cc` is untouched. Worked
around by not calling it: folding isn't actually required for correctness
here, since `export_onnx()` already pulls real data from any genuine
`Weight` node, so an un-folded `Transpose`/etc. applied to a real weight
array is already a semantically correct ONNX node. The Python binding for
`preprocess_weights()` itself was added (commit `2d13b44`, mirrors the
existing `optimize()` binding — it wasn't exposed to Python at all before
this fork) since it's independently useful once the C++ bug is fixed, but
the bug itself remains.

---

## 4. `taso`: `graph.transpose()`'s `perm` attribute doesn't survive ONNX export

**Where:** `taso/src/core/transpose.cc` (encoding, `permutation_to_index()`
+ `Transpose::Transpose()` constructor) and
`taso/python/taso/_cython/core.pyx:662-668` (`get_operator_attr('perm')`
decoding).

**Symptom:** Every `Transpose` node produced by `graph.transpose(weight,
perm=(1,0), shuffle=True)` and then exported via `export_onnx()` came back
with an invalid `perm` attribute — e.g. `[0, 0]` (a repeated index, not a
valid permutation at all). onnxruntime rejects this outright:
`[TypeInferenceError] Attribute perm for Transpose has repeated value: 0`.

**What perm actually is:** a permutation of axis indices (output axis `k`
= input axis `perm[k]`) — for 2D, `perm=(1,0)` is just `w.T`. TASO doesn't
store it as a list; it packs the whole permutation into one integer
(`transpose.cc:18-29`):
```cpp
int idx = 0;
for (i in 0..N-1) idx = idx * N + perm[i];
```
For `perm=(1,0)`, `N=2`: encodes to `2`. The Cython getter decodes the same
way in reverse (`dims[i] = perIdx % N; perIdx //= N`, walking backward from
the last axis).

**Root cause: not conclusively identified.** Hand-tracing the encode
(`transpose.cc`) and decode (`core.pyx`) formulas in isolation for exactly
this case (`perm=(1,0)`, `N=2`) gives a mathematically correct round trip
back to `(1, 0)` — not the `(0, 0)` actually observed. Also traced
`get_operator_int_attr` → `find_op_or_fail` (`ops.cc:597-605`, a plain
linear search by `.guid`, looks correct) → `Transpose::get_int_parameter`'s
`PM_PERM` case (returns the stored `permIdx` directly, no extra logic) —
every piece inspected in isolation is internally consistent, yet the
actual runtime value is wrong. There is a second `get_or_create_transpose`
overload (`transpose.cc:65-79`) that deduplicates Transpose ops by an
`(input, perm, shuffle)` key while still handing back a freshly-incremented
guid regardless of dedup hits — looked at as a candidate mechanism, no
collision found for our specific case, but not ruled out for other
shapes/graphs. Confirming the actual root cause would need runtime
instrumentation (prints or a debugger against a real build), not just
static reading.

**Status:** Not fixed. Worked around in our own reconstruction script
(`NNs/reconstruct_optimized.py`) by transposing the real weight arrays
ourselves in numpy *before* handing TASO a literal `new_weight(...)` node,
bypassing `graph.transpose()` (and this whole broken attribute path)
entirely for weight-derived transposes. This only covers transposes whose
input is a leaf weight; a transpose applied to a non-weight (activation)
tensor would still hit this bug and has no workaround yet.

---

## 5. `taso`: `export_onnx()` emits an invalid ONNX op name for `Matmul`

**Where:** `taso/python/taso/__init__.py`, `export_onnx()` (originally
~line 893).

**Symptom:** Any graph containing a `Matmul` op, exported via
`export_onnx()` and loaded by onnxruntime, fails:
`INVALID_GRAPH ... Error No Op registered for Matmul with domain_version
of 13`.

**Root cause:** TASO's internal op-table name for this op type is
`"Matmul"` (`core.pyx:129`, `op_table[OP_MATMUL] = "Matmul"`), and
`export_onnx()` passes that string straight through as the ONNX node's
`op_type`:
```python
node = helper.make_node(mytype, inputs, outputs, ...)
```
ONNX's official spelling is `"MatMul"` (capital M in the middle) — a plain
case mismatch. Every other op name TASO uses (`Relu`, `Add`, `Reshape`,
`Concat`, ...) happens to already match ONNX's official spelling; `Matmul`
apparently doesn't and was never caught, presumably because TASO's own
native pipeline never round-trips a `Matmul`-containing graph through
`export_onnx()` → an external ONNX runtime the way this session's use case
does.

**Status:** Fixed in this fork — commit `1b7bcda`, "Fix export_onnx()
emitting invalid ONNX op name for Matmul." Added a small override table
(`_onnx_op_name_overrides = {"Matmul": "MatMul"}`) used only for the
`op_type` string passed to `helper.make_node`; `_add_node_attribute()` and
`operator_attrs` are still keyed on TASO's own name, unaffected. Not
audited for other possible name mismatches beyond this one.
