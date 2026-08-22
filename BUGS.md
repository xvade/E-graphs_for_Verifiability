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

**Related, also fixed (2026-08-22, extending to `mnist_cnn_a`):**
`_conv2d()` (same file, ~line 203) had the identical pattern — it also
fetched but never used Conv's optional bias input (`inputs[2]`). Unlike
Gemm's bias (shape `[out_features]`, already aligned with the trailing axis
of a `[..., out_features]` matmul result), a Conv bias is `[C]` against an
`[N,C,H,W]` conv output — TASO's `add()` only does NumPy-style
trailing-dim broadcast, so the fix reshapes the bias to `[1,C,1,1]` first
(`graph.reshape(inputs[2], (1, num_channels, 1, 1))`) before adding.

**Status:** Fixed in this fork — `Gemm` in commit `bd8ba5d` ("Fix ONNX Gemm
importer silently dropping the bias input"), `Conv` in commit `fb0b3db`
("Fix Graph::get_operator_int_attr silently returning garbage in Release
builds", which bundles this alongside bug #6 below). Both now call
`graph.add()` with a correctly-shaped bias (TASO's `Model::broadcastable`,
`element.cc:19`, handles the rest).

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

**Root cause: found 2026-08-22, see bug #6.** Originally reported here as
"not conclusively identified" after hand-tracing the encode/decode formulas
in isolation found no flaw. The actual bug turned out to be one level up
the call stack, in `Graph::get_operator_int_attr` itself (`ops.cc:669-675`)
— see bug #6 for the full explanation. `PM_PERM`'s value is read through
exactly that same broken path, so it silently returns uninitialized stack
garbage in a Release build, same as this section originally observed.
Fixed by the same commit as bug #6.

**Status:** Fixed as of commit `fb0b3db` (see bug #6) — `graph.transpose()`
now round-trips correctly through `export_onnx()` in a Release build.
`NNs/reconstruct_optimized.py` still transposes weight-derived arrays
directly in numpy rather than calling `graph.transpose()`, since that
workaround was never actually wrong, just no longer necessary; not
reverted, to avoid re-touching working code without a reason.

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

---

## 6. `taso`: `Graph::get_operator_int_attr` returns garbage in Release builds

**Where:** `taso/src/core/ops.cc:669-675`.

**Symptom:** Found 2026-08-22 while extending the pipeline to
`mnist_cnn_a` (adds Conv2D). `export_onnx()`'s attribute export for any
`Conv` node came back with `strides=[0,0]`, `kernel_shape=[0,0]`,
`pads=[0,0,0,0]` — every numeric attribute zeroed, even though the same
graph's `strideH`/`strideW`/etc. were provably correct moments earlier
(the numbered-format export from `Graph::export_to_file`, which reads
these fields directly off the `Conv2D` object rather than through this
path, wrote the right values). This is also, with high confidence, the
real root cause of bug #4 above (`Transpose`'s `perm` attribute) — same
call path, same failure mode, first misdiagnosed as a `Transpose`-specific
issue before this session's Conv2D work exposed the actual mechanism.

**Root cause:**
```cpp
int Graph::get_operator_int_attr(size_t guid, PMParameter attr)
{
  Op op = find_op_or_fail(guid);
  int ret;
  assert(op.ptr->get_int_parameter(attr, &ret));
  return ret;
}
```
`get_int_parameter(attr, &ret)` is the *only* thing that populates `ret` —
it's not just a boolean check, it has a real side effect. Wrapping it in
`assert(...)` means that in any build with `NDEBUG` defined (i.e. any
CMake `Release` build — exactly what `taso/build_gpu/config.cmake` sets,
since that's the config used for real GPU/cuDNN measurement), the standard
library's `assert` macro expands to nothing and **never evaluates its
argument at all**. The call that was supposed to fill in `ret` simply
never happens, and the function returns whatever garbage was already on
the stack at that address — which happened to read as all-zeros here,
producing exactly the "zeroed attribute" and "corrupted perm" symptoms
both bugs showed. A `Debug` build (asserts compiled in) would never have
shown this at all, which is presumably how it went unnoticed: TASO's own
benchmark suite and this project's earlier CPU-only work never exercised
this exact path under a Release build with attribute export in the loop.

**Scope beyond what's fixed here:** the identical anti-pattern —
`assert(some_call_with_side_effects(...))` — recurs many times in
`taso/src/core/substitution.cc` (e.g. lines 261-264, 272-273, 288-289,
320, 328-330, 584, 840, 932-936, 1253-1257, 1298-1299), which implements
the rewrite-rule matching/substitution engine used during equality
saturation itself. None of those are touched by this fix. Whether any of
them cause incorrect rewrite-rule matching in a Release build is
**unaudited** — flagging as a real open risk, not confirmed to cause
observable incorrect behavior (the `mnist_cnn_a` optimize run in this
session completed and the final result verified numerically correct
end-to-end, but that graph also didn't happen to have any rewrite fire, so
it's not evidence either way for whether `substitution.cc`'s copies of
this pattern are safe under Release builds).

**Status:** Fixed the one call site that blocked this session's work —
commit `fb0b3db`, splitting the call and the assert:
```cpp
bool found = op.ptr->get_int_parameter(attr, &ret);
assert(found);
```
`substitution.cc`'s occurrences are not fixed and not audited.

---

## 7. `taso`: `export_onnx()` never emits a node for a fused Conv/Pool activation

**Where:** `taso/python/taso/__init__.py`, `export_onnx()` (~line 856).

**Symptom:** Not actually triggered in this session (the one `tensat`
optimize run done so far didn't fuse an activation into a `Conv2D`'s
`activation` field), but confirmed by code inspection while writing
`NNs/reconstruct_optimized.py`'s `Conv` branch, since that script has to
call `graph.conv2d(..., activation=...)` directly when reconstructing an
optimized graph.

**Root cause:** `TASO`'s `Conv2D`/pooling ops carry an `activation` field
(`AC_MODE_NONE`/`SIGMOID`/`RELU`/`TANH`) that can be fused directly into
the op (no separate `Relu` node needed at the TASO-graph level) — this is
a real, intentional fusion TASO's own rewrite rules can produce. But
`export_onnx()`'s main loop (`__init__.py:872-908`) only ever emits one
ONNX node per TASO op, built from that op's own type and attributes; there
is no code path that checks a `Conv`/`MaxPool`/`AveragePool` op's
activation field and synthesizes a following `Relu`/`Sigmoid`/`Tanh` ONNX
node for it. A `Conv2D` op with a fused activation, exported through this
function, would silently produce an ONNX graph missing that activation
entirely, with no error.

**Status:** Not fixed (would require nontrivial graph surgery in the
exporter — synthesizing a new node not present in `get_operator_list()`
and rewiring the real consumer's input name to point to it). Worked
around in `NNs/reconstruct_optimized.py`: when reconstructing a `Conv`
node, always construct it with `activation="NONE"` and apply any actual
fused activation as its own explicit `graph.relu()`/`sigmoid()`/`tanh()`
call afterward, so `export_onnx()` never sees a Conv op with a non-`NONE`
activation to begin with. This sidesteps the bug rather than fixing it,
and only covers this project's own reconstruction path — any other code
calling `export_onnx()` on a graph with a genuinely-fused Conv/Pool
activation (e.g. one straight out of `tensat`'s optimizer, before this
project's reconstruction step) would still hit it.
