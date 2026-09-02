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

---

## 8. `taso`: ONNX importer's `Reshape` handler misses Constant-node shape args

**Where:** `taso/python/taso/__init__.py`, `_reshape()` (~line 459).

**Symptom:** Found while extending the pipeline to `resnet2b` (a real
residual network -- the first model in this session with a genuine
`torch.reshape`/`.view()` call, as opposed to the `Flatten`-based export
every earlier model happened to use, which needs no shape argument at
all). `_reshape()` only searched the ONNX graph's top-level `initializer`
list for its shape argument. ONNX also allows a constant tensor to come
from a `Constant` *node* elsewhere in the graph (its value living in that
node's own `value` attribute, not in `initializer`) -- the pattern
PyTorch's exporter actually used here. The lookup silently found nothing,
`shape` stayed an empty list, and `graph.reshape(inputs[0], ())` produced
a corrupt zero-dim tensor. That corruption didn't surface as an error
until several ops later, inside a `Gemm`'s `matmul()` call, which
segfaulted -- making the real fault site hard to find without adding
print-based instrumentation to trace it back.

**Root cause:** `_reshape()`'s only source of shape data was a linear
scan of `initializer`:
```python
for data in initializer:
    if data.name == op.input[1]:
        ...
```
When the shape argument is a `Constant` node's output instead, this scan
never matches anything, and the pre-existing bug is that there was no
fallback at all.

**Status:** Fixed -- commit `af3770a`, "Fix ONNX importer's Reshape handler
missing Constant-node shape args." `_constant()` (the handler for
`Constant` nodes) now additionally records each node's decoded value in a
module-level side-channel dict keyed by output tensor name (reset at the
start of each `load_onnx()` call); `_reshape()` falls back to that dict
when the `initializer` scan comes up empty.

---

## 9. `taso`: `export_onnx()`'s `Split` node is incompatible with opset 13

**Where:** `taso/python/taso/__init__.py`, `operator_attrs['Split'] =
['axis', 'split']` (~line 855).

**Symptom:** Found reconstructing a real `Split` node for the first time
this session (from a genuine TENSAT-selected parallel-conv-fusion rewrite
on a custom-trained `InceptionMNIST` model -- see `PROGRESS.md`). Loading
the exported ONNX in onnxruntime failed: `INVALID_GRAPH ... Unrecognized
attribute: split for operator Split`.

**Root cause:** `export_onnx()` always emits `Split`'s output sizes as a
node *attribute* (`operator_attrs['Split'] = ['axis', 'split']`,
`_add_node_attribute()` sets it via `helper.make_attribute('split',
val)`). That was ONNX's own `Split` spec through opset 12, but opset 13
moved split sizes to an optional *second input* tensor and dropped the
attribute form entirely -- onnxruntime (correctly) doesn't recognize a
`split` attribute at all once the model declares opset 13, which is what
every model in this project pins to (`onnx_model.opset_import[0].version
= 13`, done specifically because `helper.make_model()` otherwise stamps
whatever opset the installed `onnx` package currently defaults to, which
can be unreleased).

**Status:** Not fixed in `taso` itself -- `export_onnx()` doesn't inspect
the target opset at all, so a real fix would need it to either always
emit the opset-13 input form, or accept a target-opset parameter and pick
the right form. Worked around at the call site instead: any graph
containing a `Split` node is exported pinned to opset 11 (still fully
supported by current onnxruntime, and no other op this project uses needs
anything newer), rather than this project's usual opset 13. See
`NNs/reconstruct_inception_fused.py`.

---

## 10. `taso`: `export_onnx()` lists every initializer as a formal graph input too

**Where:** `taso/python/taso/__init__.py`, `export_onnx()` (~line 942).

**Symptom:** Every ONNX file this project has ever exported carries a
warning from onnxruntime that was never investigated (it loads and runs
fine regardless, so it was easy to dismiss): `Initializer X appears in
graph inputs and will not be treated as constant value/weight ... remove
it ... with the tool onnxruntime/tools/python/remove_initializer_from_
input.py`. It turned out to matter a lot once these files were fed to
`alpha-beta-CROWN`'s `auto_LiRPA` for actual bound propagation (rather
than just onnxruntime inference): `auto_LiRPA` built real `BoundBuffers`
nodes and a whole `Split`/`Squeeze`/`Unsqueeze`/`Concat` decomposition to
treat each of these "inputs" as a potentially-perturbable tensor, which
is unnecessary overhead at best and at worst tangles up its own bound-
shape bookkeeping (see the `RuntimeError: shape '[1, 1519]' is invalid
for input of size 12152` hit reconstructing `InceptionMNIST` -- this fix
didn't fully resolve that particular crash on its own, see bug #11, but
is a real, independent correctness/cleanliness issue worth fixing
regardless).

**Root cause:** every `Weight` tensor and every `Reshape`'s shape
constant gets added to *both* `graph_initializers` (with a real value)
and `graph_inputs` (as a formal graph input):
```python
if intype == 'Input' or intype == 'Weight':
    graph_inputs.append(helper.make_tensor_value_info(...))
if intype == 'Weight':
    graph_initializers.append(helper.make_tensor(...))
...
graph_inputs.append(helper.make_tensor_value_info('Reshape_attr{}'.format(op['guid']), ...))
graph_initializers.append(helper.make_tensor('Reshape_attr{}'.format(op['guid']), ...))
```
Per the ONNX spec a name present in both lists is merely "an optional
input with a default value" -- but nothing in `export_onnx()` filters
`graph_inputs` down to genuinely-external inputs before building the
graph, so `helper.make_graph()` gets every initializer listed twice.

**Status:** Fixed -- commit `e73ced7`, "Fix export_onnx() listing
initializers as formal graph inputs." Filters `graph_initializers`' names
out of `graph_inputs` right before `helper.make_graph()` -- the standard
fix onnxruntime's own warning points at.

---

## 11. `taso`-exported models are architecturally incompatible with CROWN-style verifiers when a rewrite fuses along the batch axis

**Where:** Not a single code location -- a structural property of any
graph containing a `Concat`/`Split` pair whose axis is 0 (the batch
axis), which is exactly what `tensat`'s `PRE_DEFINED_MULTI` parallel-
conv-fusion rule can produce (see `PROGRESS.md`'s `InceptionMNIST`
entry) and did produce in the one real nontrivial rewrite this project
extracted this session.

**Symptom:** Found trying to run alpha-beta-CROWN on the reconstructed
fused `InceptionMNIST` model. `auto_LiRPA`'s own bound-propagation code
explicitly refuses to backward-propagate through a `Concat` on the batch
axis:
```python
# auto_LiRPA/operators/slice_concat.py, BoundConcat.bound_backward
assert self.axis > 0
```
This isn't a shape bug to patch -- it's deliberate. `auto_LiRPA` (like
most CROWN-style verifiers) reserves axis 0 throughout its entire
architecture for batching *verification queries themselves* (multiple
images, multiple branch-and-bound sub-domains, multiple output-margin
specs), and none of that machinery expects the network's own forward
computation to also be using that axis for something else. Confirmed
this isn't even an `auto_LiRPA`-specific limitation: plain onnxruntime
inference on the same file at batch=2 fails outright too
(`Cannot split using values in 'split' attribute ... Sum of sizes in
'split' ... was 2` -- the fused conv's split is hardcoded to exactly 2
along axis 0, which only happens to work when the *real* batch is 1,
since the fusion trick itself doubles that axis internally).

**Status:** Not a bug in the traditional sense -- a genuine structural
fact about *this specific* TENSAT-selected rewrite, not about fusion via
`Concat`/`Split` in general. There is no ONNX-level or
`auto_LiRPA`-config-level workaround for a graph that already has an
axis-0 `Concat`: making the model batch-flexible elsewhere (`graph.
input`'s batch `dim_param`, the flatten `Reshape`'s shape) doesn't help,
since the conflict is that a *true* batch of anything other than exactly
1 collides with the fusion trick's *own* internal use of axis 0.

**Update, 2026-08-23 -- found and verified a second, ab-CROWN-compatible
fusion instead of trying to patch this one.** `tensat`'s relu-merge
multi-pattern rule (the one that produced this axis-0 Concat) has a
*second* variant using axis 1 (the channel axis) instead -- mathematically
the identical trick, just concatenating along a different axis, and
`auto_LiRPA`'s restriction is specifically `axis > 0`, so axis 1 is fine.
The extractor didn't pick this variant on its own -- confirmed (by
re-running extraction with `--favor_fusion` disabled entirely) that the
axis-0 variant wins on genuine cost-model merit even *without* any
discount, because it lets the stem's ReLU be reused instead of
recomputed; a first attempt at merely *not discounting* axis-0
(neutral, rather than favoring it) still produced the axis-0 extraction
for the same reason. `tensat/src/optimize.rs`'s `--favor_fusion` was
extended to actively *penalize*
(1000x) axis-0 `Concat`/`Split` while still discounting axis!=0 ones and
`Enlarge`, which reliably produces the axis-1 variant instead
(`NNs/reconstruct_inception_fused_v2.py`, `tensat` commit history --
`CostModel::get_self_cost`'s `axis0_concat_or_split` check).

This second fusion **is** verifiable by alpha-beta-CROWN (with the
branching-heuristic caveat in bug #12) -- verified numerically correct
(~1e-6) and batch-flexible, and produced real bounds: final verified
accuracy 10.0% (1/10 safe) vs. the unfused baseline's 20.0% (2/10 safe)
under the *same* branching method, a controlled, fusion-attributable
difference. See `PROGRESS.md`'s 2026-08-23 fusion-v2 entry for the full
writeup and `NNs/abcrown_out_inception_mnist_fused_v2.log` /
`NNs/abcrown_out_inception_mnist_unfused_randombranch.log` for the raw
results. The original axis-0 finding above remains accurate for that
*specific* extraction -- it just turned out not to be the only rewrite
available for this model.

---

## 12. alpha-beta-CROWN's `babsr`/`kfsb` branching heuristic doesn't support `Concat` layers

**Where:** `alpha-beta-CROWN/complete_verifier/heuristics/babsr.py`,
`get_babsr_biases()` (not this project's code -- part of the
`alpha-beta-CROWN` checkout itself).

**Symptom:** Found running the axis-1 (channel) fused `InceptionMNIST`
model (bug #11's update) through alpha-beta-CROWN with default settings.
Bound propagation itself succeeded completely -- real CROWN bounds
computed, 8/9 output-margin specs verified directly, branch-and-bound
even got as far as computing refined intermediate bounds for the one
remaining property -- then crashed choosing which neuron to split next:
```
File ".../heuristics/utils.py", line 67, in get_babsr_biases
    raise NotImplementedError(type(input_node))
NotImplementedError: <class 'auto_LiRPA.operators.slice_concat.BoundConcat'>
```
The default branching heuristic (`kfsb`, built on `babsr` scoring) tries
to compute a "bias" term for every layer with unstable neurons when
picking a branching candidate, and `get_babsr_biases` only has cases for
the usual handful of layer types (`BoundLinear`, `BoundConv`, etc.) --
`BoundConcat` was never one of them, presumably because a plain,
unfused, non-tensat-optimized network essentially never has a `Concat`
sitting in the middle of its unstable-neuron layers the way this
rewrite's structure does.

**Status:** Not fixed (would mean patching `alpha-beta-CROWN` itself, out
of scope). Worked around by setting `bab.branching.method: random`
instead of the default `kfsb` -- `random` doesn't need per-layer bias
scoring at all, so it sidesteps `BoundConcat` entirely. Real cost:
`random` is a much weaker heuristic than `kfsb`/`babsr` (no informed
choice of which neuron to split), so BaB is less effective per unit
time -- part of why the comparison in bug #11's update deliberately runs
*both* the fused and unfused models under `random`, to keep the
heuristic itself from confounding the fusion-vs-no-fusion comparison.

---

## 13. `taso`: `export_onnx()` exports asymmetric SAME-conv padding, disagreeing with `Conv2D`'s own (symmetric) padding semantics

**Where:** `taso`'s ONNX export path (`ts.export_onnx()`, not this
project's code) vs. `taso/src/core/conv2d.cc`'s `Conv2D::get_padding()`.

**Symptom:** Found while generalizing per-model reconstruction into
`NNs/reconstruct_generic.py` (this session's Phase 2) and regression-
testing it against `resnet2b` for the first time -- `mnist_cnn_a` and
`InceptionMNIST` had already passed the identical numeric check, but
`resnet2b`'s reconstruction was off by a large margin (max abs diff
1.34, vs. ~1e-6 elsewhere). `Conv2D::get_padding()` computes SAME
padding as `*padH = (totalPadH + 1) / 2;` -- one value, applied
*symmetrically* to both sides of the input (its own comment: "assert
same padding on both sides") -- but the ONNX `Conv` node
`ts.export_onnx()` emits instead uses an *asymmetric*, TF-style
floor/ceil split of the same total. These two computations only
disagree when the total required padding is odd (e.g. kernel=3,
stride=2, input%stride==0, giving totalPad=1) -- exactly `resnet2b`'s
stem conv and `layer1.0.conv1` (both kernel=3/stride=2 on an
even-sized input), and exactly why `mnist_cnn_a`/`InceptionMNIST` never
hit it (their SAME convs all landed on even totals, where floor/ceil
and symmetric-ceil happen to coincide). Confirmed directly: the
exported ONNX had `pads: [0, 0, 1, 1]` on those two nodes (0 before, 1
after -- both H and W) where the correct-per-TASO's-own-semantics value
is symmetric `[1, 1, 1, 1]`; patching just those two nodes to symmetric
padding took the reconstruction from 1.34 off to ~8e-7 (exact, modulo
float noise) against the real PyTorch reference output.

**Status:** Not fixed in `taso` itself (out of scope -- would mean
patching its C++ export path). Worked around in
`NNs/reconstruct_generic.py`'s `fix_same_padding_symmetric()`: after
`ts.export_onnx()`, walk every `Conv` node and replace any asymmetric
`pads` attribute with the symmetric max of its begin/end values per
dimension (a `VALID`-mode conv already has all-zero pads, so this is a
no-op there). Since this is a genuine `taso` export bug rather than
anything specific to one model, every future reconstruction this
project does -- including the many samples the structural-diversity-
vs-verifiability sweep (`PROGRESS.md`'s cost-function-design campaign)
will generate -- goes through this same generic script, so the fix
applies uniformly rather than needing to be rediscovered per model.

---

## 14. `tensat`'s axis-0-Concat safety check can't distinguish a weight-level concat from an activation-level one

**Where:** `tensat/src/optimize.rs`, `CostModel::get_self_cost`'s
`axis0_concat_or_split`/`is_favored_fusion_op` checks (added for bug
#11's fix, i.e. predates this session's other changes).

**Symptom:** Found while investigating why InceptionMNIST's `--favor_
fusion_strength` sweep (`PROGRESS.md`'s 2026-08-24 campaign, Phase 4)
behaved unintuitively. The conv-fusion multi-pattern rule's `Concat`
legitimately has `axis == 0` too -- but that's the *weight* tensor's own
output-channel axis (concatenating two conv kernels along their shared
dim 0), never a real activation's batch axis -- yet
`axis0_concat_or_split` flags any `Concat`/`Split` with `axis == 0` as
unsafe purely by the numeric value, with no way to tell "this operand is
weight-derived" from "this operand is an activation." So the
conv-fusion rule's safe weight-level `Concat` gets the same 1000x
penalty as the genuinely unsafe batch-axis relu-merge `Concat` bug #11
was written to suppress.

**Status:** Not fixed -- confirmed harmless rather than worth chasing.
A weight-only `Concat`/`Split` (both operands trace only to `Weight`
leaves) is numerically folded away entirely during Python reconstruction
(`reconstruct_generic.py`'s dual-path dispatch: fold in numpy if
`all(g in weight_arrays for g in src_guids)`, else emit a real graph op)
-- it never survives to become a real ONNX `Concat`/`Split` node, so it
can never reach `auto_LiRPA`'s axis restriction either way. The
mislabeling only makes the cost model slightly less eager to select the
conv-fusion pattern during extraction (an efficiency question, not a
safety one) -- confirmed via `tensat`'s own `weight_names` provenance
(bug -- see Phase 1's `ValTnsr.weight_names` field): a `Concat` at
`axis=0` with a fully weight-derived `weight_names` set (e.g.
`["branchA.weight", "branchB.weight"]`) is exactly the safe case this
note describes. Worth fixing properly (check `weight_names`/`all_
weights` instead of axis value alone) if a future cost function needs
to favor the conv-fusion rule specifically and precisely, but not
blocking for anything done so far.

---

## TASO ONNX importer: `MatMul` (capital) silently skipped (2026-08-30)

`taso/python/taso/__init__.py` registered the ONNX->TASO handler under the key
`xf_operators['Matmul']` (lowercase 'm'), but ONNX's standard op name is **`MatMul`**
(capital M). The load loop dispatches on `op.op_type` verbatim (`if op.op_type in
xf_operators`), so every `MatMul` node fell through to "Found unsupported ONNX operator:
MatMul (Skipped)". In a pure FC/MatMul graph (e.g. the VNN-COMP **tll** net) this skips
ALL compute: downstream Add/Relu then can't find their input tensors and are skipped too,
leaving a degenerate graph of just inputs + weights (the long-standing "tll -> 1 Input +
29 Weight, zero compute" mystery). Fix: `xf_operators['MatMul'] = _matmul` alias. This
was THE barrier to ingesting MatMul-based FC nets; with it, tll ingests and the
[[tll semantic lift]] round-trips.

Residual (separate, pre-known): even after ingestion, taso's SGEMM COST MEASUREMENT
(`src/cudnn/matmul_kernel.cu:165`, `cublasSgemm`) aborts with "parameter number 10 had an
illegal value" on small-N matmuls (tll's width-1/2/4 output layers). The vector trick
(pad widths >= 8) sidesteps it; a real fix would guard/clamp the leading dims (or skip
cost measurement) for small N, since the measured runtime is irrelevant to the tensat
pipeline. Blocks mechanical import of scalar-output FC nets until fixed.

## TASO ONNX importer: `Sigmoid`/`Tanh` unregistered -> skipped (2026-09-01)

**Same bug class as the `MatMul` casing above, and the ACTUAL ffnnSIGMOID barrier.**
`xf_operators` had no `Sigmoid` (or `Tanh`) entry, so `load_onnx` skipped every Sigmoid
node -- degenerating any sigmoid MLP exactly like the MatMul skip degenerated tll.
Re-scoping the ffnnSIGMOID (6x200 MNIST) retry showed its ops are
`{Constant, Sub, Div, Flatten, Gemm x7, Sigmoid x6}` -- and Sub/Div/Flatten/Gemm are ALL
already registered, so the **MatMul-casing fix is irrelevant here** (ffnnSIGMOID uses
`Gemm`, not bare `MatMul`); `Sigmoid` was the sole missing op. Its Gemm widths are >= 10,
so the small-N SGEMM abort above does not apply either. Fix: added `_sigmoid`/`_tanh`
builders (mirror `_relu`, calling `graph.sigmoid`/`graph.tanh` which already exist in
`core.pyx`) + `xf_operators['Sigmoid'|'Tanh']`. Pairs with the standalone Sigmoid/Tanh
arms added to `NNs/reconstruct_generic.py` the same day, so a sigmoid/tanh MLP now has a
path through BOTH ingestion and reconstruct.

**NOT YET RUN.** The taso python + onnx combination is unavailable in-repo (taso is
compiled for python 3.14 = the container only; the container's onnx is currently broken;
the only working onnx is in the 3.10 `taso_py` host env, which can't load the 3.14 taso
`.so`). Verified by inspection (mirrors the known-correct `_relu`) and `py_compile`;
end-to-end ingestion must be run in a taso+onnx env (see the memory note
`taso-python-unrunnable-in-repo-envs`). Expected result once run: ffnnSIGMOID ingests to a
non-degenerate `Gemm x7 / Sigmoid x6` graph.

## `tensat`: `-m verify` axiom set (`rules()`) bit-rotted against the `Mdl` language for ~6 years (2026-08-31)

`rewrites.rs::rules()` is the axiom set the GPU-free axiom verifier (`-m verify` /
`prove_taso_rules`) saturates with. It was written in 2020-06 against the op arities of the
day (2-arg `matmul`, 1-arg `transpose`, 3-arg `concat`, 6-arg params-first pool). ~6 weeks
later the `Mdl` `define_language!` grew params to represent real models for the OPTIMIZER
(f2109cc 2020-07-16 `matmul` +activation → `[Id;3]`; 86a2617 2020-07-31 `transpose`
+perm/shuffle → `[Id;3]`; `concat` +ndim → `[Id;4]`; pool → 7-arg input-first; `enlarge` →
2-arg ref-based). `rules()` was never updated to match.

Because these are `rw!(...)` macros parsed EAGERLY when the `Vec` is constructed, a single
stale pattern (e.g. `(transpose (transpose ?x))` against a `[Id;3]` `Transpose`) PANICS at
pattern-parse — so `-m verify` could not run on ANY rule set, not just ones touching those
ops. It went undetected because (a) `rules()` is verify-only (the optimizer uses
`rules_from_str`, loaded fresh from files, always current), and (b) `prove_taso_rules` is
disabled by default (README: "uncomment it in main.rs"). Classic dead-code rot: two rule
representations, one exercised and maintained, one dormant and silently broken by a language
change. The stock `taso_rules.txt` is stale the same way and won't parse under current tensat.

**Fix (this session):** migrated `rules()` to current arities. Because `verify()` is pure-egg
(`Runner::<Mdl,(),()>` — no shape/`make()`), an axiom only needs to parse and be a TRUE
universal identity, which dictates sound migration vs. sound drop: `matmul` → literal acti 0;
`concat` → free rank var `?n`; pool → input-first + acti; `transpose` distribution → free perm
`?p`/shuffle `?s`; plus the new `matmul` relu-unfold. Five families that can't be stated
soundly in pure-egg were DROPPED (transpose-is-its-own-inverse / matmul-and-transpose /
concat-transpose are 2D/involution-specific; split-definition is arity-broken + conditional;
enlarge-convolution-kernel is a semantic mismatch), plus two inverse directions with an
unbindable concat rank on the RHS. **Guard added** (`NNs/tests/run_tests.sh` Test 4): rules()
must construct without panic (catches arity drift), 5 known-false negative canaries must all be
REJECTED (catches any future unsound axiom), and the min/max family must still prove — the exact
regression that would have caught this in 2020 had the verifier been wired into CI.

## `taso`: `export_op` has no case for the added EW_SUB/MAX/MIN ops → crashes on export (2026-08-31)

We extended TASO's generator + core enums with the elementwise `EW_SUB`/`EW_MAX`/`EW_MIN`
ops (types 26/27/28) to produce the PWL/min-max rewrite family. `Graph::export_op`
(`src/core/ops.cc`) was never given cases for them: it routes ops through a big shared
`case` block (`OP_EW_ADD`/`OP_EW_MUL`/`OP_RELU`/…, all "no special params → write op+inputs")
and a `default:` at line ~1031 that prints `op.ptr->type` and `assert(false)`. So the moment
an extracted graph contains an `ewmin`/`ewmax`/`ewsub` node, `--export_model` aborts with
`Assertion 'false' failed`, printing the bare op-type int (e.g. `28`) first.

This stayed hidden until now because the earlier min/max experiments extracted *pure-max*
maxout forms whose exported ops happened to be covered; the `1097` core's min/max
**distributivity** rules (`max(x,min(y,z)) = min(max,max)`) introduce `ewmin` nodes into
maxout's saturated e-graph, so the diverse/verif_cost extraction produced a form containing an
`ewmin` and export died on it (op type 28). The CPU-taso saturation, extraction, and cost were
all fine — only the *serializer* was missing the ops.

**Fix:** add `OP_EW_SUB`/`OP_EW_MAX`/`OP_EW_MIN` to the shared elementwise `case` block in
`export_op` (they serialize identically to `EW_ADD` — op type + inputs, no params). After the
fix all forms export cleanly.

Build note: `taso/build` (the CPU, `USE_CUDA=OFF` lib that tensat loads for GPU-free
saturation/export) was originally configured with a host cmake (`toolchain-tensat/miniconda3`,
now broken — its `share/cmake-*` modules dir is missing), so `make` fails with "Could not find
CMAKE_ROOT". Rebuilt the one object directly instead: recompile `ops.cc` with the flags from
`CMakeFiles/taso_runtime.dir/flags.make` and relink via `CMakeFiles/taso_runtime.dir/link.txt`
(both use `/usr/bin/c++`). Separately, `taso/build_gpu` still fails `Cuda failure 35` at
`ops_cudnn.cu:24` (cuDNN init) even though the bare CUDA runtime now works — so the CPU lib is
the right one for the shape-only saturation/export path anyway.

## `taso`: python C-ext `core.*.so` has DT_RPATH hardcoded to `build_gpu`, so reconstruct always loads the GPU libtaso and dies on cuDNN init (2026-08-31)

**Symptom:** Every attempt to *reconstruct* a tensat-extracted `.model` back to ONNX
(`reconstruct_generic.py`, which does `import taso`) aborted at import with
`Cuda failure: 35 … taso/src/cudnn/ops_cudnn.cu:24 … Aborting`, **even on an L40S GPU node**
and even with `LD_LIBRARY_PATH=taso/build` (the CPU, cuda-free lib) explicitly set. The
verif_cost/diverse *extraction* stages (pure tensat, no taso-python) were fine; only the
reconstruct step crashed, silently zeroing out the downstream alpha-CROWN bound.

**Cause:** the compiled extension `taso/python/taso/core.cpython-314-…so` carries a
`DT_RPATH` (not `DT_RUNPATH`) of
`/opt/conda/lib:<repo>/taso/build_gpu`. **`DT_RPATH` is searched *before* `LD_LIBRARY_PATH`**,
so the loader always resolves `libtaso_runtime.so` to the *GPU* build, which `NEEDED`s
`libcudnn.so.9`/`libcudart.so.12` and runs cuDNN handle-init at load — failing with Cuda-35
(the persistent `build_gpu` cuDNN bug) the instant `import taso` runs. `LD_LIBRARY_PATH` can
never win against `DT_RPATH`, which is why every "run it on a GPU node / set LD_LIBRARY_PATH"
attempt failed. Reconstruction needs **zero** GPU compute (it only rebuilds the graph and
serializes ONNX), so loading the GPU lib is both unnecessary and fatal.

**Fix (applied):** the CPU lib name (`taso/build`) is exactly 4 chars shorter than
`taso/build_gpu`, so patch the RPATH string *in place, same length* — no `patchelf`
(unavailable here) and no relink needed:
```python
b = bytearray(open(SO,'rb').read())
b = b.replace(b"/taso/build_gpu\x00", b"/taso/build\x00\x00\x00\x00\x00")  # 16->16 bytes; 1st NUL terminates
open(SO,'wb').write(b)   # RPATH becomes .../taso/build ; verify: readelf -d SO | grep RPATH
```
The CPU `libtaso_runtime.so` has no cuda `NEEDED`s, so `import taso` then succeeds on any node
(no GPU), and reconstruct + bound run CPU-only. Validated: import OK, ONNX exported, numeric
gate ~1e-6, bounds reproduce recorded baselines to 4 dp (rules out an ABI mismatch from the
Aug-22 ext vs the Aug-31 CPU-lib rebuild). Backup of the original ext: `core.*.so.gpubak`.

**Durable fix (not yet done):** `taso/python/setup.py` links the ext against `build_gpu`; point
it at `taso/build` (or emit `DT_RUNPATH` so `LD_LIBRARY_PATH` can override). **Any rebuild of the
cython ext silently restores RPATH→`build_gpu` and Cuda-35 returns with no obvious cause.**
