# Min/Max reassociation vs verifiability — light-version findings

**Question (light version of "step 2").** Hand-author a small number of min/max
*reassociation* rules and test whether they change verifiability. This targets the
one rewrite family that can escape the project's "semantics-preserving rewrites are
verifiability-neutral" wall: a piecewise-linear function has *many* ReLU
decompositions, and re-associating a min/max reduction tree changes the ReLU
topology (hence the relaxation the verifier branches on) **while preserving the
function and even the total ReLU count.**

The rule tested: `max(u,v) = u + relu(v-u)`, `min(u,v) = u - relu(u-v)`, applied to
reduce N affine functions two ways —
- **chain** (running accumulator, depth N-1)
- **balanced** (binary tree, depth ceil(log2 N))

Both realize the *identical* function with the *identical* number of ReLUs (N-1);
only the association (topology) differs. This isolates reassociation as the single
variable. Verifiability is measured as the alpha-CROWN (`CROWN-Optimized`) certified
upper bound on the scalar output over an input box — the bounding ab-CROWN actually
uses — plus the count of *unstable* ReLUs (pre-activation straddling 0), which is the
driver of both bound looseness and branch-and-bound size.

Script: `maxtree_bounds.py` (this dir). Runs on GPU via the alpha-beta-CROWN venv +
auto_LiRPA. Distributional design (20 random nets/config, untuned properties) — no
threshold was tuned to manufacture a separation.

## Result 1 — reassociation is NOT verifiability-neutral (pure max-reduction)

N=16, d=8, eps=0.5, 20 reps, alpha-CROWN:

| metric | chain | balanced |
|---|---|---|
| tighter certified bound | **17/20** | 3/20 |
| mean certified upper bound | **12.15** | 13.04 |
| mean gap to true max | **4.13** | 5.02 |
| mean unstable ReLUs (of 15) | **8.35** | 12.70 |

- Mean bound advantage of chain: **0.89** (lower = tighter = more verifiable).
- **The effect is not an alpha-optimization artifact.** Vanilla CROWN at the same
  N=16x20 also shows chain tighter (**14/20**, mean delta 0.68, unstable ReLUs
  9.45 vs 12.70) — the same direction and mechanism as alpha-CROWN. (A small
  N=8/eps=0.3 3-rep CPU probe gave *byte-identical* bounds, but that is a
  small-perturbation regime where too few ReLUs are unstable for topology to
  matter; it does not generalize — corrected after the N=16x20 gating run.)

So: same function, same ReLU count, but reassociation moves the certified bound
under standard CROWN bounding. **The effect the whole project needed — a
semantics-preserving rewrite that changes verifiability — exists.**

## Mechanism — confirmed: reassociation changes ReLU *stability*

Chain has far fewer unstable ReLUs (8.35 vs 12.70), and in *every* rep individually.
The running max dominates later candidates, so each later `relu(candidate - running_max)`
has a pre-activation that is usually <= 0 -> that ReLU is **stable/inactive** -> exact
(tight) relaxation. The balanced tree combines "fresh" candidates whose difference
straddles 0 -> more **unstable** ReLUs -> looser bound. The lever is stability, not
count.

Direction note: this is the **opposite** of the pre-registered prediction (balanced >=
chain). The prediction was made first precisely so the reversal counts as a finding.
The actionable design rule flips to: **chain-ify the reduction** (and, conjectured,
order it so likely-large terms reduce first — untested extension).

## Result 2 — the effect is structure-dependent (weak for tll-shaped min-of-max)

Two-level lattice `min_g max_k f_{g,k}` (G=4, K=4 = 16 leaves), same settings:

| metric | chain | balanced |
|---|---|---|
| tighter certified bound | 8/20 | 11/20 |
| mean unstable ReLUs (of 15) | 11.85 | 12.70 |

Essentially a **wash** (mean bound delta -0.04). Consistent with the mechanism, not a
contradiction: the stability gain comes from *long* running-max chains, but a
two-level lattice caps reduction depth at K-1=3 per level and the outer *min* dilutes
the gain. **Reassociation strongly moves verifiability for deep max-reductions, but
barely for the shallow-grouped min-of-max that tll is.**

## Why the real tll benchmark was not lifted

The real `tllBench_N16` ONNX is a **deep sequential chain of MatMul->Relu->MatMul
"bank" blocks** (2->16->256->512->128->...->1, 8 ReLUs total) with the min/max lattice
semantics **baked into the weight matrices** — not a graph-level tree that can be
rebalanced. A clean semantics-preserving reassociation would require reverse-
engineering the encoded lattice, and Result 2 predicts little payoff. Per the
project's established theme, this is the same reason TASO can't ingest it and
sequential nets don't rewrite. The controlled hand-built distribution above is the
deliverable — a distributional result is stronger than one real instance anyway.

## Bottom line for the light->heavy gate

Light version **succeeds** on its actual criterion: a semantics-preserving rewrite
(min/max reassociation) demonstrably moves the certified bound under the optimized
bounding ab-CROWN uses, with a confirmed mechanism (ReLU stability). Caveat carried
forward: the gain is large only for deep max-reductions, and the min/max family is
*not* expressible in TASO's op set — so the heavy step (rerun TASO generation without
the speed bias) is a **separate rewrite family** (broadened tensor-algebra corpus on
Conv/Matmul models), not a scale-up of these rules.

## Gating checks

- **Budget robustness (rules out early-stop artifact):** at a forced 200-iter
  alpha-CROWN budget (patience 200), chain tighter is **17/20** — identical to the
  default-budget 17/20 (mean delta 0.97, unstable 8.15 vs 12.70). The effect is not
  an artifact of the optimizer stopping early. `budget_max_N16_it200.log`.
- **Vanilla-at-scale:** chain tighter **14/20** under vanilla CROWN (above) — effect
  is not alpha-specific. `vanilla_max_N16.log`.
- **Operating-point sweep (eps in {0.1,1.0} x N in {8,32}), alpha-CROWN.** The
  *direction* is robust but the *magnitude* is regime-dependent. In EVERY point
  mean(ub_bal - ub_chain) >= 0 (chain tighter-or-equal on average) and chain has <=
  unstable ReLUs; chain-wins >= balanced-wins everywhere:
    - N8 eps0.1: 2 chain / 0 balanced wins (18 ties), mean delta +0.002, unstable 0.6 vs 1.0
    - N32 eps0.1: 4 chain / 1 balanced (15 ties), mean delta +0.019, unstable 2.15 vs 5.20
    - N16 eps0.5: 17 chain / 3 balanced, mean delta +0.97, unstable 8.15 vs 12.70 (headline)
    - N8 eps1.0: 10 chain / 10 balanced, mean delta +0.225 (chain ahead on average)
    - N32 eps1.0: 19 chain / 1 balanced, mean delta +4.51, unstable 24.2 vs 30.35
  Reading: the effect scales with chain DEPTH (N). At N=8 it is weak (few relus /
  shallow); at N=16 decisive (17/20); at N=32 very strong (19/20 even at eps=1.0,
  mean +4.51). Magnitude also grows with eps until relus saturate. "Chain-ify" never
  hurts on average, and helps more the deeper the reduction.
- **G=2,K=8 lattice (mechanism confirmation).** The G4K4 lattice was a wash (8/20,
  mean -0.04). With LONGER within-group chains (K=8 vs K=4) the effect RETURNS:
  chain 14/20, mean +0.30, unstable 10.30 vs 12.70. This confirms Result 2's depth
  explanation is not post-hoc: the lattice effect is diluted only because shallow
  groups cap chain depth; lengthen the chains and it comes back.

## Extending TASO's generator with min/max/sub (heavy follow-on)

**Result: TASO's generator CAN produce the verifiability-relevant PWL rewrite family
once given min/max/subtraction — the op-set gap was the only barrier.**

- min/max/sub are already in TASO's core enum (`ops.h`, with ONNX refs). The generator
  (`src/generator/generator.cc`) is a standalone g++ tool (no cuDNN/GPU; its own
  random-numeric equivalence testing). Adding them = ~4 edits to `ElementTemp` (sub as
  ordered pairs since non-commutative; max/min commutative) + recovering a minimal
  `xflow/ops.h` (the original, git 34d0138, pulled in cuDNN/TensorRT/MKL; the generator
  needs only the enums + SplitInfo). Build recipe: `src/generator/compile_pwl.sh`
  (regenerate protobuf against the container's protoc, then g++). `#ifdef PWL_FOCUS`
  trims to the PWL op set.
- **Depth-3 focused run generated the target rules** (`pwl_generator_transfers.txt`,
  `pwl_graph_subst.pb`): min/max **re-association** `max(max(a,b),max(a,c))=max(a,max(b,c))`,
  the **bridge** `max(a,b)=(a+b)-min(a,b)`, and relu<->max/sub identities
  (`max(x,relu(x))=relu(x)`; `(x-y)-relu(x-y)`=min-form). These are exactly the family
  the light PoC showed moves verifiability, and were previously unrepresentable.

**Depth question:** the limit is one line (`if (depth >= GEN_MAX_DEPTH) return;`, now a
compile define). Depth 10 is intractable: the generator materializes every enumerated
graph in an in-memory hashmap, so cost is ~b^depth (b in the hundreds) in BOTH time and
RAM -- TASO's own published run was ~4 ops/hours; depth 10 is ~10^20 graphs, impossible.
We don't need it: the PWL reassociation rules are depth<=3. **Depth-4 pricing
(measured, focused op set):** ~12 GB RAM after 42 s and still climbing (98% CPU),
vs depth-3 = trivial (seconds, tiny) -- a ~1000x blowup from one level. Depth 4 is
already at the edge for the trimmed 9-op set; the full op set is far worse; depth 5+
is intractable and depth 10 impossible. Killed before it exhausted the shared node.

**tensat integration -- DONE (end-to-end).** Added Ewmax/Ewmin/Ewsub to the egg
language (model.rs), make + cost (TASO element(); runtime-sound -- cudnn opTensor
MAX/MIN, custom-kernel SUB), graph builders + parse_model ingestion (input.rs/parse.rs),
and CheckApply node processing (rewrites.rs). ONNX reconstruction lowers min/max to the
relu form (`max(a,b)=a+relu(b-a)`, `min(a,b)=a-relu(a-b)`, Sub native) -- NOT native
ONNX Min/Max, so ab-CROWN sees the ReLU topology the PoC measured. Verified end-to-end:
tensat ingests a max-tree TASO model, saturates with a min/max reassociation rule, and
extracts (no crash); reconstruction emits exactly the relu decomposition (confirmed by
ONNX op inspection). Fixed several latent tensat bugs along the way (empty-param model
parse, trailing-newline rule parse, CheckApply todo!()). Depth-4 pricing on the 1.5TB
node: runs (not OOM) but 174M+ graphs / ~52GB / 19+ min and climbing -- decisively
confirms depth is super-exponential; depth 3 is the operating point.

**Remaining (next block):** the pb->egg rule converter (note: the generator's xflow
shim enum has EW_MAX=27 while TASO/tensat use 33 -- the converter must map xflow-ints
to op names), a realistic single-input min/max model for a full numeric+verification
loop (the synthetic 3-input test hit a TASO multi-input export merge), and Z3
verification (z3-solver + real ITE semantics for relu/max/min/sub -- random-numeric
testing can pass false PWL equivalences, the bidir-soundness class).
