# Pipeline tests

Run inside the container (no pytest dependency; plain assert harness):
```
apptainer exec --no-mount bind-paths tensat.sif bash NNs/tests/run_tests.sh
```
Exits nonzero if any test fails. (`--no-mount bind-paths` disables the site
apptainer.conf bind mounts — e.g. `/var/run/slurm` — that otherwise abort
container creation on non-slurm nodes. Drop it if your node has those paths.)

Current status: **40 assertions, all passing** (tests 1–12).

## Tests
1. **Regression -- non-clean drop.** pb2egg on the original `taso/graph_subst.pb` must drop
   ZERO non-clean ops and emit conv2d/concat rules. The pre-fix clean-only pb2egg dropped
   all 72 non-clean rules (conv2d=0, concat=0) -- this test fails on that version and
   passes on the full-op pb2egg (116 rules, 0 non-clean dropped).
2. **Parse-validity.** Every emitted rule must parse in current tensat (`-m parse_check`,
   0 FAIL). Catches op arity/child-order drift forever -- the authoritative check, since
   the Mdl comments and converted_full660 are both stale.
3. **Reproduction (coverage).** Full-op families (conv/concat/matmul) recovered from the
   original pb with expected counts. NOTE: exact byte/set reproduction of the hand-committed
   `taso_rules.txt` is NOT feasible -- the git-tracked `graph_subst.pb` is a different,
   smaller corpus (116 rules) and taso_rules.txt is in a stale egg format (5-param poolmax,
   3-arg enlarge, 2-arg matmul). So this pins to the available original pb.

4. **Axiom verifier soundness + liveness.** `-m verify`'s `rules()` axiom set
   must (a) construct without panic (every axiom parses at the current `Mdl`
   arity — stale arities panic), (b) reject all 5 negative canaries (no axiom
   proves a known-false rewrite — soundness), and (c) still prove the 8
   representative min/max rules (liveness). Guards the class of bug that
   dead-code'd `rules()` for years.
5. **prededup alpha-equivalence.** `prededup.canon` must collapse input-renamed
   copies of the *same* rewrite to one representative, while keeping a
   commutativity swap and its identity *distinct* (they canonicalize
   differently). Regression guard: an AC-aware `canon` would wrongly drop the
   commutativity rules the lattice needs.
6. **sexpr_to_functional round-trip.** S-expr `(op a b)=>(op b a)` converts to
   the whitespace-free functional `op(a,b)==op(b,a)` that `-m verify` consumes,
   and a bare-atom-sided rule is dropped (a bare-atom side would merge with the
   next rule's op under the grammar's lack of a whitespace rule).
7. **structural_signature parse + features.** The 4-line-per-node `.model`
   parser recovers node count, the longest-dependency-chain depth, and every
   Concat/Split axis (axis 0 vs >0 — the project's most verifiability-relevant
   structural signal, BUGS.md #11/#12).

8. **redundancy mode.** `tensat -m redundancy` must recognize two PWL rules as
   groundable and prune a renamed duplicate (add-commutativity), keeping exactly
   one representative. Exercises the pruner on the prebuilt binary.

9. **transpose emission — gated by default (tier-2, PROBLEMATIC.md #8).** transpose
   is apply-UNsafe (tensat can't build it during saturation), so pb2egg drops it by
   default (0 emitted, 20 counted unapplicable); `--emit-unapplicable` emits all 20 +
   parse_check. Fixture `transpose_fixture.pb`.
10. **transpose perm decode round-trip (PROBLEMATIC.md #6).** `pb2egg._decode_perm`
    matches a hardcoded `permutation_to_index` idx oracle (encoder-independent) and
    rejects non-permutations, plus a taso `core` round-trip `transpose(perm) ->
    get_operator_attr('perm')`. Guards the Release-build uninitialized-read that
    `fb0b3db` fixed. See `test_transpose_perm.py`.
11. **const_* emission — gated by default (tier-2, PROBLEMATIC.md #8).** the const
    ops are apply-UNsafe, so pb2egg gates them by default (0 emitted, 24 unapplicable);
    `--emit-unapplicable` emits all 24 across the 4 types + parse_check. Fixture
    `const_fixture.pb`.
12. **apply-smoke (PROBLEMATIC.md #8 application gap).** The gate the
    emission/parse_check/Z3 tests can't give: a guaranteed-fire rule per op family
    through a 2-iteration saturation on `mnist_tiny_mlp` — apply-safe ops
    (ewadd/matmul/ewmax/ewmul) must NOT hit the `rewrites.rs` `todo!()` panic, and
    gated ops (transpose, const Iewmul) MUST (proving the gating is load-bearing).
    This is what would have caught transpose/const shipping as apply-panicking rules.

Tests 5–7 are pure-Python (stdlib only) and need neither GPU nor the tensat
binary; tests 2, 4, 8, 9, 11, 12 drive the prebuilt `tensat` binary (no rebuild). They
run in-container because that is where the toolchain lives.

## Z3 tensor-axiom lane (separate suite)

`test_z3_axioms.sh` covers the conv/concat/matmul verifier lane
(`NNs/tensor_axioms.py`). It runs with the **`taso_py` env**, not in the
container (z3 lives there, not in the container python — #5), so it is a separate
host-run suite:
```
bash NNs/tests/test_z3_axioms.sh
```
Asserts (18, all passing): the 8 negative canaries in `z3_canaries_false.txt`
stay unproven (soundness — a consistent axiom set can't prove `conv(x,w)=conv(w,x)`,
a non-involutive `1_2_0` double-transpose = identity, or the identity-matrix-vs-
all-ones confusions `ewmul(x,Imatmul)=x` / `matmul(x,Iewmul)=x`); 6 flips
(conv-linearity, relu(conv)=conv+relu, relu-over-concat, 2-D transpose involution,
matmul·identity-matrix, conv·identity-conv) that lane 1 rejects and lane 2 proves;
2 shuffle-invariance rewrites (transpose shuffle 0≡1) that must verify; and 2 PWL
regressions still proven by lane 1.

## Outstanding (tracked, not yet done)
- **Z3 conv-axioms — DONE (2026-09-01).** conv/concat/matmul rewrites now verify via the
  `tensor_axioms.py` lane (port of `taso/verify/verify.py`): 35→104/116 on the tracked pb.
  See `../../PROBLEMATIC.md` #7. Follow-up: rerun the full 6k `fullop` corpus through it;
  ~12 grouped-conv/matmul-fold rules remain (TASO's own verifier doesn't prove them either).
- **Tensat application gap (open):** transpose + const_* emission/verification are done,
  but tensat can't APPLY those ops (rewrites.rs `todo!()`), so pb2egg gates them by default
  (`--emit-unapplicable` to keep). enlarge is blocked behind the same gap; reshape absent;
  split is the multi-pattern lane. See PROBLEMATIC.md #8 and docs/ADD_AN_OP.md.
