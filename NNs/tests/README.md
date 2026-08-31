# Pipeline tests

Run inside the container (no pytest dependency; plain assert harness):
```
apptainer exec tensat.sif bash NNs/tests/run_tests.sh
```
Exits nonzero if any test fails.

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

## Outstanding (tracked, not yet done)
- **Z3 conv-axioms.** pb2egg now EMITS conv/concat rules, but z3_verify_egg treats
  conv/pool/concat as UNINTERPRETED (sound but conservative) -> most conv rewrites get
  REJECTED at verification. Keeping them needs the op linearity/distribution axioms (cf.
  taso/verify/validate_axioms.py). Until then, full-op pb2egg unblocks parsing but conv
  rules won't survive verification.
- **Tier-2 ops still dropped:** transpose/reshape (config-as-name-string), enlarge (pb
  kernel-based vs egg ref-based -- semantic mismatch), split (multi-output -> multi-pattern
  lane). See pb2egg.py operator_data comments.
