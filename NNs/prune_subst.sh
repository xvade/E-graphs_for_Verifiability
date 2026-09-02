#!/bin/bash
# Stage 5: redundancy-prune the VERIFIED subst corpus -> fullop_subst_d3_core.txt.
#
# The redundancy pruner materializes each rule into a concrete taso graph to test
# re-derivability, and that materializer only implements the PWL/algebraic op arms
# (rewrites.rs apply-match). Two failure modes on structural ops:
#   * pool on a non-4D transient  -> Pool2D assertion `_input.numDim == 4` (pool2d.cc:132)
#   * conv2d/concat/transpose/Iconv/Cpool -> `other => todo!()` (rewrites.rs:1038, on a Var)
# Either one ABORTS the whole run. So we SPLIT the verified corpus:
#   SAFE   = ewX / smul / matmul / relu / Imatmul / Iewmul only            -> prune
#   STRUCT = anything with conv2d|poolavg|poolmax|concat|transpose|Iconv|Cpool -> keep verbatim
# core = prune(SAFE) ++ STRUCT. Pruning SAFE in isolation is CONSERVATIVE: it has fewer
# "other" rules to derive from, so it keeps a SUPERSET of a full-corpus prune -- never drops
# a rule a full prune would have needed. tensat's debug binary is built against the container
# glibc, so tensat MUST run inside tensat.sif (bare g3109 lacks GLIBC_2.29+).
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
OUT="$REPO/NNs/reassoc_results"
P=fullop_subst_d3
DENY='conv2d|poolavg|poolmax|concat|transpose|Iconv|Cpool'
ts(){ date +%H:%M:%S; }

V="$OUT/${P}_verified.txt"
SAFE="$OUT/${P}_safe.txt"
STRUCT="$OUT/${P}_struct.txt"
SAFE_CORE="$OUT/${P}_safe_core.txt"
CORE="$OUT/${P}_core.txt"

grep -vE "$DENY" "$V" > "$SAFE"
grep -E  "$DENY" "$V" > "$STRUCT"
echo "[$(ts)] split verified ($(wc -l < "$V")): SAFE=$(wc -l < "$SAFE")  STRUCT=$(wc -l < "$STRUCT")"

echo "[$(ts)] redundancy-prune SAFE (inside tensat.sif) -> $(basename "$SAFE_CORE")"
S=$(date +%s)
apptainer exec "$REPO/tensat.sif" bash -lc "
  cd '$REPO/tensat'
  export LD_LIBRARY_PATH=\$PWD/../taso/build:/opt/conda/lib
  ./target/debug/tensat -m redundancy -r '$SAFE' -o '$SAFE_CORE' \
    --redundancy_iters 4 --n_nodes 8000 --n_sec 4 2>&1 | tail -8
"
E=$(date +%s); echo "[$(ts)] prune wall: $((E-S)) s"

if [ ! -s "$SAFE_CORE" ]; then
  echo "[$(ts)] ERROR: SAFE prune produced no output; aborting (core not assembled)"; exit 1
fi

# newline guard: tensat's SAFE_CORE has no trailing newline, so a bare cat would glue
# the last SAFE rule onto the first STRUCT rule into a garbage double-=> rewrite.
{ cat "$SAFE_CORE"; [ -n "$(tail -c1 "$SAFE_CORE")" ] && echo; cat "$STRUCT"; } > "$CORE"
echo "[$(ts)] core assembled: $(wc -l < "$CORE") = pruned-SAFE $(wc -l < "$SAFE_CORE") + STRUCT $(wc -l < "$STRUCT") (verbatim)"
echo "[$(ts)] min/max in core: $(grep -cE 'ewmax|ewmin' "$CORE" || true)"
echo "[$(ts)] DONE"
