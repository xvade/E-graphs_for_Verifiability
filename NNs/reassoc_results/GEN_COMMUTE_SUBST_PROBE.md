# Probe: does substitution pruning kill GEN_COMMUTE's rules? (2026-09-01)

> **Current counts live in the test, not this table.** The magnitudes in the
> table below were an early measurement; the generator has since gained ops, so
> the current deterministic depth-2 counts are 38/267/488/34524 transfers
> (commutativity 0/2/0/36). The **conclusions are unchanged** — the 0/2/0/many
> commutativity pattern is the point. `taso/src/generator/tests/test_flags_probe.sh`
> is the executable, always-current form of this probe.

**Question:** With `RELAX_SUBST` OFF (substitution dedup active), does GEN_COMMUTE's
commutativity output get pruned away immediately?

**Answer: NO.** The dedup collapses GEN_COMMUTE's redundant *copies* but KEEPS the canonical
commutativity representatives. GEN_COMMUTE + RELAX_SUBST-off is the config that yields
commutativity WITHOUT the copy-flood that broke the 2 GB protobuf ceiling.

## Method
Depth-2 PWL-focus generator (`-DPWL_FOCUS -DGEN_MAX_DEPTH=2`), flags toggled by **presence**
(unset = off; NOTE `X=0` is still "set" -> TRUE, since the generator does `getenv("X")!=nullptr`).
Decoded each `graph_subst.pb` with `pb2egg.py`, grepped bare 2-leaf ewmax/ewmin swaps.

| config | subst-dedup | transfers | egg rules | bare commut |
|---|---|---|---|---|
| CANON (nothing set)        | active   | 33   | 23  | **0** |
| COMMUTE_only (GEN_COMMUTE) | active   | 98   | 72  | **2** |
| VARORDER_only              | active   | 33   | 23  | 0 |
| COMMUTE+VARORDER           | active   | 98   | 72  | 2 |
| SUBST_only (RELAX_SUBST)   | disabled | 260  | 223 | **0** |
| COMMUTE+SUBST              | disabled | 2624 | 885 | **20** |

## Conclusions
1. **Canonical suppresses commutativity** (CANON = 0). Confirmed by construction.
2. **GEN_COMMUTE survives the dedup** (COMMUTE_only = 2): the two surviving rules are exactly
   `(ewmax a b)=>(ewmax b a)` and `(ewmin a b)=>(ewmin b a)`. The substitution dedup does NOT
   delete them -- it collapses the 20-way flood down to these 2 canonical representatives.
3. **RELAX_SUBST alone is NOT a source of commutativity** (SUBST_only = 0). You need GEN_COMMUTE
   to BUILD the swapped operand; disabling the dedup only decides whether the copies are kept.
4. **RELAX_VARORDER is irrelevant to commutativity** (VARORDER_only = 0; COMMUTE+VARORDER = COMMUTE_only).
5. **Flood factor:** subst-off keeps ~12x the rules at depth 2 (885 vs 72) for the same
   commutativity content -- this multiplicative blowup at depth 3 is what pushed commute_d3
   past the single-message 2 GB protobuf limit (3.22 GB / 14.4M transfers).

## Recommendation for the commute regeneration
Run **`GEN_COMMUTE=1` with `RELAX_SUBST` UNSET** (keep RELAX_SUBGRAPH/SUPERGRAPH/VARORDER if you
want their coverage). You get the ewmax/ewmin commutativity rules the lattice needs, and the
substitution dedup keeps the transfer count bounded -- no 2 GB explosion. `commute_d3` failed
precisely because it set RELAX_SUBST=1 *and* GEN_COMMUTE=1, keeping every swapped copy.

## Env gotchas (cost me two invalid runs)
- Generator flags are **presence-checked** (`getenv != nullptr`). `GEN_COMMUTE=0` = ON. Toggle by
  unsetting.
- Working python for pb2egg/rules_pb2 is **`toolchain-tensat/miniconda3/envs/taso_py/bin/python3`**
  (3.10, protobuf 7.36). The bare miniconda3 python3.14 and /usr/bin/python3 both fail to import.
- The generator binary needs **`LD_LIBRARY_PATH=toolchain-tensat/miniconda3/lib`** (libprotobuf.so.32).
  `env -i` strips it -> exit 127, and a stale `graph_subst.pb` gets copied silently. Always check
  the pb is non-empty / freshly written.

Raw outputs: `$CLAUDE_JOB_DIR/tmp/probe/x_*.egg` (job-scratch, ephemeral).
