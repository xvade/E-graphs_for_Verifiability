# Learned multi-pattern corpus — firing probe on resnet2b

**Task:** build a `.pb → two-line text` converter for the 16,878 generated multi-pattern
rules (`fullop_subst_d3_egg.multi.pb`, which had no text form), then run the instrumented
firing probe on resnet2b to measure how many actually fire under the cycle filter.

## Converter (`NNs/pb2multi.py`)
Reuses `pb2egg.build()`; adds a `split`-root arm for multi-output fusion. Drops per pair:
mappedOutput≠2, same-src, unconvertible op, unbound RHS var (over BOTH lines' merged subst),
apply-unsafe dst (concat3/4/5 panic `check_pat`), and **vacuous (id,id) pairs** (10,655 of
16,878 — assert nothing, add no e-node, but still inflate `cycle_ok`). Output:
`fullop_subst_d3_multi.txt` = **4,258 pairs / 8,516 lines** (no trailing newline).
Smoke-tested 8 stratified real pairs on resnet2b: exit 0, no panic, output parseable + apply-safe.

## Probe (`NNs/multi_firing_probe.sh`), resnet2b, both cycle modes, both **Saturated**
Funnel columns = pairs → compatible (merged subst agrees) → valid (both dst pass `check_pat`)
→ cycle_ok (dst built + unioned). `--no_cycle` present = cycle filter ON = the answer run.
Per-rule `this_rule` counters and the totals line are **cumulative across the whole run and
monotonic** (verified: rule[3930] 2→32→144; totals 284→6,056→61,904→99,728); the last totals
line is the final count.

| run | pairs | compatible | valid | cycle_ok | stop |
|-----|------:|-----------:|------:|---------:|------|
| **cycle-filter ON (answer)** | 99,728 | 77,388 | 77,340 | **64,458** | Saturated (iter 4) |
| cycle-filter OFF (upper bound) | 18,886,681 | 18,864,359 | 3,996,240 | 3,996,240 | Saturated |

## HEADLINE: it fires, but its NET e-graph contribution on resnet2b is **zero**
The learned multi corpus applies **64,458** times on resnet2b (cycle filter ON), so it is not
inert here — but a control settles what those applications *do*:

| run | final nodes | final classes | saturate |
|-----|------------:|--------------:|---------:|
| `-r converted.txt -t multi.txt` (with multi lane) | **183** | **112** | 3.73 s, iter 4 |
| `-r converted.txt` only (**control, no multi lane**) | **183** | **112** | 0.02 s, iter 7 |

Identical. The multi lane grows the e-graph by **0 nodes / 0 classes** — every one of its 64,458
applications re-derives a node the single-pattern lane already builds (union of already-merged
classes / hashcons hit), and it costs +3.7 s for that zero structural gain. The reason is direct:
stock `converted.txt` **already contains the ewadd 3-term reassociations** (lines 22–31, e.g.
line 24 `a+(b+c)⇒b+(a+c)`), and ewadd-AC is the *only* learned family whose LHS binds on
resnet2b's op set (no ewmax/ewmin/ewsub/smul; matmul-nest needs a matmul nest resnet2b lacks).
So the corpus's binding content is fully redundant with stock single-pattern rules on this model.

This does **not** contradict the earlier "0 fired" probe (`multi-rule-firing-probe`) — that was
the *stock predefined* multi rules (`converted_multi.txt`). Different rule set: the learned corpus
fires where the stock multi corpus didn't; neither adds structure resnet2b's single-pattern lane
lacks.

## Cycle filter and shape gating (answer run)
`check_cycle` (rewrites.rs:1838–1848) is an **ancestry guard**: reject iff a matched *input*
eclass already has an *output* root among its descendants. Valid→cycle_ok drops 77,340→64,458, so
it blocks 12,882 (17%), spread as partial blocking across the firing ewadd rules. Fully-blocked
rules (valid>0, cycle_ok=0) in the answer run:
- **6 matmul-gated ewadd forms** (rule[1],[2],[166],[167],[216],[218]: valid=192, cycle_ok=0) —
  the gating matmul is the final FC, ancestor of every ewadd, so every application would close a
  cycle. (The smoke's "100% blocked rule[1]" is this family — matmul-*gated*, **not** matmul-nest;
  the matmul-nest smoke rule[0] matched **0 eclasses**, resnet2b having a single matmul and no nest.)
- **2 relu split-fusion forms** (rule[3665],[3731]: valid=6, cycle_ok=0).

The rest of the split-fusion family is **shape-blocked, not cycle-blocked** — it never reaches
valid on resnet2b's shapes (valid=0). Only in the cycle-OFF upper bound do 4 relu split-fusion
forms reach applications (~2,053 each), and even there their net node contribution is 0 (control
above).

## What actually fires (`classify_fired_multi.py`, i < 4,258 = ours; predefined fired 0)
All fired OUR rules are **ewadd 3-term reassociations**.
**Answer run (cycle ON):** 36 indices → **3 distinct real forms** — `a+(b+c)⇒b+(c+a)` (35,520
apps), `⇒c+(b+a)` (28,650), `⇒b+(a+c)` (288). The first two are in the single-pattern dedup; the
third is not, but is derivable from ewadd comm+assoc (the redundancy pruner is the arbiter). Per-
form app sums exceed the 64,458 total because a (real,real) pair is counted under both its forms.
**Upper bound (cycle OFF):** the same 3 ewadd forms plus 4 relu split-fusion forms.

## Takeaway
On resnet2b the learned multi corpus is a **null result at the e-graph level**: it fires 64,458×
but its only binding family (ewadd AC) is already provided by stock single-pattern rules, so it
adds no node, no class, and no new extractable term — it only adds runtime. The multi lane's
distinctive payoff (split-fusion) is shape/cycle-blocked here. The corpus's value is therefore not
on resnet2b at all but on nets whose op set makes its **ewmin/ewmax/ewsub** halves bind — the
maxout/PWL family, where those AC reassociations are exactly the generator's AC-blind gap and are
*not* stock. Recommended (not executed, beyond "run the probe"): extract the real halves of the
(id,real) pairs — the identity half is only a firing gate, the real half is unconditionally sound —
Z3-verify the min/max ones not in dedup, and feed *those* (not the stock-redundant ewadd ones) to
the single-pattern lane, then re-probe on a maxout net.
