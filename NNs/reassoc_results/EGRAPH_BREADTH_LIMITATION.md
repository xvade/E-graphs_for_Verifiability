# Note: breadth-first saturation may be a fundamental e-graph downside

**Status:** open methodological concern (HYPOTHESIS, not yet demonstrated). We are NOT
changing project direction (still e-graphs / equality saturation for verifiability). This
records a limitation and two candidate remedies to revisit.

> **CORRECTION (important):** the lattice run cited below as the "concrete" instance of
> breadth-first starvation was RE-DIAGNOSED as a RULE GAP -- the 621 rule set simply has
> no pure ewmax associativity/commutativity, so the group max-trees can't be reassociated
> at all (adding those 4 rules -> Saturated at 612 classes, tight depth-7 chain present;
> see VERIF_COST_RESULT.md). So the concern below stands as a GENERAL worry about
> equality saturation, but I have NO clean empirical instance of it yet. Read the rest as
> motivation, not evidence.

## The observation

Equality saturation grows the e-graph **breadth-first**: every iteration applies every
matching rule everywhere, materializing all one-step rewrites before going deeper. The
e-graph blows up in width long before any single deep rewrite path is realized.

We hit this concretely (see `VERIF_COST_RESULT.md`, chain-query). On the min-of-max
lattice, the tight (left-deep chain) form is provably NOT in the saturated e-graph:

- `Stopped: TimeLimit(120s)` at iter 10/12; 37,806 nodes / 15,808 classes — i.e. only
  **7% of the 500k node budget** used, so it died on the **wall clock**, not node count.
- The natural-order running-max spine only reached depth 2/7; an order-independent
  subset closure found **no** depth-7 chain over the 8 leaves in any order.
- Associativity *does* fire (shallow spines exist) — the rule is capable, the search
  just spent its whole budget on width and never reached that particular depth-7 tail.

So a rewrite that is only a handful of *targeted* rule applications away can be
effectively **unreachable** because saturation insists on exploring everything shallower
first. For verifiability specifically, the payoff structure (a deep left-deep spine that
maximizes ReLU stability) is exactly the kind of narrow-and-deep target this penalizes.

## Why this matters for the project

Our verifiability wins depend on *reaching a specific structure*, then extracting it.
Saturation is good at the "extract from a rich set" half but can starve the "reach the
target structure" half. The bottleneck we keep hitting on the lattice is **reachability**,
not the extraction cost — breadth-first search is the mechanism behind that wall.

## Candidate remedies (to revisit, not committing now)

1. **Monte-Carlo Tree Search over rewrite sequences.** Instead of applying all rules
   everywhere, treat rule application as a sequential decision problem: MCTS (or another
   best-first / guided search) can go **deep along promising branches** — e.g. keep
   extending a running-max spine — spending budget where a value estimate (here, a
   verifiability proxy like the ReLU-gap surrogate we already built) says the payoff is.
   Trades the completeness of saturation for depth-reachability. Related prior art:
   guided/`beam`-style egg extensions and RL-guided term rewriting.

2. **Heuristic-based rule application (priority-ordered / greedy saturation).** Keep the
   e-graph but stop treating all matches as equal: order or gate rule firings by a
   heuristic (verifiability-relevant matches first; deprioritize width-only rewrites),
   possibly a priority queue of pending applications rather than a full-iteration sweep.
   Cheaper than MCTS, keeps the e-graph machinery, directly attacks the "never reaches
   the deep target" failure. A middle ground: run normal saturation, but with a targeted
   deepening pass that repeatedly applies the associativity direction that extends the
   spine.

## Cheap next check before either

Before investing in MCTS/heuristic search, raise `--n_sec` (e.g. 1200) and re-run
`--query_chain`. If the depth-7 chain then materializes, the wall is purely a *budget*
knob and the fancy search is optional; if saturation *completes* (`Stopped: Saturated`)
still without the chain, that's a genuine rule-direction gap — check the ewmax assoc
rule directions in the 621. Either way, the breadth-first concern above stands as the
reason a targeted search could beat brute-force saturation.
