# Full 621 rules on a min-of-max lattice + breadth on TASO models (follow-on)

## Min-of-max lattice -- NULL result (informative)
Model: single-input lattice g(x)=min_g max_k (W.x+b), G=2 groups of K=8 (same W,b as
the maxout run), balanced trees, vector trick. tensat saturated with the FULL 621
Z3-verified PWL rules (incl. the max=(a+b)-min bridges), extracted 40 diverse forms.
- All 40 forms pass the numeric gate (~5e-7): semantics preserved.
- **Every form's certified ub = 8.5019 = the input** (14/120 unstable, identical).
  Improvement = -0.0000. Depths vary (7,9x38,11) but the bound does not.
- Envelope: chain=7.8026, balanced(input)=8.5019 -- so a tighter (chain) form EXISTS,
  but tensat's n_diverse forms are all near-balanced (~depth 9) and never reach it.

Why (vs maxout's +1.46): the outer MIN dominates the lattice's certified bound, and
reassociating the inner max-groups doesn't move ub of either group -> min(A,B) bound is
invariant. With only G=2 groups the min layer itself has no reassociation freedom. And
the deep-chain form (envelope 7.80) is a vanishing-probability tail for n_diverse
sampling. This echoes the PoC Result 2 (min-of-max lattices are the washed-out case).
Reaching 7.80 would need a VERIFIABILITY-AWARE extraction cost (steer toward the chain)
-- the explicit next lever, not random sampling.

## Breadth: 621 on the other TASO-ingestible (Conv/Matmul) models -- INERT
Ran the full 621 through tensat (n_diverse 8), counting distinct structures:
- mnist_tiny_mlp: 8 -> 2 distinct   resnet2b: 8 -> 1 (!=input)
- mnist_cnn_a:    8 -> 1            inception: 8 -> 1 (!=input)
No panics. These nets have no min/max ops, so the min/max/PWL rules are essentially
inert (1-2 forms). Confirms the min/max verifiability lever is specific to
min/max-structured models; Conv/Matmul nets don't rewrite meaningfully under it.

## Net
The verifiability improvement is REAL but STRUCTURE-SPECIFIC: pure-max (maxout) +1.46;
min-of-max (lattice) 0 under n_diverse (min dominates + chain not sampled); Conv/Matmul
inert. The clear next lever is a verifiability-aware extraction cost so tensat STEERS to
the tight form instead of us sampling+picking.
