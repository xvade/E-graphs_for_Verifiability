#!/usr/bin/env python3
"""Verifier-conditional demonstration for the /goal task.

Claim under test: does folding two consecutive INPUT-DEPENDENT linear ops
(y = B(Ax) -> (BA)x, no nonlinearity between) make a plain-ReLU net more
verifiable? Answer depends entirely on the verifier's bounding discipline:

  IBP   RE-BOXES the intermediate Ax into a fresh interval, then applies B to
        that box -> pre-activation radius |B|(|A| r) = |B||A| r.
        Folding -> single map, radius |BA| r. Since |BA| <= |B||A| elementwise
        (triangle ineq on the matrix product), folding is >= as tight, and
        STRICTLY tighter exactly where a row of BA has sign cancellation.

  CROWN back-substitutes SYMBOLICALLY: it never re-boxes a linear chain, it
        composes the coefficient to the input (B*A) and boxes ONCE at the ReLU
        -> pre-activation radius |BA| r for BOTH forms. Folding changes nothing.

So folding's entire value is to recover, under IBP, the tightness CROWN already
gets for free. The project verifies with alpha-beta-CROWN (bound_maxout_forms.py:
method='CROWN-Optimized'), so folding a plain-ReLU net is NEUTRAL for it -- which
is why every resnet rewrite probe came back verify-neutral.

This script proves all four numbers on two concrete linear pairs.
"""
import numpy as np
np.random.seed(0)

def ibp_relu_chain(maps, c, r):
    """IBP through a list of linear maps with RE-BOXING between each.
    Returns the pre-activation interval (lo, hi) of the final map's output."""
    for W in maps:
        c = W @ c
        r = np.abs(W) @ r          # re-box: interval width recomputed each step
    return c - r, c + r

def crown_linear_prefix(maps, c, r):
    """CROWN on a purely-linear prefix: compose the symbolic map to the input,
    box ONCE. This is exactly what back-substitution does for a ReLU whose
    entire cone of ancestors up to the input is linear."""
    M = np.eye(maps[0].shape[1])
    for W in maps:                 # M := W @ ... @ maps[0]
        M = W @ M
    return M @ c - np.abs(M) @ r, M @ c + np.abs(M) @ r

def unstable_count(lo, hi):
    """A ReLU neuron is unstable (needs a relaxation triangle) iff its
    pre-activation interval straddles 0."""
    return int(np.sum((lo < 0) & (hi > 0)))

def report(name, A, B):
    print(f"===== {name} =====")
    n = A.shape[1]
    c = np.zeros(n)                      # input box center 0, radius 1 (l_inf)
    r = np.ones(n)

    # (i) the two forms compute the IDENTICAL function
    X = np.random.randn(200, n)
    two_step = (B @ (A @ X.T)).T
    folded   = ((B @ A) @ X.T).T
    print(f"  function identical on 200 samples: max|diff| = {np.max(np.abs(two_step-folded)):.2e}")

    # (ii) IBP pre-activation interval into the ReLU that consumes B(Ax)
    lo_u, hi_u = ibp_relu_chain([A, B], c, r)     # unfolded: re-box after A
    lo_f, hi_f = ibp_relu_chain([B @ A], c, r)    # folded:   single map
    # (iii) CROWN pre-activation interval -- same for both forms
    lo_c, hi_c = crown_linear_prefix([A, B], c, r)

    print(f"  IBP  unfolded pre-act radius (|B||A|r): {(hi_u-lo_u)/2}")
    print(f"  IBP  folded   pre-act radius (|BA| r ): {(hi_f-lo_f)/2}")
    print(f"  CROWN both forms pre-act radius       : {(hi_c-lo_c)/2}")
    print(f"    -> CROWN == IBP-folded ? {np.allclose((hi_c-lo_c),(hi_f-lo_f))}")
    print(f"    -> IBP folded <= IBP unfolded (elementwise)? "
          f"{np.all((hi_f-lo_f) <= (hi_u-lo_u)+1e-9)}")
    print(f"  unstable ReLUs  IBP-unfolded={unstable_count(lo_u,hi_u)}  "
          f"IBP-folded={unstable_count(lo_f,hi_f)}  CROWN={unstable_count(lo_c,hi_c)}")

    # (iv) final certified OUTPUT radius after ReLU -> one more linear map C.
    # Use IBP for the ReLU step so the pre-act difference propagates.
    C = np.random.randn(2, B.shape[0])
    def after_relu_ibp(lo, hi):
        rl_lo, rl_hi = np.maximum(lo,0), np.maximum(hi,0)
        oc, orad = (rl_lo+rl_hi)/2, (rl_hi-rl_lo)/2
        return np.abs(C) @ orad          # output radius
    print(f"  final output radius (IBP tail): unfolded={after_relu_ibp(lo_u,hi_u)}  "
          f"folded={after_relu_ibp(lo_f,hi_f)}")
    print()

# ---- pair 1: MIXED-SIGN maps (a factored/low-rank or conv->conv site): STRICT ----
A1 = np.array([[ 1.0, -1.0],
               [ 1.0,  1.0]])
B1 = np.array([[ 1.0,  1.0],
               [-1.0,  1.0]])
# B1@A1 = [[2,0],[0,2]] -- massive cancellation vs |B1||A1| = [[2,2],[2,2]]
report("pair 1: mixed-sign linear-linear (conv->conv / factored FC)  [STRICT]", A1, B1)

# ---- pair 2: NONNEGATIVE disjoint (globalavgpool -> FC): NO benefit even under IBP ----
P = np.array([[0.5, 0.5, 0.0, 0.0],     # avgpool: nonneg, disjoint windows
              [0.0, 0.0, 0.5, 0.5]])
F = np.abs(np.random.randn(3, 2))        # FC after pool; sign of F is what matters
report("pair 2: nonneg pool -> FC (a real resnet head site)  [IBP gives NOTHING]",
       P, F)
print("Note pair 2: |F P| == |F| P exactly (each (FP)_ij is a single term, no")
print("cancellation), so even IBP is neutral -- the one input-dependent linear")
print("pair a v1/v2 resnet actually has (globalavgpool->FC) has provably zero")
print("folding benefit under any verifier.")
