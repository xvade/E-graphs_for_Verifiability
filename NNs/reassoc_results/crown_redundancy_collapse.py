"""GOAL: a manual, EXACT, min/max-FREE rewrite that tightens a full CROWN bound.

Prior finding ([[plain-relu-rewrites-cant-move-crown-bound]]): rearranging the LINEAR
skeleton of a plain-ReLU net is CROWN-neutral, and only min/max reassociation moves a
CROWN bound. That induction fixes the NEURON SET (up to nonneg-monomial relabeling). Its
hole: equivalent nets with a DIFFERENT NUMBER of ReLU nodes whose pre-activations are
linearly dependent -- REDUNDANT ReLU structure.

The rewrite (exact, no min/max): MERGE duplicated unstable neurons that share a
pre-activation. If two neurons both compute relu(z) (identical row) and feed the output
with coeffs c1, c2, CROWN relaxes the two copies INDEPENDENTLY -> slack (|c1|+|c2|)*gap;
the merged single neuron (coeff c1+c2) has slack |c1+c2|*gap. Strictly tighter iff
sign(c1) != sign(c2) (coefficient cancellation the duplicated form can't see).

Baseline = an over-parameterized / "compiled" net with redundant ReLU copies (the tll
lift showed compiled nets really do carry redundant relu gadgets); rewrite = collapse
them. CONTROL = a same-sign pair, which the math says merges EXACTLY neutrally -- that
control is what proves the mechanism is coefficient cancellation, not a generic effect.

TRAP (audited below): the duplicates MUST be distinct ROWS of the weight matrix. Reusing
one tensor makes torch trace ONE node and auto_LiRPA shares it -> false neutrality.

Run in the abcrown venv:
  alpha-beta-CROWN/.venv/bin/python NNs/reassoc_results/crown_redundancy_collapse.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "alpha-beta-CROWN", "complete_verifier"))
import numpy as np, torch, torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

torch.manual_seed(0); np.random.seed(0)
D_IN, W1_N = 8, 16          # input dim, first hidden width
EPS = 1.0
x0 = torch.zeros(1, D_IN)
ptb = PerturbationLpNorm(norm=np.inf, eps=EPS)

# shared first layer (identical in both forms; its relaxation is a common term that
# cancels in the dup-vs-merged DELTA, so the delta isolates the layer-2 mechanism).
W1 = torch.randn(W1_N, D_IN) * 0.7
b1 = torch.randn(W1_N) * 0.3

def h1_samples(n=40000):
    xs = (torch.rand(n, D_IN) * 2 - 1) * EPS + x0
    return torch.relu(xs @ W1.T + b1)

H1 = h1_samples()

def make_pair_rows(rng, n_pairs):
    """Each pair: one shared row w (mixed sign) + a bias tuned so the pre-activation
    STRADDLES 0 over the input box (=> genuinely unstable, real relaxation slack)."""
    rows, biases, gaps = [], [], []
    for _ in range(n_pairs):
        w = torch.tensor(rng.standard_normal(W1_N), dtype=torch.float32)
        z = H1 @ w
        bz = -float(z.median())                 # center pre-activation on 0
        pre = z + bz
        assert float(pre.min()) < 0 < float(pre.max()), "planted neuron not unstable"
        rows.append(w); biases.append(bz); gaps.append(float(pre.max() - pre.min()))
    return rows, biases, gaps


def build_forms(n_pairs, coeff_signs, seed=1, n_normal=4):
    """Returns (dup_net, merged_net) that are function-identical. `coeff_signs`:
    'oppose' -> the two copies get +a,-b (cancel); 'same' -> +a,+b (control)."""
    rng = np.random.default_rng(seed)
    rows, biases, gaps = make_pair_rows(rng, n_pairs)
    # downstream coeffs per pair. PINNED to moderate values (a=1.0, b=0.8) rather than
    # drawn: a random draw can hand one pair a near-zero net |cA+cB| (~"merged neuron
    # vanishes"), inflating the headline into a lucky seed. Fixed 0.2 net (ratio 0.111)
    # is a defensible moderate cancellation, identical across pairs -> robust, reproducible.
    cAs, cBs = [], []
    for _ in range(n_pairs):
        a, b = 1.0, 0.8
        cAs.append(a)
        cBs.append(-b if coeff_signs == "oppose" else b)
    # a few ordinary (non-duplicated) neurons, identical in both forms
    nrows, nbias, ncoef = [], [], []
    for _ in range(n_normal):
        w = torch.tensor(rng.standard_normal(W1_N), dtype=torch.float32)
        z = H1 @ w; bz = -float(z.median())
        nrows.append(w); nbias.append(bz); ncoef.append(float(rng.uniform(-1, 1)))

    def assemble(dup):
        W2_rows, b2, c3 = [], [], []
        for p in range(n_pairs):
            if dup:                                  # two DISTINCT identical rows
                W2_rows += [rows[p], rows[p]]; b2 += [biases[p], biases[p]]
                c3 += [cAs[p], cBs[p]]
            else:                                    # merged single neuron
                W2_rows += [rows[p]]; b2 += [biases[p]]
                c3 += [cAs[p] + cBs[p]]
        for q in range(n_normal):                    # normals: identical both forms
            W2_rows += [nrows[q]]; b2 += [nbias[q]]; c3 += [ncoef[q]]
        m = len(W2_rows)
        lin1 = nn.Linear(D_IN, W1_N); lin1.weight.data = W1.clone(); lin1.bias.data = b1.clone()
        lin2 = nn.Linear(W1_N, m); lin2.weight.data = torch.stack(W2_rows); lin2.bias.data = torch.tensor(b2)
        lin3 = nn.Linear(m, 1, bias=False); lin3.weight.data = torch.tensor(c3).view(1, m)
        return nn.Sequential(lin1, nn.ReLU(), lin2, nn.ReLU(), lin3)

    return assemble(dup=True), assemble(dup=False), gaps


def bounds(net, method):
    bm = BoundedModule(net, x0, verbose=False)
    lb, ub = bm.compute_bounds(x=(BoundedTensor(x0, ptb),), method=method)
    return float(lb), float(ub)


def relu_node_count(net):
    bm = BoundedModule(net, x0, verbose=False)
    from auto_LiRPA.bound_ops import BoundRelu
    return sum(isinstance(n, BoundRelu) for n in bm.nodes())


def run(tag, coeff_signs, n_pairs=4):
    print(f"\n===== {tag} (n_pairs={n_pairs}, coeffs={coeff_signs}) =====")
    dup, merged, gaps = build_forms(n_pairs, coeff_signs)
    print(f"planted-pair pre-activation gaps (unstable): "
          f"{[round(g,2) for g in gaps]}")
    print(f"layer-2 width: dup={dup[2].out_features}  merged={merged[2].out_features} "
          f"(distinct rows -> {relu_node_count(dup)} vs {relu_node_count(merged)} ReLU nodes)")
    xs = torch.randn(50, D_IN)
    assert torch.allclose(dup(xs), merged(xs), atol=1e-5), "forms not function-identical!"
    print("function-identical on 50 samples: OK")
    for method in ["IBP", "CROWN", "CROWN-Optimized"]:
        ld, ud = bounds(dup, method); lm, um = bounds(merged, method)
        wd, wm = ud - ld, um - lm
        rel = (wd - wm) / abs(wd) * 100 if wd else 0.0
        verdict = ("MERGED TIGHTER" if wm < wd - 1e-6 else
                   "neutral" if abs(wm - wd) <= 1e-6 else "MERGED LOOSER")
        print(f"  {method:16s} dup=[{ld:8.4f},{ud:8.4f}] w={wd:8.4f} | "
              f"merged=[{lm:8.4f},{um:8.4f}] w={wm:8.4f} | "
              f"delta={wd-wm:+.4f} ({rel:+.1f}%) -> {verdict}")


# ---------------------------------------------------------------------------
# Rule 2: COMPLEMENTARY-PAIR COLLAPSE (the more natural case -- no artificial
# duplication). A net that computes BOTH relu(z) and relu(-z) (a common
# "two-sided feature") can collapse via the EXACT identity relu(-z) = relu(z) - z:
#   c1*relu(z) + c2*relu(-z) = (c1+c2)*relu(z) - c2*z
# Two unstable ReLUs -> ONE ReLU + a LINEAR correction (-c2*z, a skip from h1).
# Same cancellation condition: strictly tighter iff sign(c1) != sign(c2).
class CollapsedNet(nn.Module):
    """h1=relu(W1 x+b1); h2=relu(W2 h1+b2); out = c3.h2 + (skipW.h1 + skipb).
    The skip carries the exact -c2*z linear corrections."""
    def __init__(self, W1, b1, W2, b2, c3, skipW, skipb):
        super().__init__()
        self.l1 = nn.Linear(*W1.shape[::-1]); self.l1.weight.data = W1; self.l1.bias.data = b1
        self.l2 = nn.Linear(*W2.shape[::-1]); self.l2.weight.data = W2; self.l2.bias.data = b2
        self.l3 = nn.Linear(c3.shape[1], 1, bias=False); self.l3.weight.data = c3
        self.skip = nn.Linear(skipW.shape[1], 1); self.skip.weight.data = skipW; self.skip.bias.data = skipb
    def forward(self, x):
        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        return self.l3(h2) + self.skip(h1)


def build_complementary(n_pairs, coeff_signs, seed=2, n_normal=4):
    rng = np.random.default_rng(seed)
    rows, biases, gaps = make_pair_rows(rng, n_pairs)
    cAs, cBs = [], []
    for _ in range(n_pairs):
        a, b = 1.0, 0.8   # pinned moderate cancellation (see build_forms note)
        cAs.append(a); cBs.append(-b if coeff_signs == "oppose" else b)
    nrows, nbias, ncoef = [], [], []
    for _ in range(n_normal):
        w = torch.tensor(rng.standard_normal(W1_N), dtype=torch.float32)
        z = H1 @ w; bz = -float(z.median())
        nrows.append(w); nbias.append(bz); ncoef.append(float(rng.uniform(-1, 1)))

    # BASELINE (Sequential): rows for z AND -z per pair, both relu'd.
    W2b, b2b, c3b = [], [], []
    for p in range(n_pairs):
        W2b += [rows[p], -rows[p]]; b2b += [biases[p], -biases[p]]; c3b += [cAs[p], cBs[p]]
    for q in range(n_normal):
        W2b += [nrows[q]]; b2b += [nbias[q]]; c3b += [ncoef[q]]
    mb = len(W2b)
    l1 = nn.Linear(D_IN, W1_N); l1.weight.data = W1.clone(); l1.bias.data = b1.clone()
    l2 = nn.Linear(W1_N, mb); l2.weight.data = torch.stack(W2b); l2.bias.data = torch.tensor(b2b)
    l3 = nn.Linear(mb, 1, bias=False); l3.weight.data = torch.tensor(c3b).view(1, mb)
    baseline = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3)

    # COLLAPSED: one relu(z) per pair (coeff cA+cB) + skip carrying -cB*z = -cB*(w.h1+bz).
    W2m, b2m, c3m = [], [], []
    skipW = torch.zeros(1, W1_N); skipb = torch.zeros(1)
    for p in range(n_pairs):
        W2m += [rows[p]]; b2m += [biases[p]]; c3m += [cAs[p] + cBs[p]]
        skipW += (-cBs[p]) * rows[p].view(1, -1); skipb += (-cBs[p]) * biases[p]
    for q in range(n_normal):
        W2m += [nrows[q]]; b2m += [nbias[q]]; c3m += [ncoef[q]]
    collapsed = CollapsedNet(
        W1.clone(), b1.clone(), torch.stack(W2m), torch.tensor(b2m),
        torch.tensor(c3m).view(1, len(c3m)), skipW, skipb)
    return baseline, collapsed, gaps


def run_complementary(tag, coeff_signs, n_pairs=4):
    print(f"\n===== {tag} (n_pairs={n_pairs}, coeffs={coeff_signs}) =====")
    base, coll, gaps = build_complementary(n_pairs, coeff_signs)
    print(f"planted gaps: {[round(g,2) for g in gaps]}")
    xs = torch.randn(50, D_IN)
    assert torch.allclose(base(xs), coll(xs), atol=1e-4), "complementary forms not identical!"
    print(f"function-identical on 50 samples: OK  (baseline relu-layer2 width="
          f"{base[2].out_features}, collapsed width={coll.l2.out_features}+skip)")
    for method in ["IBP", "CROWN", "CROWN-Optimized"]:
        lb, ub = bounds(base, method); lc, uc = bounds(coll, method)
        wb, wc = ub - lb, uc - lc
        rel = (wb - wc) / abs(wb) * 100 if wb else 0.0
        verdict = ("COLLAPSED TIGHTER" if wc < wb - 1e-6 else
                   "neutral" if abs(wc - wb) <= 1e-6 else "COLLAPSED LOOSER")
        print(f"  {method:16s} baseline w={wb:8.4f} | collapsed w={wc:8.4f} | "
              f"delta={wb-wc:+.4f} ({rel:+.1f}%) -> {verdict}")


if __name__ == "__main__":
    print("### RULE 1: merge duplicated proportional neurons ###")
    run("OPPOSITE-SIGN redundancy (the rewrite)", "oppose")
    run("SAME-SIGN redundancy (control -- must be neutral)", "same")
    print("\n### RULE 2: collapse complementary relu(z)+relu(-z) pairs ###")
    run_complementary("OPPOSITE-SIGN complementary (the rewrite)", "oppose")
    run_complementary("SAME-SIGN complementary (control -- must be neutral)", "same")
