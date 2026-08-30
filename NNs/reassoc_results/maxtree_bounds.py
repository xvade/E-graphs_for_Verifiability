#!/usr/bin/env python3
"""Light PoC: does re-associating a max-reduction ReLU tree change certified
bound tightness (verifiability), holding the FUNCTION and the ReLU COUNT fixed?

max(u,v) = u + relu(v-u).  A max over N affine functions is a binary tree of
N-1 such nodes REGARDLESS of tree shape -> same #relus, different depth/topology.
  chain    : depth N-1  (running accumulator)
  balanced : depth ceil(log2 N)

We build both as pure-ReLU torch modules (auto_LiRPA-boundable), check they are
numerically identical to the true max, then compute the certified UPPER bound on
the scalar output over an input box with alpha-CROWN. Tighter (smaller) ub = more
verifiable. Prediction: balanced <= chain (tighter), untuned, over a distribution.
"""
import sys, argparse, math
import numpy as np
import torch, torch.nn as nn

def relu(t): return torch.relu(t)

class MaxTree(nn.Module):
    """g(x) = max_i (W x + b)_i, realized as a pure-ReLU binary max tree."""
    def __init__(self, W, b, shape):
        super().__init__()
        self.lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
            self.lin.bias.copy_(torch.tensor(b, dtype=torch.float32))
        self.shape = shape  # 'chain' or 'balanced'
        self.N = W.shape[0]

    def _maxpair(self, u, v):
        return u + relu(v - u)

    def forward(self, x):
        z = self.lin(x)                      # [batch, N]
        cols = [z[:, i:i+1] for i in range(self.N)]
        if self.shape == 'chain':
            acc = cols[0]
            for i in range(1, self.N):
                acc = self._maxpair(acc, cols[i])
            return acc
        else:  # balanced
            level = cols
            while len(level) > 1:
                nxt = []
                for i in range(0, len(level) - 1, 2):
                    nxt.append(self._maxpair(level[i], level[i+1]))
                if len(level) % 2 == 1:
                    nxt.append(level[-1])
                level = nxt
            return level[0]

class MinMaxLattice(nn.Module):
    """g(x) = min_g max_k (W x + b)_{g,k}  -- a two-level lattice (tll-shaped),
    realized pure-ReLU. Both the inner max-reduction (per group) and the outer
    min-reduction can be chain or balanced; we set both from `shape`.
    max(u,v)=u+relu(v-u);  min(u,v)=u-relu(u-v)."""
    def __init__(self, W, b, G, K, shape):
        super().__init__()
        self.lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
            self.lin.bias.copy_(torch.tensor(b, dtype=torch.float32))
        self.G, self.K, self.shape = G, K, shape

    def _reduce(self, cols, op):
        f = (lambda u, v: u + relu(v - u)) if op == 'max' else (lambda u, v: u - relu(u - v))
        if self.shape == 'chain':
            acc = cols[0]
            for c in cols[1:]:
                acc = f(acc, c)
            return acc
        level = list(cols)
        while len(level) > 1:
            nxt = [f(level[i], level[i+1]) for i in range(0, len(level)-1, 2)]
            if len(level) % 2 == 1:
                nxt.append(level[-1])
            level = nxt
        return level[0]

    def forward(self, x):
        z = self.lin(x)  # [batch, G*K]
        group_max = []
        for g in range(self.G):
            cols = [z[:, g*self.K + k:g*self.K + k + 1] for k in range(self.K)]
            group_max.append(self._reduce(cols, 'max'))
        return self._reduce(group_max, 'min')

def true_lattice(W, b, x, G, K):
    z = x @ W.T + b                    # [n, G*K]
    z = z.reshape(z.shape[0], G, K)
    return z.max(axis=2).min(axis=1, keepdims=True)

def true_max(W, b, x):  # reference
    return (x @ W.T + b).max(axis=1, keepdims=True)

def count_unstable_relus(bm):
    """After compute_bounds, count ReLU nodes whose pre-activation straddles 0
    (lb<0<ub). Same total #relus in chain vs balanced -> this isolates how
    reassociation changes STABILITY, the driver of bound looseness & BaB size."""
    n_relu = n_unstable = 0
    for m in bm.modules():
        if type(m).__name__ == 'BoundRelu':
            pre = m.inputs[0]
            l = getattr(pre, 'lower', None); u = getattr(pre, 'upper', None)
            if l is None or u is None:
                continue
            l = l.flatten(); u = u.flatten()
            n_relu += l.numel()
            n_unstable += int(((l < -1e-9) & (u > 1e-9)).sum().item())
    return n_relu, n_unstable

def certified_ub(model, x0, eps, method, iters=None, patience=None):
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    dev = x0.device
    bopts = {'conv_mode': 'matrix'}
    if iters is not None:  # override alpha-CROWN optimization budget
        bopts['optimize_bound_args'] = {'iteration': iters,
                                        'early_stop_patience': patience or iters}
    bm = BoundedModule(model, torch.zeros_like(x0), device=str(dev),
                       bound_opts=bopts)
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    bx = BoundedTensor(x0, ptb)
    lb, ub = bm.compute_bounds(x=(bx,), method=method)
    nr, nu = count_unstable_relus(bm)
    return float(lb.item()), float(ub.item()), nr, nu

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=16)
    ap.add_argument('--d', type=int, default=8)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--eps', type=float, default=0.5)
    ap.add_argument('--method', default='CROWN-Optimized')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--iters', type=int, default=None, help='alpha-CROWN opt iteration budget (default: auto)')
    ap.add_argument('--patience', type=int, default=None, help='early-stop patience (default: =iters)')
    ap.add_argument('--lattice', action='store_true', help='min-of-max two-level lattice (tll-shaped)')
    ap.add_argument('--G', type=int, default=4, help='#groups (lattice)')
    ap.add_argument('--K', type=int, default=4, help='#members/group (lattice)')
    args = ap.parse_args()
    if args.lattice:
        args.N = args.G * args.K
    dev = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"# N={args.N} d={args.d} reps={args.reps} eps={args.eps} "
          f"method={args.method} dev={dev}")
    print(f"# depth chain={args.N-1} balanced={math.ceil(math.log2(args.N))} "
          f"(#relus identical = {args.N-1})")
    rows = []
    for r in range(args.reps):
        W = rng.standard_normal((args.N, args.d)).astype(np.float32)
        b = rng.standard_normal(args.N).astype(np.float32)
        x0c = rng.standard_normal(args.d).astype(np.float32)
        x0 = torch.tensor(x0c, dtype=torch.float32, device=dev).unsqueeze(0)

        if args.lattice:
            mc = MinMaxLattice(W, b, args.G, args.K, 'chain').to(dev).eval()
            mb = MinMaxLattice(W, b, args.G, args.K, 'balanced').to(dev).eval()
            ref = lambda X: true_lattice(W, b, X, args.G, args.K)
        else:
            mc = MaxTree(W, b, 'chain').to(dev).eval()
            mb = MaxTree(W, b, 'balanced').to(dev).eval()
            ref = lambda X: true_max(W, b, X)
        # numeric equivalence sanity: both == true function on random samples in box
        xs = x0c + rng.uniform(-args.eps, args.eps, size=(64, args.d)).astype(np.float32)
        xt = torch.tensor(xs, device=dev)
        with torch.no_grad():
            oc = mc(xt).cpu().numpy(); ob = mb(xt).cpu().numpy()
        tm = ref(xs)
        dmax = max(float(np.abs(oc - tm).max()), float(np.abs(ob - tm).max()))
        assert dmax < 1e-4, f"rep{r}: not equal to true max, diff={dmax}"

        lb_c, ub_c, nr_c, nu_c = certified_ub(mc, x0, args.eps, args.method, args.iters, args.patience)
        lb_b, ub_b, nr_b, nu_b = certified_ub(mb, x0, args.eps, args.method, args.iters, args.patience)
        # concrete max over box (dense sample) as the tightest true reference
        big = x0c + rng.uniform(-args.eps, args.eps, size=(20000, args.d)).astype(np.float32)
        true_ub = float(ref(big).max())
        gap_c = ub_c - true_ub; gap_b = ub_b - true_ub
        rows.append((ub_c, ub_b, gap_c, gap_b, nu_c, nu_b, nr_c, nr_b))
        print(f"rep{r:02d}  ub_chain={ub_c:8.4f} ub_bal={ub_b:8.4f}  "
              f"gap_chain={gap_c:7.4f} gap_bal={gap_b:7.4f}  "
              f"unstable c/b={nu_c}/{nu_b} of {nr_c}  "
              f"chain_tighter={ub_c < ub_b - 1e-6}  (eq={dmax:.1e})")

    a = np.array(rows)
    ubc, ubb, gc, gb, nuc, nub = a[:,0], a[:,1], a[:,2], a[:,3], a[:,4], a[:,5]
    print("\n=== SUMMARY ===")
    print(f"chain tighter (ub_chain<ub_bal):    {int((ubc<ubb-1e-6).sum())}/{len(a)}")
    print(f"balanced tighter:                   {int((ubb<ubc-1e-6).sum())}/{len(a)}")
    print(f"mean gap-to-true  chain={gc.mean():.4f}  balanced={gb.mean():.4f}")
    print(f"mean ub           chain={ubc.mean():.4f}  balanced={ubb.mean():.4f}")
    print(f"mean (ub_bal-ub_chain) = {float((ubb-ubc).mean()):.4f}  "
          f"(>0 => chain tighter)")
    print(f"mean unstable relus  chain={nuc.mean():.2f}  balanced={nub.mean():.2f}  "
          f"(of {int(a[0,6])} total)")

if __name__ == '__main__':
    main()
