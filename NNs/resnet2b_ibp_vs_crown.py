#!/usr/bin/env python
"""
resnet2b: IBP vs CROWN vs CROWN-Optimized, with a semantics-preserving rewrite.

Purpose (2026-09-04). Two questions, both grounded in measured OUTPUT-INTERVAL WIDTHS
(not just verified/not):

  Q1 (baseline / vacuity hypothesis): on stock resnet2b (standard-trained, non-IBP),
      IBP width is expected to blow up ~||W||_1 per layer over 7 linear layers, so IBP
      should be VACUOUS at any eps where CROWN is informative. => the bonus targets
      "IBP(rewritten) beats CROWN(original)" and "... beats CROWN(rewritten)" are
      expected UNREACHABLE for a reason independent of any rewrite. This harness turns
      that expectation into numbers.

  Q2 (the one exact structural rewrite resnet2b's residuals admit): block B's identity
      shortcut can be eliminated exactly. z (block-A output) is a ReLU output => z>=0,
      so  conv2_B(relu(conv1_B(z))) + z  ==  wideconv( concat[ relu(conv1_B(z)), z ] )
      with wideconv = [conv2_B | I] (identity kernel on the z half). This removes the
      residual Add, giving a plain conv-relu-conv-relu block. Expected IBP-NEUTRAL
      (IBP already does optimal interval addition on the Add) and CROWN-neutral (the
      identity-routed channels are stably active). We MEASURE the delta.

  Block A's shortcut is a mixed-sign CONV (not a ReLU output), so it is NOT eliminable
  without introducing extra ReLUs (which is strictly IBP-looser); left as an Add.

The real IBP-improvement demonstration (a rewrite that DOES tighten IBP ~37%) lives on a
net CONSTRUCTED with a mixed-sign consecutive-linear fold site that resnet2b lacks:
NNs/reassoc_results/plain_relu_more_verifiable.py . resnet2b has no such site.

HEAVY-COMPUTE NOTE: IBP and plain CROWN are cheap (a few passes). CROWN-Optimized runs
alpha optimization over ~6244 neurons and is the heavy part -- gate it behind --full and
run that on a compute node, not the login node.

Usage:
  python resnet2b_ibp_vs_crown.py                 # IBP + CROWN, 3 images, eps in {2/255, 8/255}
  python resnet2b_ibp_vs_crown.py --full          # also CROWN-Optimized (COMPUTE NODE)
  python resnet2b_ibp_vs_crown.py --n 5 --eps 0.00784313725490196
"""
import argparse, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CV = os.path.join(REPO, "alpha-beta-CROWN", "complete_verifier")
sys.path.insert(0, CV)
sys.path.insert(0, os.path.join(CV, "auto_LiRPA"))  # ensure auto_LiRPA importable

from model_defs import resnet2b                         # noqa: E402
from auto_LiRPA import BoundedModule, BoundedTensor      # noqa: E402
from auto_LiRPA.perturbations import PerturbationLpNorm  # noqa: E402

torch.set_num_threads(4)
CKPT = os.path.join(CV, "models", "cifar10_resnet", "resnet2b.pth")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------------------
# Q2 rewrite: block-B identity-shortcut elimination via a wide [conv2_B | I] conv.
# We rebuild the residual-free variant on top of the SAME trained weights and verify
# numeric equivalence to the stock model before any bound comparison.
# --------------------------------------------------------------------------------------
class BlockAResidual(nn.Module):
    """Block A, unchanged: conv-relu-conv + mixed-sign conv shortcut, then relu."""
    def __init__(self, blk):
        super().__init__()
        self.conv1, self.conv2 = blk.conv1, blk.conv2
        self.shortcut = blk.shortcut  # nn.Sequential(Conv 1x1 s2)

    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
        out = out + self.shortcut(x)
        return F.relu(out)


class BlockBResFree(nn.Module):
    """Block B with the identity residual folded into a wide conv [conv2 | I]."""
    def __init__(self, blk):
        super().__init__()
        self.conv1 = blk.conv1
        c2 = blk.conv2                       # Conv(16->16, k3, s1, p1)
        cout, cin, kh, kw = c2.weight.shape  # 16,16,3,3
        assert cout == cin and kh == kw and kh % 2 == 1
        # identity kernel: channel i -> i, center tap = 1
        idk = torch.zeros(cout, cin, kh, kw)
        for i in range(cout):
            idk[i, i, kh // 2, kw // 2] = 1.0
        wide = nn.Conv2d(cin * 2, cout, kernel_size=kh, stride=c2.stride,
                         padding=c2.padding, bias=True)
        with torch.no_grad():
            wide.weight.copy_(torch.cat([c2.weight, idk], dim=1))  # (16, 32, 3,3)
            wide.bias.copy_(c2.bias)
        self.wide = wide

    def forward(self, z):
        h = F.relu(self.conv1(z))            # >= 0
        cat = torch.cat([h, z], dim=1)       # z is a relu output upstream => >= 0
        return F.relu(self.wide(cat))        # == relu(conv2(h) + z)


class Resnet2bResFree(nn.Module):
    """resnet2b with block B's identity residual eliminated (semantics-preserving)."""
    def __init__(self, base):
        super().__init__()
        self.conv1 = base.conv1
        self.blockA = BlockAResidual(base.layer1[0])
        self.blockB = BlockBResFree(base.layer1[1])
        self.linear1, self.linear2 = base.linear1, base.linear2

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.blockA(out)
        out = self.blockB(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        return self.linear2(out)


def load_models():
    base = resnet2b()
    ckpt = torch.load(CKPT, map_location="cpu")
    base.load_state_dict(ckpt["state_dict"])
    base.eval()
    rf = Resnet2bResFree(base).eval()
    return base, rf


def get_images(n):
    """A few normalized CIFAR-ish inputs. Real test images if torchvision has them
    cached; otherwise deterministic pseudo-images (widths are the diagnostic, and we use
    the clean argmax as a pseudo-label for the robustness margin)."""
    try:
        import torchvision
        ds = torchvision.datasets.CIFAR10(
            root=os.path.join(REPO, "data"), train=False, download=False)
        xs, ys = [], []
        for i in range(n):
            img, y = ds[i]
            xs.append(torch.tensor(list(img.getdata()), dtype=torch.float32)
                      .view(32, 32, 3).permute(2, 0, 1) / 255.0)
            ys.append(y)
        x = torch.stack(xs)
        src = "CIFAR10 test"
    except Exception as e:
        g = torch.Generator().manual_seed(0)
        x = torch.rand(n, 3, 32, 32, generator=g)
        ys = None
        src = f"pseudo-random (CIFAR unavailable: {type(e).__name__})"
    xn = (x - MEAN) / STD
    return xn, ys, src


def out_widths(bm, x, eps, method):
    """Return (mean_width, max_width, min_true_margin) over the 10 logits for one image.
    eps is in normalized input space (per-channel eps/std)."""
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps, x_L=None, x_U=None)
    bx = BoundedTensor(x, ptb)
    lb, ub = bm.compute_bounds(x=(bx,), method=method)
    w = (ub - lb)[0]
    return w.mean().item(), w.max().item(), lb, ub


def eps_normalized(eps_pixel):
    # Linf eps in [0,1] pixel space becomes per-channel eps/std after normalization.
    return (eps_pixel / STD).view(-1)  # 3 channel-specific values; use max for a scalar box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--eps", type=float, nargs="*",
                    default=[2 / 255, 8 / 255])
    ap.add_argument("--full", action="store_true",
                    help="also run CROWN-Optimized (heavy; compute node)")
    args = ap.parse_args()

    base, rf = load_models()
    x, ys, src = get_images(args.n)
    print(f"# images: {src}; n={args.n}")

    # ---- Q2 numeric equivalence: rewrite must match stock to fp tolerance ----
    with torch.no_grad():
        d = (base(x) - rf(x)).abs().max().item()
    print(f"# Q2 rewrite equivalence: max|base-resfree| = {d:.3e}  "
          f"({'OK' if d < 1e-4 else 'MISMATCH -- FIX'})")

    methods = ["IBP", "CROWN"] + (["CROWN-Optimized"] if args.full else [])
    print(f"# methods: {methods}")
    print(f"# NOTE: CROWN-Optimized is heavy; run --full on a compute node.\n")

    # Build BoundedModules once per model (per-image bounds reuse them).
    dummy = x[:1]
    bm_base = BoundedModule(base, dummy, device="cpu")
    bm_rf = BoundedModule(rf, dummy, device="cpu")

    for ep_pix in args.eps:
        ep = eps_normalized(ep_pix).max().item()  # scalar box (loosest channel)
        print(f"=== eps_pixel={ep_pix:.5f}  (normalized box eps={ep:.4f}) ===")
        for name, bm in [("orig", bm_base), ("resfree", bm_rf)]:
            for m in methods:
                mw = xw = 0.0
                for i in range(args.n):
                    a, b, _, _ = out_widths(bm, x[i:i + 1], ep, m)
                    mw += a; xw = max(xw, b)
                mw /= args.n
                print(f"  {name:8s} {m:16s} mean_width={mw:12.3f}  max_width={xw:12.3f}")
        print()

    print("# INTERPRETATION:")
    print("#  Q1: if IBP mean_width >> CROWN mean_width (orders of magnitude), IBP is")
    print("#      vacuous on resnet2b -> bonus/double-bonus (IBP beats CROWN) unreachable.")
    print("#  Q2: orig vs resfree widths equal under IBP (and CROWN) => residual")
    print("#      elimination is verifiability-NEUTRAL, as predicted.")


if __name__ == "__main__":
    main()
