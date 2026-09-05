#!/usr/bin/env python
"""
Bound-measurement harness for the VNN-COMP'23 ViT rewrites (2026-09-04). Heavy: run on a compute node.

For a (model, variant, softmax-mode) it loads the faithful PyTorch ViT (vit_model.py) with the stock ONNX
weights, and for every benchmark instance (its exact vnnlib eps-box and 9 margin specs Y_label - Y_i)
computes incomplete certified LOWER bounds with auto_LiRPA under IBP / vanilla CROWN / alpha-CROWN
(CROWN-Optimized). Metric = per-spec lb (tighter is better) and the crisp count of instances where the
incomplete method alone certifies all 9 specs. Optional Monte-Carlo soundness check (sampled margins must
be >= lb). --diag runs the (inexact) slack-attribution linearizations; --width also computes upper bounds.
Results -> results/<model>__<variant>__<softmax>[__tag].npz + .json

  python vit_bounds.py --model pgd_2_3_16 --variant base --softmax lse --methods CROWN,CROWN-Optimized
"""
import sys, os, re, glob, json, time, argparse, numpy as np, torch
REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
sys.path.insert(0, os.path.join(REPO, "alpha-beta-CROWN/complete_verifier")); sys.path.insert(0, os.path.join(REPO, "NNs/vit_rewrite"))
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from vit_model import ViT, VARIANTS
BENCH = os.path.join(REPO, "vnncomp2023_benchmarks/benchmarks/vit")

def parse_vnnlib(path):
    lb, ub, rhs = {}, {}, []
    for line in open(path):
        s = line.strip()
        mo = re.match(r"\(assert \((<=|>=) X_(\d+) ([-\d.eE]+)\)\)", s)
        if mo: (ub if mo.group(1) == "<=" else lb)[int(mo.group(2))] = float(mo.group(3)); continue
        mo = re.search(r"\(>= Y_(\d+) Y_(\d+)\)", s)
        if mo: rhs.append(int(mo.group(2)))
    n = max(lb) + 1
    assert len(set(rhs)) == 1, f"unexpected output spec in {path}"
    return (np.array([lb[i] for i in range(n)], np.float32), np.array([ub[i] for i in range(n)], np.float32), rhs[0])

def specs_C(label, n_out=10):
    C = torch.zeros(1, n_out - 1, n_out); r = 0
    for i in range(n_out):
        if i == label: continue
        C[0, r, label] = 1.0; C[0, r, i] = -1.0; r += 1
    return C

def instance_files(model, which):
    files = sorted(glob.glob(os.path.join(BENCH, "vnnlib", model + "_*.vnnlib")))
    if which != "all":
        ids = set(which.split(",")); files = [f for f in files if re.search(r"_(\d+)\.vnnlib$", f).group(1) in ids]
    return files

def centers(model, dev):
    xs = []
    for f in instance_files(model, "all"):
        l, u, _ = parse_vnnlib(f); xs.append(torch.tensor((l + u) / 2).reshape(3, 32, 32))
    return torch.stack(xs).to(dev)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pgd_2_3_16"); ap.add_argument("--variant", default="base")
    ap.add_argument("--softmax", default="lse", help="auto_LiRPA BoundSoftmax mode for 'native' softmax: lse|complex")
    ap.add_argument("--methods", default="CROWN,CROWN-Optimized"); ap.add_argument("--instances", default="all")
    ap.add_argument("--iters", type=int, default=50); ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--share_alphas", type=int, default=1); ap.add_argument("--mc", type=int, default=0, help="Monte-Carlo soundness samples per instance")
    ap.add_argument("--diag", default="", help="INEXACT slack-attribution linearizations: comma subset of linQK,linSM,linAV")
    ap.add_argument("--width", type=int, default=0, help="also compute upper bounds (width = ub - lb)")
    ap.add_argument("--gauge_file", default="", help=".pt with {'qk': (L,H,dh,dh), 'av': (L,H,dh,dh)} learned gauges (applied on top of --variant)")
    ap.add_argument("--out", default=os.path.join(REPO, "NNs/vit_rewrite/results")); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
    onnx_path = os.path.join(BENCH, "onnx", a.model + ".onnx"); methods = a.methods.split(",")
    files = instance_files(a.model, a.instances); os.makedirs(a.out, exist_ok=True)
    diag = [d for d in a.diag.split(",") if d]
    tag = f"{a.model}__{a.variant}__{a.softmax}" + ("__diag_" + "_".join(diag) if diag else "") + (("__" + a.tag) if a.tag else "")
    bound_opts = {"softmax": a.softmax, "conv_mode": "matrix", "disable_optimization": ["Exp"],  # mirrors official vit.yaml
                  "optimize_bound_args": {"iteration": a.iters, "lr_alpha": a.lr, "use_shared_alpha": bool(a.share_alphas),
                                          "early_stop_patience": 10, "lr_decay": 0.98}}
    kw = dict(VARIANTS[a.variant])
    if a.gauge_file:
        Gs = torch.load(a.gauge_file); kw.update(qk_gauge=Gs.get("qk"), av_gauge=Gs.get("av")); tag += "__G_" + os.path.splitext(os.path.basename(a.gauge_file))[0]
    net = ViT(onnx_path, **kw).to(dev)
    if a.variant == "R1_rowmean": net.set_shift_from_centers(centers(a.model, dev))
    if diag: net.set_diag(diag)
    print(f"# {tag}: {len(files)} instances, methods={methods}, meta={net.meta}, bound_opts={bound_opts}", flush=True)
    if net.gauge_info: print("# gauge: " + " | ".join(net.gauge_info), flush=True)
    LB = {m: np.full((len(files), 9), np.nan, np.float32) for m in methods}; UB = {m: np.full((len(files), 9), np.nan, np.float32) for m in methods}
    T = {m: np.zeros(len(files)) for m in methods}; labels = np.zeros(len(files), int); ids = []; mc_viol = {m: -np.inf for m in methods}
    for k, f in enumerate(files):
        xl, xu, y = parse_vnnlib(f); labels[k] = y; ids.append(re.search(r"_(\d+)\.vnnlib$", f).group(1))
        xl = torch.tensor(xl).reshape(1, 3, 32, 32).to(dev); xu = torch.tensor(xu).reshape(1, 3, 32, 32).to(dev); x0 = (xl + xu) / 2
        C = specs_C(y).to(dev)
        if diag: net.set_diag_constants(x0)
        for m in methods:
            t0 = time.time()
            try:
                lirpa = BoundedModule(net, torch.empty_like(x0), bound_opts=bound_opts, device=dev)  # fresh per instance: no alpha carry-over
                bx = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, x_L=xl, x_U=xu))
                with torch.no_grad() if m != "CROWN-Optimized" else torch.enable_grad():
                    lb, ub = lirpa.compute_bounds(x=(bx,), method=m, C=C, bound_lower=True, bound_upper=bool(a.width))
                LB[m][k] = lb.detach().cpu().numpy().ravel()
                if a.width: UB[m][k] = ub.detach().cpu().numpy().ravel()
                if a.mc and not diag:
                    with torch.no_grad():
                        xs = xl + (xu - xl) * torch.rand(a.mc, 3, 32, 32, device=dev)
                        marg = (C[0] @ net(xs).t()).t()
                        mc_viol[m] = max(mc_viol[m], float((lb.detach() - marg).max().item()))  # >0 would be UNSOUND
            except Exception as e:
                print(f"  !! {m} failed on id={ids[-1]}: {type(e).__name__}: {str(e)[:140]}", flush=True)
            T[m][k] = time.time() - t0
        line = f"  [{k+1:3d}/{len(files)}] id={ids[-1]} y={y} "
        for m in methods:
            line += f"{m}: min_lb={np.nanmin(LB[m][k]):+.4f} ver={int((LB[m][k] > 0).all())}"
            if a.width: line += f" width={np.nanmean(UB[m][k]-LB[m][k]):.4f}"
            line += f" ({T[m][k]:.1f}s)  "
        print(line, flush=True)
    summ = {"tag": tag, "model": a.model, "variant": a.variant, "softmax": a.softmax, "diag": diag, "n": len(files), "bound_opts": str(bound_opts), "methods": {}}
    for m in methods:
        ok = ~np.isnan(LB[m]).any(1)
        summ["methods"][m] = {"n_ok": int(ok.sum()), "n_verified_all9": int((LB[m][ok] > 0).all(1).sum()), "mean_lb": float(np.nanmean(LB[m])),
                              "mean_min_lb": float(np.nanmean(np.nanmin(LB[m], 1))), "median_min_lb": float(np.nanmedian(np.nanmin(LB[m], 1))),
                              "mean_width": float(np.nanmean(UB[m] - LB[m])) if a.width else None,
                              "mean_time_s": float(T[m].mean()), "mc_max_violation": (mc_viol[m] if a.mc and not diag else None)}
    np.savez(os.path.join(a.out, tag + ".npz"), **{f"lb_{m}": LB[m] for m in methods}, **{f"ub_{m}": UB[m] for m in methods},
             **{f"t_{m}": T[m] for m in methods}, labels=labels, ids=np.array(ids))
    json.dump(summ, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    print("# SUMMARY " + json.dumps(summ["methods"]), flush=True)

if __name__ == "__main__":
    main()
