#!/usr/bin/env python
"""
Export a rewritten ViT variant to ONNX so the UNMODIFIED official alpha-beta-CROWN pipeline (vnncomp23/vit.yaml:
complex softmax + alpha-CROWN + BaB, 100s timeout) can run it drop-in against the identical vnnlib specs.
Builds a benchmark dir  vnncomp2023_benchmarks/benchmarks/vit_<variant>/  with the exported onnx, a symlinked
vnnlib/, and an instances.csv rewritten to point at the new onnx. Validates exported-ONNX vs stock-ONNX on all
instance centers with onnxruntime.

  python vit_export.py --model pgd_2_3_16 --variant R45_both_svd
"""
import sys, os, re, glob, argparse, numpy as np, torch
REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
sys.path.insert(0, os.path.join(REPO, "NNs/vit_rewrite"))
from vit_model import ViT, VARIANTS
from vit_bounds import parse_vnnlib, instance_files, centers
BENCH = os.path.join(REPO, "vnncomp2023_benchmarks/benchmarks")

def export(model, variant, G_path=None, name=None):
    import onnxruntime as ort
    onnx_path = os.path.join(BENCH, "vit/onnx", model + ".onnx")
    kw = dict(VARIANTS[variant])
    if G_path:  # learned gauges saved by vit_gauge_opt.py: dict(qk=(L,H,dh,dh), av=(L,H,dh,dh)); applied on top of --variant
        Gs = torch.load(G_path); kw.update(qk_gauge=Gs.get("qk"), av_gauge=Gs.get("av"))
    net = ViT(onnx_path, **kw).eval()
    if variant == "R1_rowmean": net.set_shift_from_centers(centers(model, "cpu"))
    name = name or (variant + ("__G_" + os.path.splitext(os.path.basename(G_path))[0] if G_path else ""))
    out_dir = os.path.join(BENCH, f"vit_{name}"); os.makedirs(os.path.join(out_dir, "onnx"), exist_ok=True)
    if not os.path.exists(os.path.join(out_dir, "vnnlib")): os.symlink(os.path.join(BENCH, "vit/vnnlib"), os.path.join(out_dir, "vnnlib"))
    out_onnx = os.path.join(out_dir, "onnx", f"{model}.onnx")
    x = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(net, x, out_onnx, input_names=["input"], output_names=["output"], opset_version=14,
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, do_constant_folding=True, dynamo=False)
    # onnx2pytorch (the official loader) cannot parse BatchNormalization.training_mode (opset>=14): strip it like the stock file
    import onnx
    m = onnx.load(out_onnx)
    for n in m.graph.node:
        if n.op_type == "BatchNormalization":
            for att in [att for att in n.attribute if att.name in ("training_mode",)]: n.attribute.remove(att)
    onnx.checker.check_model(m); onnx.save(m, out_onnx)
    # instances.csv: only this model's rows, pointing at the new onnx (same vnnlibs, same timeouts)
    rows = [l for l in open(os.path.join(BENCH, "vit/instances.csv")) if l.startswith(f"onnx/{model}.onnx")]
    with open(os.path.join(out_dir, "instances.csv"), "w") as f: f.writelines(rows)
    # validate vs stock onnx on all instance centers
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s0 = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"]); s1 = ort.InferenceSession(out_onnx, so, providers=["CPUExecutionProvider"])
    xs = centers(model, "cpu").numpy(); md = 0.0
    for i in range(len(xs)):
        a = s0.run(None, {s0.get_inputs()[0].name: xs[i:i+1]})[0]; b = s1.run(None, {s1.get_inputs()[0].name: xs[i:i+1]})[0]
        md = max(md, float(np.abs(a - b).max()))
    import onnx; g = onnx.load(out_onnx).graph
    from collections import Counter
    print(f"# exported {out_onnx}\n#   ops={dict(Counter(n.op_type for n in g.node))}\n#   max|stock-exported| over {len(xs)} centers = {md:.3e}\n#   {len(rows)} instances -> {out_dir}/instances.csv")
    return out_dir, md

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="pgd_2_3_16"); ap.add_argument("--variant", default="R45_both_svd"); ap.add_argument("--name", default=None)
    ap.add_argument("--gauge_file", default=None, help="learned gauges .pt from vit_gauge_opt.py (applied on top of --variant)")
    a = ap.parse_args(); export(a.model, a.variant, G_path=a.gauge_file, name=a.name)
