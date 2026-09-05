#!/usr/bin/env python
"""
Weight-only patch: write the gauge-transformed attention weights INTO THE STOCK ONNX GRAPH (identical structure,
opset, node order, names; only the initializer VALUES of the per-layer query/key/value/out MatMul weights and
biases change). This removes the export-path confound of vit_export.py (which re-exports the PyTorch
re-implementation with a different opset and shape-op structure) from the official-pipeline comparison.

  python vit_patch_onnx.py --model pgd_2_3_16 --gauge_file gauges/pgd_mix_svdinit.pt --name learnedG_patched
"""
import sys, os, argparse, numpy as np, torch, onnx
from onnx import numpy_helper
REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
sys.path.insert(0, os.path.join(REPO, "NNs/vit_rewrite"))
from vit_model import ViT, VARIANTS
from vit_bounds import centers
BENCH = os.path.join(REPO, "vnncomp2023_benchmarks/benchmarks")

def patch(model, gauge_file=None, variant="base", name=None):
    import onnxruntime as ort
    onnx_path = os.path.join(BENCH, "vit/onnx", model + ".onnx")
    kw = dict(VARIANTS[variant])
    if gauge_file: Gs = torch.load(gauge_file); kw.update(qk_gauge=Gs.get("qk"), av_gauge=Gs.get("av"))
    net = ViT(onnx_path, **kw).eval()
    m = onnx.load(onnx_path); g = m.graph; inits = {x.name: x for x in g.initializer}; nodes = {n.name: n for n in g.node}
    n_changed = 0; max_rel = 0.0
    for l, blk in enumerate(net.blocks):
        p = f"/1/1.{l}/1.{l}.0/fn/fn.1"; b = f"1.{l}.0.fn.1"; at = blk.attn
        for r, W, bias in zip(("query", "key", "value", "out"), (at.Wq, at.Wk, at.Wv, at.Wo), (at.bq, at.bk, at.bv, at.bo)):
            for tname, t in ((nodes[f"{p}/{r}/MatMul"].input[1], W), (f"{b}.{r}.bias", bias)):
                old = numpy_helper.to_array(inits[tname]); new = t.detach().cpu().numpy().astype(old.dtype)
                assert old.shape == new.shape, (tname, old.shape, new.shape)
                max_rel = max(max_rel, float(np.abs(new - old).max() / (np.abs(old).max() + 1e-12)))
                if not np.array_equal(old, new): n_changed += 1
                inits[tname].CopyFrom(numpy_helper.from_array(new, tname))
    onnx.checker.check_model(m)
    name = name or (variant + ("__G_" + os.path.splitext(os.path.basename(gauge_file))[0] if gauge_file else "")) + "_patched"
    out_dir = os.path.join(BENCH, f"vit_{name}"); os.makedirs(os.path.join(out_dir, "onnx"), exist_ok=True)
    if not os.path.exists(os.path.join(out_dir, "vnnlib")): os.symlink(os.path.join(BENCH, "vit/vnnlib"), os.path.join(out_dir, "vnnlib"))
    out_onnx = os.path.join(out_dir, "onnx", f"{model}.onnx"); onnx.save(m, out_onnx)
    rows = [l_ for l_ in open(os.path.join(BENCH, "vit/instances.csv")) if l_.startswith(f"onnx/{model}.onnx")]
    with open(os.path.join(out_dir, "instances.csv"), "w") as f: f.writelines(rows)
    # structural identity check + numerical check vs stock on all instance centers
    g2 = onnx.load(out_onnx).graph; g1 = onnx.load(onnx_path).graph
    same_struct = [(n.op_type, n.name, list(n.input), list(n.output)) for n in g1.node] == [(n.op_type, n.name, list(n.input), list(n.output)) for n in g2.node]
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s0 = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"]); s1 = ort.InferenceSession(out_onnx, so, providers=["CPUExecutionProvider"])
    xs = centers(model, "cpu").numpy(); md = 0.0
    for i in range(len(xs)):
        a = s0.run(None, {s0.get_inputs()[0].name: xs[i:i + 1]})[0]; b_ = s1.run(None, {s1.get_inputs()[0].name: xs[i:i + 1]})[0]
        md = max(md, float(np.abs(a - b_).max()))
    print(f"# patched {out_onnx}\n#   graph structure identical to stock: {same_struct}; initializers changed: {n_changed}/{2 * 4 * len(net.blocks)} (max rel change {max_rel:.3f})"
          f"\n#   max|stock-patched| over {len(xs)} centers = {md:.3e}\n#   {len(rows)} instances -> {out_dir}/instances.csv", flush=True)
    return out_dir, md

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="pgd_2_3_16"); ap.add_argument("--gauge_file", default=None)
    ap.add_argument("--variant", default="base"); ap.add_argument("--name", default=None)
    a = ap.parse_args(); patch(a.model, a.gauge_file, a.variant, a.name)
