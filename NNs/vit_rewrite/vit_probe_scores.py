#!/usr/bin/env python
"""Probe the pre-softmax attention-score magnitudes of the VNN-COMP'23 ViT on real instance
centers, to check whether deleting the numerical-stability max-shift (softmax shift-invariance
rewrite R1) is float-safe (exp overflows at ~88 in fp32). Also reports per-instance eps."""
import sys, re, glob, os, numpy as np, onnx
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
BENCH=os.path.join(REPO,"vnncomp2023_benchmarks/benchmarks/vit")
model_name=sys.argv[1] if len(sys.argv)>1 else "pgd_2_3_16"
m=onnx.load(os.path.join(BENCH,"onnx",model_name+".onnx"))
# expose every pre-softmax tensor (input of each Softmax node) as an extra graph output
sm_in=[n.input[0] for n in m.graph.node if n.op_type=="Softmax"]
for t in sm_in:
    m.graph.output.append(onnx.helper.make_tensor_value_info(t,onnx.TensorProto.FLOAT,None))
import onnxruntime as ort
sess=ort.InferenceSession(m.SerializeToString(),providers=["CPUExecutionProvider"])
inp=sess.get_inputs()[0].name

def parse_vnnlib(path):
    lb={}; ub={}
    for line in open(path):
        mo=re.match(r"\(assert \((<=|>=) X_(\d+) ([-\d.eE]+)\)\)",line.strip())
        if mo:
            op,i,v=mo.group(1),int(mo.group(2)),float(mo.group(3))
            (ub if op=="<=" else lb)[i]=v
    n=max(lb)+1
    l=np.array([lb[i] for i in range(n)],dtype=np.float32); u=np.array([ub[i] for i in range(n)],dtype=np.float32)
    return l,u

files=sorted(glob.glob(os.path.join(BENCH,"vnnlib",model_name+"_*.vnnlib")))
print(f"# {model_name}: {len(files)} instances; softmax inputs exposed: {len(sm_in)}")
gmin,gmax=1e9,-1e9; epss=[]
for f in files:
    l,u=parse_vnnlib(f); c=((l+u)/2).reshape(1,3,32,32); eps=float((u-l).max()/2); epss.append(eps)
    outs=sess.run(None,{inp:c})
    scores=np.concatenate([o.ravel() for o in outs[1:]])
    gmin=min(gmin,scores.min()); gmax=max(gmax,scores.max())
print(f"# eps: min={min(epss):.5f} max={max(epss):.5f} unique={sorted(set(round(e,5) for e in epss))}")
print(f"# pre-softmax scores over all centers: min={gmin:.3f} max={gmax:.3f}  (fp32 exp overflow at ~88)")
print(f"# logits check on first instance: {outs[0].ravel()}")
