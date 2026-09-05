#!/usr/bin/env python
"""Dump the ONNX attributes needed for a faithful PyTorch reimplementation of the VNN-COMP'23 ViT:
scale constant, reshape target shapes, transpose perms, BN eps, softmax axis, ReduceMean axes,
conv attrs, concat order."""
import sys, os, onnx, numpy as np
from onnx import numpy_helper
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
name=sys.argv[1] if len(sys.argv)>1 else "pgd_2_3_16"
m=onnx.load(os.path.join(REPO,"vnncomp2023_benchmarks/benchmarks/vit/onnx",name+".onnx"))
g=m.graph; init={x.name:numpy_helper.to_array(x) for x in g.initializer}
def attrs(n): return {a.name:onnx.helper.get_attribute_value(a) for a in n.attribute}
# constants produced by Constant nodes
const={}
for n in g.node:
    if n.op_type=="Constant": const[n.output[0]]=numpy_helper.to_array(attrs(n)["value"])
print("### small initializers (shape/index constants) ###")
for k,v in init.items():
    if v.size<=8 and (v.ndim==0 or v.size>=1) and not k.startswith(("0.","1.","2.")): print(f"  {k}: {v.tolist()}")
print("\n### compute-node attributes ###")
for n in g.node:
    if n.op_type in ("Conv","BatchNormalization","Softmax","ReduceMean","Gemm","Transpose","Mul","Reshape","Concat","Split","Slice","Gather"):
        a=attrs(n)
        extra=""
        if n.op_type=="Mul":
            for i in n.input:
                if i in const: extra=f" CONST={const[i].tolist()}"
                if i in init: extra=f" INIT={init[i].tolist()}"
        if n.op_type=="Reshape":
            s=n.input[1]; extra=f" shape_src={s} {'CONST='+str(const[s].tolist()) if s in const else ''}{'INIT='+str(init[s].tolist()) if s in init else ''}"
        if n.op_type=="Concat":
            extra=f" inputs={list(n.input)}"
        if n.op_type=="Gather":
            extra=f" inputs={list(n.input)} " + " ".join(f"{i}={const[i].tolist()}" for i in n.input if i in const)
        if n.op_type=="Transpose":
            extra=f" in={n.input[0]}"
        print(f"  {n.op_type:20s} {n.name:50s} {a}{extra}")
