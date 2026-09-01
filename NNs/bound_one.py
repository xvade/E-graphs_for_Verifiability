# Bound a SINGLE reconstructed form with auto_LiRPA/CROWN and print its
# certified upper bound. Args: <onnx_path> <wbx.npz (x0, eps)> <ref_onnx>.
# Patches onnx2pytorch's Add.forward to accept variadic (bias-less) adds, then
# converts the ONNX to torch and runs an Lp-norm CROWN bound around x0+-eps.
# The per-form primitive under bound_forms.py; also usable standalone.
import sys, numpy as np, torch, onnx, onnx2pytorch, functools as ft
sys.path[:0]=["NNs","NNs/reassoc_results","alpha-beta-CROWN/complete_verifier"]
import maxtree_bounds as mtb
import onnx2pytorch.operations.add as am
o=am.Add.forward
am.Add.forward=lambda self,*i:(o(self,*i) if getattr(self,"input_indices",None) else ft.reduce(lambda a,b:a+b,i))
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
onnx_path, wbx, ref_onnx = sys.argv[1], sys.argv[2], sys.argv[3]
d=np.load(wbx); x0,eps=d["x0"],float(d["eps"]); dev="cuda" if torch.cuda.is_available() else "cpu"
x0t=torch.tensor(x0,dtype=torch.float32,device=dev).view(1,-1)
def load(p): return onnx2pytorch.ConvertModel(onnx.load(p),experimental=True).to(dev).eval()
def run1(m,X): return np.concatenate([m(torch.tensor(X[i:i+1],device=dev)).detach().cpu().numpy() for i in range(X.shape[0])],0)[:,0]
xs=(x0+np.random.default_rng(3).uniform(-eps,eps,size=(64,x0.shape[0]))).astype(np.float32)
ref=run1(load(ref_onnx),xs); m=load(onnx_path); out=run1(m,xs)
dmax=float(np.abs(out-ref).max())
bm=BoundedModule(m,x0t,device=dev,bound_opts={"conv_mode":"matrix"})
lb,ub=bm.compute_bounds(x=(BoundedTensor(x0t,PerturbationLpNorm(norm=np.inf,eps=eps)),),method="CROWN-Optimized")
nr,nu=mtb.count_unstable_relus(bm)
print(f"verif form: cert_ub={float(ub.flatten()[0]):.4f}  unstable={nu}/{nr}  numchk={dmax:.1e}")
