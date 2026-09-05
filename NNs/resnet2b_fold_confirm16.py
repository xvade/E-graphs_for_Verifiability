import sys, os, torch
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; sys.path.insert(0,"NNs")
sys.path.insert(0, os.path.join(REPO,"alpha-beta-CROWN","complete_verifier"))
sys.path.insert(0, os.path.join(REPO,"alpha-beta-CROWN","complete_verifier","auto_LiRPA"))
import importlib.util
spec=importlib.util.spec_from_file_location("rf","NNs/resnet2b_residual_fold.py")
# resnet2b_residual_fold.py runs measurement on import; avoid by reading classes via exec of a trimmed copy
import torch, torch.nn as nn, torch.nn.functional as F
from model_defs import resnet2b
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
torch.set_num_threads(4)
MEAN=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1); STD=torch.tensor([0.2471,0.2435,0.2616]).view(1,3,1,1)
CV=os.path.join(REPO,"alpha-beta-CROWN","complete_verifier")
base=resnet2b(); base.load_state_dict(torch.load(os.path.join(CV,"models/cifar10_resnet/resnet2b.pth"),map_location="cpu")["state_dict"]); base.eval()

def skeleton(blk,C,H,Wd):
    N=C*H*Wd
    def lin(sf):
        s=sf.view(-1,C,H,Wd); o=blk.conv2(blk.conv1(s))
        o=o+(blk.shortcut(s) if len(list(blk.shortcut.children())) else s); return o.flatten(1)
    with torch.no_grad():
        c=lin(torch.zeros(1,N))[0]; eye=torch.eye(N); cols=[lin(eye[i:i+512])-c for i in range(0,N,512)]
        return torch.cat(cols,0).t().contiguous(), c
class FB(nn.Module):
    def __init__(s,blk,ins,outs):
        super().__init__(); s.c1=blk.conv1; s.c2=blk.conv2; s.os=outs
        M,c=skeleton(blk,*ins); s.lin=nn.Linear(M.shape[1],M.shape[0])
        with torch.no_grad(): s.lin.weight.copy_(M); s.lin.bias.copy_(c)
    def forward(s,z):
        B=z.shape[0]; ex=F.conv2d(torch.relu(-s.c1(z)),s.c2.weight,None,s.c2.stride,s.c2.padding)
        return torch.relu(s.lin(z.flatten(1)).view(B,*s.os)+ex)
class OB(nn.Module):
    def __init__(s,blk): super().__init__(); s.c1=blk.conv1; s.c2=blk.conv2; s.sc=blk.shortcut
    def forward(s,z):
        o=s.c2(torch.relu(s.c1(z))); o=o+(s.sc(z) if len(list(s.sc.children())) else z); return torch.relu(o)
class Var(nn.Module):
    def __init__(s,foldA=False):
        super().__init__(); s.c=base.conv1
        s.bA=FB(base.layer1[0],(8,16,16),(16,8,8)) if foldA else OB(base.layer1[0])
        s.bB=OB(base.layer1[1]); s.l1=base.linear1; s.l2=base.linear2
    def forward(s,x):
        o=torch.relu(s.c(x)); o=s.bA(o); o=s.bB(o); o=o.view(o.size(0),-1); return s.l2(torch.relu(s.l1(o)))

orig=Var().eval(); foldA=Var(foldA=True).eval()
g=torch.Generator().manual_seed(7); X=(torch.rand(16,3,32,32,generator=g)-MEAN)/STD
eps=(2/255/STD).max().item()
bmo=BoundedModule(orig,X[:1]); bmf=BoundedModule(foldA,X[:1])
wins=0; deltas=[]
for i in range(16):
    bx=BoundedTensor(X[i:i+1],PerturbationLpNorm(norm=float('inf'),eps=eps))
    lo,uo=bmo.compute_bounds(x=(bx,),method="IBP"); lf,uf=bmf.compute_bounds(x=(bx,),method="IBP")
    wo=(uo-lo)[0].mean().item(); wf=(uf-lf)[0].mean().item(); d=100*(wf-wo)/wo; deltas.append(d)
    if wf<wo-1e-6: wins+=1
print(f"foldA vs orig, IBP mean width, 16 imgs @ eps_pix=2/255:")
print(f"  foldA tighter on {wins}/16 images; per-image Δ%: min={min(deltas):+.1f} max={max(deltas):+.1f} mean={sum(deltas)/16:+.2f}")
print(f"  verdict: {'ROBUST IMPROVEMENT' if wins>=14 and sum(deltas)/16<-0.5 else 'NOISE / no robust improvement'}")
