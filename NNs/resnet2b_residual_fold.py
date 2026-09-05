import sys, os, torch, torch.nn as nn, torch.nn.functional as F
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
CV=os.path.join(REPO,"alpha-beta-CROWN","complete_verifier")
sys.path.insert(0, CV); sys.path.insert(0, os.path.join(CV,"auto_LiRPA"))
from model_defs import resnet2b
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
torch.set_num_threads(4)
MEAN=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1); STD=torch.tensor([0.2471,0.2435,0.2616]).view(1,3,1,1)

base=resnet2b(); ck=torch.load(os.path.join(CV,"models/cifar10_resnet/resnet2b.pth"),map_location="cpu")
base.load_state_dict(ck["state_dict"]); base.eval()

def block_linear_skeleton(blk, in_shape):
    """Return (M, c) for s -> conv2(conv1(s)) + shortcut(s), the LINEAR (relu-free) skeleton
       incl. all biases folded: value = M @ s.flatten + c."""
    C,H,Wd = in_shape; N=C*H*Wd
    def lin(sflat):
        s=sflat.view(-1,C,H,Wd)
        out=blk.conv2(blk.conv1(s))            # affine, no relu
        out=out+ (blk.shortcut(s) if len(list(blk.shortcut.children())) else s)
        return out.flatten(1)
    with torch.no_grad():
        c=lin(torch.zeros(1,N))[0]             # constant term
        eye=torch.eye(N)
        cols=[]
        for i in range(0,N,512):
            cols.append(lin(eye[i:i+512])-c)   # (b, out)
        M=torch.cat(cols,0).t().contiguous()   # (out, N)
    return M, c

class FoldedBlock(nn.Module):
    """relu( M s + c  +  conv2_weight * relu(-(conv1(s))) )  ==  original block."""
    def __init__(self, blk, in_shape, out_shape):
        super().__init__()
        self.conv1=blk.conv1; self.conv2=blk.conv2; self.out_shape=out_shape
        M,c=block_linear_skeleton(blk,in_shape)
        self.lin=nn.Linear(M.shape[1],M.shape[0])
        with torch.no_grad(): self.lin.weight.copy_(M); self.lin.bias.copy_(c)
        self._Msum=M.abs().sum().item()
    def forward(self,s):
        B=s.shape[0]
        extra=F.conv2d(torch.relu(-self.conv1(s)), self.conv2.weight, bias=None,
                       stride=self.conv2.stride, padding=self.conv2.padding)
        lin=self.lin(s.flatten(1)).view(B,*self.out_shape)
        return torch.relu(lin+extra)

class OrigBlock(nn.Module):
    def __init__(self, blk):
        super().__init__(); self.conv1=blk.conv1; self.conv2=blk.conv2; self.shortcut=blk.shortcut
    def forward(self,s):
        out=self.conv2(torch.relu(self.conv1(s)))
        out=out+(self.shortcut(s) if len(list(self.shortcut.children())) else s)
        return torch.relu(out)

class Resnet2bVar(nn.Module):
    def __init__(self, base, foldA=False, foldB=False):
        super().__init__()
        self.conv1=base.conv1
        A,Bk=base.layer1[0], base.layer1[1]
        self.blockA=FoldedBlock(A,(8,16,16),(16,8,8)) if foldA else OrigBlock(A)
        self.blockB=FoldedBlock(Bk,(16,8,8),(16,8,8)) if foldB else OrigBlock(Bk)
        self.linear1=base.linear1; self.linear2=base.linear2
    def forward(self,x):
        out=torch.relu(self.conv1(x)); out=self.blockA(out); out=self.blockB(out)
        out=out.view(out.size(0),-1); out=torch.relu(self.linear1(out)); return self.linear2(out)

# --- cancellation diagnostic: |M| vs |shortcut| baseline ---
MA,_=block_linear_skeleton(base.layer1[0],(8,16,16))
MB,_=block_linear_skeleton(base.layer1[1],(16,8,8))
print(f"block A: |conv2.conv1+shortcut|_sum = {MA.abs().sum():.1f}")
print(f"block B: |conv2.conv1+I|_sum        = {MB.abs().sum():.1f}   (|I|_sum={16*8*8})")

# --- build variants, check bit-exact equivalence ---
variants={"orig":Resnet2bVar(base), "foldA":Resnet2bVar(base,foldA=True),
          "foldB":Resnet2bVar(base,foldB=True), "foldAB":Resnet2bVar(base,foldA=True,foldB=True)}
g=torch.Generator().manual_seed(3); xt=(torch.rand(4,3,32,32,generator=g)-MEAN)/STD
with torch.no_grad():
    y0=variants["orig"](xt)
    for k,v in variants.items():
        v.eval(); d=(v(xt)-y0).abs().max().item()
        print(f"  equiv {k:8s} max|Δ|={d:.2e} {'OK' if d<1e-4 else 'BAD'}")

# --- IBP output-width comparison at a couple eps ---
def out_width(net, x, eps, method="IBP"):
    bm=BoundedModule(net, x[:1], device="cpu")
    tot_mean=tot_max=0.0
    for i in range(x.shape[0]):
        bx=BoundedTensor(x[i:i+1], PerturbationLpNorm(norm=float('inf'),eps=eps))
        lb,ub=bm.compute_bounds(x=(bx,), method=method); w=(ub-lb)[0]
        tot_mean+=w.mean().item(); tot_max=max(tot_max,w.max().item())
    return tot_mean/x.shape[0], tot_max

for eps_pix in [2/255, 8/255]:
    eps=(eps_pix/STD).max().item()
    print(f"\n=== IBP output width, eps_pix={eps_pix:.4f} (box eps={eps:.4f}) ===")
    base_w=None
    for k,v in variants.items():
        mw,xw=out_width(v, xt, eps, "IBP")
        tag=""
        if k=="orig": base_w=mw
        else: tag=f"  ({'IMPROVED' if mw<base_w-1e-3 else 'worse/equal'} vs orig {100*(mw-base_w)/base_w:+.1f}%)"
        print(f"  {k:8s} IBP mean_width={mw:12.2f} max={xw:12.2f}{tag}")
