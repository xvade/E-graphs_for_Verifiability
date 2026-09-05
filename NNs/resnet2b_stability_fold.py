#!/usr/bin/env python
"""
resnet2b IBP "fifth door": STABILITY-CONDITIONED SELECTIVE FLIP-AND-FOLD.
Validated on REAL CIFAR-10 test images, eps=2/255, IBP. (2026-09-04)

MECHANISM. Block A is relu( conv2(relu(conv1(s))) + short(s) ). The identity
relu(u)=u+relu(-u) holds for EVERY coord (global). Apply it only on a coord set S of
conv1(s): conv2(relu(conv1(s)))+short(s) = L_S(s) + R(s), with
  L_S(s)=conv2(m_S*conv1(s))+b2+short(s)  -- ONE linear op (M s + c), folds the main path's
         linear skeleton on S with the shortcut: |L_S| <= |B_S||A_S|+|short|, strict
         generically, NO weight cancellation required.
  R(s)  =conv2( m_S*relu(-conv1(s)) + (1-m_S)*relu(conv1(s)) ).
For an IBP-STABLE-ACTIVE coord (l^IBP>0), relu(-conv1(s)) has IBP width EXACTLY 0 -- the flip
is FREE -- and that coord's linear part is folded into L_S. The earlier full door-2 fold
(S=all coords) LOST because it flipped unstable coords too (each adds |conv2[:,i]|*|l_i|).
The lever is NEURON STABILITY, not residual weight cancellation.

MEASURED (real CIFAR, this script):
  (A) per-box S = that box's IBP-stable-active set (each net GLOBALLY equivalent to stock
      resnet2b, ~2e-6): IBP output width -41%, tighter 16/16, SOUND (MC in-box violation<0).
      This meets the goal "a rewrite of resnet2b that improves IBP, max err<=1e-4" per box.
  Bonus MEASURED UNREACHED: CROWN(orig) mean width ~3.4 << IBP(fold) ~994; IBP stays vacuous
      vs CROWN (standard-trained net), so no rewrite makes IBP beat CROWN.
  (B/C) A SINGLE input-independent FIXED rewrite does NOT meaningfully help on real images:
      coords stable-active across ALL calib images are only 0.3-0.6% (box-stability is
      input-dependent at 2/255) -> ~ -0.01% held-out; the naive majority mask (14.7%) even
      LOSES out-of-sample (+8.4%). So the fifth door is a PER-BOX exact rewrite, not a single
      fixed net. (On uniform-noise inputs the per-box gain is -53% and majority is 37%/-14.5%;
      real images are the honest numbers.)

VANILLA CROWN: the fold is NEUTRAL (section D: 0/16, Δ within float32 reconstruction).
  Vanilla CROWN back-substitutes CROWN intermediate bounds EXACTLY through stable (linear)
  ReLUs, so removing their box slack is invisible to it -- the fifth door is IBP-specific
  (IBP and plain CROWN lose tightness in DIFFERENT places; the stable-neuron place is already
  exact for CROWN). It IS visible to CROWN-IBP (-41.7%), which takes IBP intermediate bounds.
  Improving vanilla CROWN needs a HULL-CHANGING rewrite on UNSTABLE neurons (redundancy-
  collapse / min-max reassoc); stock resnet2b has no such exact site (parallel-filter scan
  closest pair cos 0.993 / snap 0.037; no min/max tree).

Same |BA| mechanism as the validated 37% win in
NNs/reassoc_results/plain_relu_more_verifiable.py, here located inside stock resnet2b via
stable ReLUs (box-conditional consecutive linears the graph-level door-1 check missed).
Login-light. Run: python NNs/resnet2b_stability_fold.py
"""
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
A=base.layer1[0]; IN=(8,16,16); OUTA=(16,8,8); N=IN[0]*IN[1]*IN[2]
has_short=len(list(A.shortcut.children()))>0
def sc_d(s):
    if not has_short: return s
    c=A.shortcut[0]; return F.conv2d(s,c.weight.double(),c.bias.double() if c.bias is not None else None,stride=c.stride,padding=c.padding)
def ibp_conv(cl,cu,conv):
    c=(cl+cu)/2; r=(cu-cl)/2
    yc=F.conv2d(c,conv.weight,conv.bias,stride=conv.stride,padding=conv.padding)
    yr=F.conv2d(r,conv.weight.abs(),None,stride=conv.stride,padding=conv.padding); return yc-yr,yc+yr

# ---- REAL CIFAR-10 test images ----
import torchvision
ds=torchvision.datasets.CIFAR10(root=os.path.join(CV,"datasets"),train=False,download=False)
xs=[]
for i in range(32):
    img,_=ds[i]; xs.append(torch.tensor(list(img.getdata()),dtype=torch.float32).view(32,32,3).permute(2,0,1)/255.0)
xall=(torch.stack(xs)-MEAN)/STD
CAL=xall[:16]; TEST=xall[16:32]   # disjoint calibration / test
eps=(2/255/STD).max().item()
print(f"# REAL CIFAR-10 test images; eps_box={eps:.4f}; 16 calib / 16 held-out")

def active_mask(xi):
    sl,su=ibp_conv(xi-eps,xi+eps,base.conv1); sl,su=torch.relu(sl),torch.relu(su)
    al,au=ibp_conv(sl,su,A.conv1); return (al>0).float()
def build_LS(m):
    md=m.double()
    def Ls(sflat):
        s=sflat.view(-1,*IN)
        v=md*F.conv2d(s,A.conv1.weight.double(),A.conv1.bias.double(),stride=A.conv1.stride,padding=A.conv1.padding)
        out=F.conv2d(v,A.conv2.weight.double(),A.conv2.bias.double(),stride=A.conv2.stride,padding=A.conv2.padding)
        return (out+sc_d(s)).flatten(1)
    with torch.no_grad():
        c=Ls(torch.zeros(1,N,dtype=torch.float64))[0]; eye=torch.eye(N,dtype=torch.float64)
        M=torch.cat([Ls(eye[i:i+256])-c for i in range(0,N,256)],0).t().contiguous()
    return M.float(), c.float()
class SelFoldBlock(nn.Module):
    def __init__(self,M32,c32,m):
        super().__init__(); self.conv1=A.conv1; self.conv2=A.conv2
        self.lin=nn.Linear(N,M32.shape[0])
        with torch.no_grad(): self.lin.weight.copy_(M32); self.lin.bias.copy_(c32)
        self.register_buffer("m",m.float())
    def forward(self,s):
        B=s.shape[0]; p=self.conv1(s); gg=self.m*torch.relu(-p)+(1-self.m)*torch.relu(p)
        R=F.conv2d(gg,self.conv2.weight,None,stride=self.conv2.stride,padding=self.conv2.padding)
        return torch.relu(self.lin(s.flatten(1)).view(B,*OUTA)+R)
class OrigBlock(nn.Module):
    def __init__(self): super().__init__(); self.conv1=A.conv1; self.conv2=A.conv2; self.short=A.shortcut
    def forward(self,s):
        o=self.conv2(torch.relu(self.conv1(s))); return torch.relu(o+(self.short(s) if has_short else s))
class OrigBlockB(nn.Module):
    def __init__(s2):
        super().__init__(); B=base.layer1[1]; s2.conv1=B.conv1; s2.conv2=B.conv2; s2.short=B.shortcut; s2.hs=len(list(B.shortcut.children()))>0
    def forward(s2,z): o=s2.conv2(torch.relu(s2.conv1(z))); return torch.relu(o+(s2.short(z) if s2.hs else z))
class Net(nn.Module):
    def __init__(self,bA): super().__init__(); self.c1=base.conv1; self.bA=bA; self.bB=OrigBlockB(); self.l1=base.linear1; self.l2=base.linear2
    def forward(self,x):
        o=torch.relu(self.c1(x)); o=self.bA(o); o=self.bB(o); o=o.view(o.size(0),-1); return self.l2(torch.relu(self.l1(o)))
orig=Net(OrigBlock()).eval()

def bounds(net,xi,method,conv_mode=None):
    # conv_mode="matrix" needed for CROWN on the folded net: the constant-mask Mul breaks
    # auto_LiRPA's default Patches conv mode (same class as the Split note in
    # convfused-verified-neutral). Matrix mode is mathematically identical, just dense/slower.
    opts={"conv_mode":conv_mode} if conv_mode else None
    bm=BoundedModule(net,xi[:1],device="cpu",bound_opts=opts)
    bx=BoundedTensor(xi,PerturbationLpNorm(norm=float('inf'),eps=eps))
    lb,ub=bm.compute_bounds(x=(bx,),method=method); return lb.detach(),ub.detach()
def width(lb,ub): return (ub-lb)[0].mean().item()

# ============ (A) per-image S + SOUNDNESS MC + CROWN(orig) ============
print("\n=== (A) per-image S: IBP(fold) vs IBP(orig) vs CROWN(orig); + soundness ===")
gg=torch.Generator().manual_seed(7); xrand=(torch.rand(64,3,32,32,generator=gg)-MEAN)/STD
mo=mf=mc=0.0; tighter=0; worse_than_crown=0; maxeq=0.0; max_viol=-1e9
for i in range(16):
    xi=CAL[i:i+1]; m=active_mask(xi); M32,c32=build_LS(m); fnet=Net(SelFoldBlock(M32,c32,m)).eval()
    with torch.no_grad(): maxeq=max(maxeq,(orig(xrand)-fnet(xrand)).abs().max().item())
    lbo,ubo=bounds(orig,xi,"IBP"); lbf,ubf=bounds(fnet,xi,"IBP"); lbc,ubc=bounds(orig,xi,"CROWN")
    wo,wf,wc=width(lbo,ubo),width(lbf,ubf),width(lbc,ubc)
    mo+=wo; mf+=wf; mc+=wc; tighter+=(wf<wo-1e-6); worse_than_crown+=(wf>wc)
    # soundness: sample 200 in-box points, must lie within fold bounds
    U=(torch.rand(200,3,32,32)*2-1)*eps + xi
    with torch.no_grad(): yo=orig(U)
    viol=max((lbf-yo).max().item(),(yo-ubf).max().item()); max_viol=max(max_viol,viol)
mo/=16; mf/=16; mc/=16
print(f"  equiv max|Δ|(global rand)={maxeq:.2e} {'OK' if maxeq<1e-4 else 'BAD'}")
print(f"  SOUNDNESS max in-box violation of fold bounds = {max_viol:.2e}  {'SOUND (<=0)' if max_viol<=1e-4 else 'UNSOUND!'}")
print(f"  IBP(orig)   mean_width={mo:.3f}")
print(f"  IBP(fold)   mean_width={mf:.3f}   ({100*(mf-mo)/mo:+.2f}%, tighter {tighter}/16)")
print(f"  CROWN(orig) mean_width={mc:.3f}   [IBP(fold) worse than CROWN(orig) on {worse_than_crown}/16 -> bonus {'UNREACHED' if worse_than_crown>0 else 'reached?'}]")

# ============ (B) fixed majority S, HELD-OUT ============
votes=sum(active_mask(CAL[i:i+1]) for i in range(16)); mmaj=(votes>=9).float()
M32,c32=build_LS(mmaj); fnet=Net(SelFoldBlock(M32,c32,mmaj)).eval()
print(f"\n=== (B) fixed majority S (calib active>=9/16): {int(mmaj.sum())}/{mmaj.numel()}={100*mmaj.mean():.1f}%, measured on HELD-OUT 16 ===")
mo2=mf2=0.0; t2=0
for i in range(16):
    xi=TEST[i:i+1]; lbo,ubo=bounds(orig,xi,"IBP"); lbf,ubf=bounds(fnet,xi,"IBP")
    wo,wf=width(lbo,ubo),width(lbf,ubf); mo2+=wo; mf2+=wf; t2+=(wf<wo-1e-6)
mo2/=16; mf2/=16
print(f"  IBP(orig)  held-out mean_width={mo2:.3f}")
print(f"  IBP(fold)  held-out mean_width={mf2:.3f}   ({100*(mf2-mo2)/mo2:+.2f}%, tighter {t2}/16)")

# ============ (C) conservative input-independent fixed S, HELD-OUT ============
print("\n=== (C) conservative fixed S (stable-active across calib), HELD-OUT ===")
for thr,lab in [(16,"ALL 16/16"),(15,">=15/16"),(13,">=13/16")]:
    m=(votes>=thr).float(); M32,c32=build_LS(m); fn=Net(SelFoldBlock(M32,c32,m)).eval()
    a=b=0.0; t=0
    for i in range(16):
        xi=TEST[i:i+1]; lbo,ubo=bounds(orig,xi,"IBP"); lbf,ubf=bounds(fn,xi,"IBP")
        a+=width(lbo,ubo); b+=width(lbf,ubf); t+=(width(lbf,ubf)<width(lbo,ubo)-1e-6)
    a/=16; b/=16
    print(f"  S={lab}: |S|={int(m.sum())}/{m.numel()}={100*m.mean():.1f}%  held-out {a:.1f}->{b:.1f} ({100*(b-a)/a:+.2f}%, tighter {t}/16)")

# ============ (D) does the per-box fold help VANILLA CROWN? (no) ============
# Vanilla CROWN back-substitutes CROWN intermediate bounds EXACTLY through stable (linear)
# ReLUs, so folding them is invisible -> NEUTRAL. It IS visible to any CROWN-class method
# whose INTERMEDIATE bounds come from IBP (CROWN-IBP), because the tighter block-A box then
# propagates to tighter block-B ReLU hulls. Uses conv_mode="matrix" (Patches breaks on the
# constant-mask Mul). CROWN-family bound is 2 logits => small; light on CPU for 16 imgs.
print("\n=== (D) per-box fold under the CROWN family (conv_mode=matrix) ===")
folds=[Net(SelFoldBlock(*build_LS(active_mask(CAL[i:i+1])),active_mask(CAL[i:i+1]))).eval() for i in range(16)]
for method in ["CROWN","CROWN-IBP","IBP"]:
    mo=mf=0.0; t=0; dlo=1e9; dhi=-1e9
    for i in range(16):
        xi=CAL[i:i+1]
        wo=width(*bounds(orig,xi,method,"matrix")); wf=width(*bounds(folds[i],xi,method,"matrix"))
        mo+=wo; mf+=wf; t+=(wf<wo-1e-6); dlo=min(dlo,wf-wo); dhi=max(dhi,wf-wo)
    mo/=16; mf/=16
    print(f"  {method:9s}: orig {mo:.4f} -> fold {mf:.4f}  ({100*(mf-mo)/mo:+.3f}%, tighter {t}/16, signed Δ[{dlo:+.1e},{dhi:+.1e}])")
print("# vanilla CROWN NEUTRAL (0/16, Δ within float32 fold-reconstruction) -- the fifth door is")
print("# IBP-specific: IBP loses tightness at stable neurons (box-forgetting); CROWN does not.")
