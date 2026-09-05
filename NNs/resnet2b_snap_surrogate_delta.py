import sys, os, torch, torch.nn as nn, copy
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; CV=os.path.join(REPO,"alpha-beta-CROWN","complete_verifier")
sys.path.insert(0,CV); sys.path.insert(0,os.path.join(CV,"auto_LiRPA"))
from model_defs import resnet2b
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
torch.set_num_threads(4)
MEAN=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1); STD=torch.tensor([0.2471,0.2435,0.2616]).view(1,3,1,1)
base=resnet2b(); base.load_state_dict(torch.load(os.path.join(CV,"models/cifar10_resnet/resnet2b.pth"),map_location="cpu")["state_dict"]); base.eval()
i,j,beta=0,1,-0.948
snap=copy.deepcopy(base); snap.eval()
with torch.no_grad():
    snap.conv1.weight[i]=beta*base.conv1.weight[j]; snap.conv1.bias[i]=beta*base.conv1.bias[j]
class Diff(nn.Module):
    def __init__(s): super().__init__(); s.a=base; s.b=snap
    def forward(s,x): return s.a(x)-s.b(x)
diff=Diff().eval()
import torchvision
ds=torchvision.datasets.CIFAR10(root=os.path.join(CV,"datasets"),train=False,download=False)
xs=[torch.tensor(list(ds[k][0].getdata()),dtype=torch.float32).view(32,32,3).permute(2,0,1)/255.0 for k in range(16)]
X=(torch.stack(xs)-MEAN)/STD; eps=(2/255/STD).max().item()
def cb(net,xi,method):
    bm=BoundedModule(net,xi[:1],device="cpu",bound_opts={"conv_mode":"matrix"})
    bx=BoundedTensor(xi,PerturbationLpNorm(norm=float('inf'),eps=eps))
    return bm.compute_bounds(x=(bx,),method=method)
du_ibp=du_crown=0.0; wo=ws=0.0
for k in range(16):
    xi=X[k:k+1]
    lb,ub=cb(diff,xi,"IBP");   du_ibp=max(du_ibp,ub.abs().max().item(),lb.abs().max().item())
    lb,ub=cb(diff,xi,"CROWN"); du_crown=max(du_crown,ub.abs().max().item(),lb.abs().max().item())
    lo,uo=cb(base,xi,"CROWN"); ls,us=cb(snap,xi,"CROWN")
    wo+=(uo-lo)[0].mean().item(); ws+=(us-ls)[0].mean().item()
wo/=16; ws/=16
print(f"# SOUND delta_upper = sup_box max_logit |orig-snap|  (per-logit, worst over 16 boxes)")
print(f"#   IBP  bound: {du_ibp:.3f}")
print(f"#   CROWN bound: {du_crown:.3f}   (sampled lower bound was 0.09)")
print(f"# vanilla CROWN mean width: orig={wo:.4f}  snap(8ch, no collapse)={ws:.4f}  (snap alone Δ={ws-wo:+.4f})")
print(f"# surrogate needs: [CROWN width tightening from COLLAPSE] > 2*delta_upper = {2*du_crown:.3f}")
