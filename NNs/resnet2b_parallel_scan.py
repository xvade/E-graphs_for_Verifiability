import sys, os, torch, math
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
CV=os.path.join(REPO,"alpha-beta-CROWN","complete_verifier"); sys.path.insert(0,CV)
from model_defs import resnet2b
base=resnet2b(); base.load_state_dict(torch.load(os.path.join(CV,"models/cifar10_resnet/resnet2b.pth"),map_location="cpu")["state_dict"]); base.eval()

def scan(W):
    R=W.detach().reshape(W.shape[0],-1).double(); n=R.shape[0]
    nrm=R.norm(dim=1); keep=nrm>1e-9
    idx=torch.arange(n)[keep]; R=R[keep]; nrm=nrm[keep]
    U=R/nrm[:,None]; C=(U@U.t()); C.fill_diagonal_(0.0)
    aM=C.abs().argmax().item(); i,j=aM//C.shape[0], aM%C.shape[0]
    cos=C[i,j].item(); beta=(R[i]@R[j]/(R[j]@R[j])).item()   # w_i ~ beta w_j
    # exact merge error if we snap row i := beta * row j :
    snap_err=(R[i]-beta*R[j]).abs().max().item()
    rel_err=snap_err/ (R[i].abs().max().item()+1e-12)
    resid=math.sqrt(max(0.0,2*(1-abs(cos))))
    return dict(n=n,pair=(idx[i].item(),idx[j].item()),cos=cos,beta=beta,resid=resid,
                snap_err=snap_err,rel_err=rel_err)

print("# closest-to-parallel row/filter pair per layer (for EXACT merge, need snap_err<<1e-4):")
for nm,layer in [("conv1_stem",base.conv1),("conv1_A",base.layer1[0].conv1),("conv2_A",base.layer1[0].conv2),
                 ("conv1_B",base.layer1[1].conv1),("conv2_B",base.layer1[1].conv2),
                 ("linear1",base.linear1),("linear2",base.linear2)]:
    d=scan(layer.weight)
    print(f"  {nm:11s} n={d['n']:4d} pair={str(d['pair']):10s} |cos|={abs(d['cos']):.5f} resid={d['resid']:.4f} "
          f"beta={d['beta']:+.3f} snap_abs_err={d['snap_err']:.2e} rel={d['rel_err']:.2e}")
