#!/usr/bin/env python
"""
Fire-condition scan for the vanilla-CROWN-improving rewrite (redundancy-collapse) on stock
resnet2b. (2026-09-04)

The ONLY exact rewrite that tightens vanilla CROWN is collapsing redundant unstable ReLUs
(proved by induction: plain CROWN's per-neuron relaxation gap is rewrite-invariant except by
changing the neuron SET). Two variants, both needing PRE-ACTIVATION redundancy (weight AND
bias), separated by the sign of the cosine of the augmented [weight|bias] row vectors:
  - proportional-merge   (cos -> +1): rows w_i = beta*w_j, b_i = beta*b_j; tightens iff the two
    neurons' DOWNSTREAM coeffs oppose in sign. Measured -59.4% vanilla CROWN on a toy
    (crown_redundancy_collapse.py).
  - complementary-collapse(cos -> -1): rows w_i = -w_j, b_i = -b_j (a relu(z)/relu(-z) pair).
    Measured -33.4% vanilla CROWN on a toy.
An EXACT merge needs snap-error << 1e-4. RESULT on stock resnet2b: no proportional site (best
cos +0.59); the closest pair anywhere is the stem's near-complementary pair (cos -0.9932) at
snap 4.25e-2 = 14% rel, 425x over 1e-4. So standard training produced no exact site; the
mechanism needs planted/regularized redundancy (a different net than the stock checkpoint).
Weight-only |cos| (the older resnet2b_parallel_scan.py) mislabels this pair as proportional;
the bias-augmented, sign-separated scan here is the correct fire condition.
"""
import sys, os, torch, math
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; CV=os.path.join(REPO,"alpha-beta-CROWN","complete_verifier"); sys.path.insert(0,CV)
from model_defs import resnet2b
base=resnet2b(); base.load_state_dict(torch.load(os.path.join(CV,"models/cifar10_resnet/resnet2b.pth"),map_location="cpu")["state_dict"]); base.eval()

def scan(layer, name):
    W=layer.weight.detach().reshape(layer.weight.shape[0],-1).double()
    b=layer.bias.detach().double().reshape(-1,1) if layer.bias is not None else torch.zeros(W.shape[0],1,dtype=torch.float64)
    A=torch.cat([W,b],1)                       # AUGMENTED pre-activation vector [w | bias]
    n=A.shape[0]; nrm=A.norm(dim=1); keep=nrm>1e-9
    A=A[keep]; nrm=nrm[keep]; m=A.shape[0]
    U=A/nrm[:,None]; C=(U@U.t()); C.fill_diagonal_(0.0)
    # closest by SIGNED cos, separately for +1 (proportional) and -1 (complementary)
    def best(mask_sign):
        Cs=C.clone()
        if mask_sign>0: Cs[Cs<0]=-1  # only consider positive cos for proportional
        else: Cs[Cs>0]=1             # only negative cos for complementary; we then take min
        if mask_sign>0:
            k=Cs.argmax().item(); i,j=k//m,k%m; cos=C[i,j].item()
        else:
            k=Cs.argmin().item(); i,j=k//m,k%m; cos=C[i,j].item()
        beta=(A[i]@A[j]/(A[j]@A[j])).item()
        snap=(A[i]-beta*A[j]).abs().max().item()
        rel=snap/(A[i].abs().max().item()+1e-12)
        return cos,beta,snap,rel,(i,j)
    cp,bp,sp,rp,ijp=best(+1); cc,bc,sc,rc,ijc=best(-1)
    print(f"  {name:11s} n={m:4d} | PROP cos={cp:+.4f} beta={bp:+.3f} snap={sp:.2e} rel={rp:.1e}"
          f"  | COMPL cos={cc:+.4f} beta={bc:+.3f} snap={sc:.2e} rel={rc:.1e}")
    return min(sp,sc)

print("# augmented [weight|bias] pre-activation proportionality scan (EXACT merge needs snap<<1e-4)")
print("# PROP=proportional-merge (cos->+1), COMPL=complementary-collapse (cos->-1)")
layers=[("conv1_stem",base.conv1),("conv1_A",base.layer1[0].conv1),("conv2_A",base.layer1[0].conv2),
        ("conv1_B",base.layer1[1].conv1),("conv2_B",base.layer1[1].conv2),("linear1",base.linear1),("linear2",base.linear2)]
mins=[scan(l,nm) for nm,l in layers]
print(f"\n# best (smallest) exact-merge snap tolerance anywhere: {min(mins):.2e}  (need <=1e-4 for an exact site)")
