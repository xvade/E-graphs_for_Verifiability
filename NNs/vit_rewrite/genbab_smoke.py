import sys, os, re, glob, importlib.util, numpy as np, torch
REPO = "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; GB = os.path.join(REPO, "genbab_benchmarks/cifar")
sys.path.insert(0, os.path.join(REPO, "alpha-beta-CROWN/complete_verifier"))
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
name = sys.argv[1]; cls = "ViT_" + name.split("_", 1)[1]
spec = importlib.util.spec_from_file_location("gbvit", os.path.join(GB, "models/vit.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
net = getattr(mod, cls)(); sd = torch.load(os.path.join(GB, name, "model.pth"), map_location="cpu")
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd; print("# ckpt keys:", list(sd.keys())[:4], "...", len(sd)); net.load_state_dict(sd); net.eval()
def strip_dropout(m):   # p=0 / eval-mode dropout is the identity; auto_LiRPA otherwise demands a training-mode forward
    for k, c in m.named_children():
        if isinstance(c, torch.nn.Dropout): setattr(m, k, torch.nn.Identity())
        else: strip_dropout(c)
strip_dropout(net)
def parse(path):
    t = open(path).read(); lab = int(re.search(r"label: (\d+)", t).group(1))
    lb = [float(v) for v in re.findall(r"\(assert \(>= X_\d+ ([-\d.eE]+)\)\)", t)]; ub = [float(v) for v in re.findall(r"\(assert \(<= X_\d+ ([-\d.eE]+)\)\)", t)]
    return torch.tensor(lb).reshape(1, 3, 32, 32), torch.tensor(ub).reshape(1, 3, 32, 32), lab
files = [os.path.join(GB, name, l.strip()) for l in open(os.path.join(GB, name, "instances.csv")) if l.strip()]
correct = 0; boxes = []
for f in files:
    xl, xu, y = parse(f); x0 = (xl + xu) / 2
    with torch.no_grad(): correct += int(net(x0).argmax(1).item() == y)
    boxes.append((xl, xu, y))
print(f"# {name}/{cls}: {len(files)} instances, center accuracy {correct}/{len(files)}")
lirpa = BoundedModule(net, torch.zeros(1, 3, 32, 32), bound_opts={"softmax": "lse", "conv_mode": "matrix"})
for xl, xu, y in boxes[:3]:
    C = torch.zeros(1, 9, 10); r = 0
    for i in range(10):
        if i != y: C[0, r, y] = 1; C[0, r, i] = -1; r += 1
    bx = BoundedTensor((xl + xu) / 2, PerturbationLpNorm(norm=np.inf, x_L=xl, x_U=xu))
    for m in ["IBP", "CROWN"]:
        lb, ub = lirpa.compute_bounds(x=(bx,), method=m, C=C); print(f"  y={y} {m}: min lb {lb.min().item():+.4f}  mean width {(ub - lb).mean().item():.4f}")
