import sys
import numpy as np
import torch

sys.path.insert(0, "alpha-beta-CROWN/complete_verifier")
from model_defs import resnet2b

np.random.seed(0)
x = np.random.randn(1, 3, 32, 32).astype(np.float32)

model = resnet2b()
ckpt = torch.load("alpha-beta-CROWN/complete_verifier/models/cifar10_resnet/resnet2b.pth", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

with torch.no_grad():
    out = model(torch.from_numpy(x)).numpy()

np.save("NNs/reference_input_resnet2b.npy", x)
np.save("NNs/reference_output_resnet2b.npy", out)
print("PyTorch reference output:", out)
