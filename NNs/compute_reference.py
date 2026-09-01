# Compute a fixed (seed-0) PyTorch reference (input, output) pair for
# mnist_tiny_mlp, saved as NNs/reference_{input,output}.npy. The reconstruction
# round-trip check (verify_reconstruction.py) asserts the rebuilt ONNX matches
# this to atol 1e-4.
import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
x = np.random.randn(1, 1, 28, 28).astype(np.float32)

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 20), nn.ReLU(), nn.Linear(20, 10))
sd = torch.load(
    "alpha-beta-CROWN/complete_verifier/models/toy/mnist_2_20.pth", map_location="cpu"
)
model.load_state_dict(sd)
model.eval()

with torch.no_grad():
    out = model(torch.from_numpy(x)).numpy()

np.save("NNs/reference_input.npy", x)
np.save("NNs/reference_output.npy", out)
print("PyTorch reference output:", out)
