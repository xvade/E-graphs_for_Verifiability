import sys
import numpy as np
import torch

sys.path.insert(0, "alpha-beta-CROWN/complete_verifier")
from model_defs import mnist_cnn_4layer

np.random.seed(0)
x = np.random.randn(1, 1, 28, 28).astype(np.float32)

model = mnist_cnn_4layer()
sd = torch.load("alpha-beta-CROWN/complete_verifier/models/sdp/mnist_cnn_a_adv.model", map_location="cpu")
model.load_state_dict(sd)
model.eval()

with torch.no_grad():
    out = model(torch.from_numpy(x)).numpy()

np.save("NNs/reference_input_cnn_a.npy", x)
np.save("NNs/reference_output_cnn_a.npy", out)
print("PyTorch reference output:", out)
