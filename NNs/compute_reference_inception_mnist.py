import numpy as np
import torch

from inception_mnist_model import InceptionMNIST

np.random.seed(0)
x = np.random.randn(1, 1, 28, 28).astype(np.float32)

model = InceptionMNIST()
model.load_state_dict(torch.load("NNs/inception_mnist.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    out = model(torch.from_numpy(x)).numpy()

np.save("NNs/reference_input_inception_mnist.npy", x)
np.save("NNs/reference_output_inception_mnist.npy", out)
print("PyTorch reference output:", out)
