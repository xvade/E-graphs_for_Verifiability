import torch

from inception_mnist_model import InceptionMNIST

model = InceptionMNIST()
model.load_state_dict(torch.load("NNs/inception_mnist.pth", map_location="cpu"))
model.eval()

dummy_input = torch.zeros(1, 1, 28, 28)
torch.onnx.export(
    model, dummy_input, "NNs/inception_mnist.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=11, dynamo=False,
)
print("exported NNs/inception_mnist.onnx OK")
