# Exports alpha-beta-CROWN's mnist_cnn_a model (model_defs.mnist_cnn_4layer:
# Conv(1,16,4x4,s2,p1) -> ReLU -> Conv(16,32,4x4,s2,p1) -> ReLU -> Flatten
# -> Linear(1568,100) -> ReLU -> Linear(100,10)) to ONNX, real trained
# weights from models/sdp/mnist_cnn_a_adv.model. Run with the host `taso_py`
# conda env (has torch, no GPU needed for this step).
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "alpha-beta-CROWN/complete_verifier")
from model_defs import mnist_cnn_4layer

model = mnist_cnn_4layer()
sd = torch.load("alpha-beta-CROWN/complete_verifier/models/sdp/mnist_cnn_a_adv.model", map_location="cpu")
model.load_state_dict(sd)
model.eval()

dummy_input = torch.zeros(1, 1, 28, 28)
torch.onnx.export(
    model, dummy_input, "NNs/mnist_cnn_a.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=11, dynamo=False,
)
print("exported NNs/mnist_cnn_a.onnx OK")
