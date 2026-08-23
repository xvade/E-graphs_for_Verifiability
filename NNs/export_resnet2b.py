# Exports alpha-beta-CROWN's resnet2b model (model_defs.resnet2b:
# CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")
# -- a real residual network with a parallel conv+shortcut branch merged via
# elementwise Add, unlike the purely-sequential models tried so far) to
# ONNX, real trained weights from models/cifar10_resnet/resnet2b.pth.
# Run with the host `taso_py` conda env (has torch, no GPU needed).
import sys
import torch

sys.path.insert(0, "alpha-beta-CROWN/complete_verifier")
from model_defs import resnet2b

model = resnet2b()
ckpt = torch.load("alpha-beta-CROWN/complete_verifier/models/cifar10_resnet/resnet2b.pth", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

dummy_input = torch.zeros(1, 3, 32, 32)
torch.onnx.export(
    model, dummy_input, "NNs/resnet2b.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=11, dynamo=False,
)
print("exported NNs/resnet2b.onnx OK")
