# Load mnist_tiny_mlp.onnx into TASO and export to the .taso graph format
# (the input to tensat). Run in the container / with taso on LD_LIBRARY_PATH.
import taso

graph = taso.load_onnx("NNs/mnist_tiny_mlp.onnx")
print("loaded OK")
taso.export_to_file(graph, b"NNs/mnist_tiny_mlp.taso")
print("exported OK")
