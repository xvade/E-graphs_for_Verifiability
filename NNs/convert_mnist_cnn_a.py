# Load mnist_cnn_a.onnx into TASO and export to the .taso graph format.
import taso

graph = taso.load_onnx("NNs/mnist_cnn_a.onnx")
print("loaded OK")
taso.export_to_file(graph, b"NNs/mnist_cnn_a.taso")
print("exported OK")
