import taso

graph = taso.load_onnx("NNs/mnist_tiny_mlp.onnx")
print("loaded OK")
taso.export_to_file(graph, b"NNs/mnist_tiny_mlp.taso")
print("exported OK")
