import taso

graph = taso.load_onnx("NNs/inception_mnist.onnx")
print("loaded OK")
taso.export_to_file(graph, b"NNs/inception_mnist.taso")
print("exported OK")
