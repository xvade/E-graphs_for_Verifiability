# Load resnet2b.onnx into TASO and export to the .taso graph format.
import taso

graph = taso.load_onnx("NNs/resnet2b.onnx")
print("loaded OK")
taso.export_to_file(graph, b"NNs/resnet2b.taso")
print("exported OK")
