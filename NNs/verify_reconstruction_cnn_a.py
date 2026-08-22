import numpy as np
import onnxruntime as ort

x = np.load("NNs/reference_input_cnn_a.npy")
reference_out = np.load("NNs/reference_output_cnn_a.npy")

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
sess = ort.InferenceSession("NNs/mnist_cnn_a_optimized.onnx", sess_options=so,
                             providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
out = sess.run(None, {input_name: x})[0]
print("ONNX (reconstructed) output:", out)
print("PyTorch (reference) output: ", reference_out)
print("max abs diff:", np.max(np.abs(out - reference_out)))
assert np.allclose(out, reference_out, atol=1e-4), "MISMATCH between reconstructed ONNX and reference PyTorch output"
print("MATCH: reconstructed model is numerically equivalent to the original")
