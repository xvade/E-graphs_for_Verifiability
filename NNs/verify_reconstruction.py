# Assert the reconstructed mnist_tiny_mlp_optimized.onnx reproduces the seed-0
# PyTorch reference output (compute_reference.py) to atol 1e-4, under
# single-threaded onnxruntime. Exits nonzero (assert) on mismatch.
import numpy as np
import onnxruntime as ort

x = np.load("NNs/reference_input.npy")
reference_out = np.load("NNs/reference_output.npy")

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
sess = ort.InferenceSession("NNs/mnist_tiny_mlp_optimized.onnx", sess_options=so,
                             providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
out = sess.run(None, {input_name: x})[0]
print("ONNX (reconstructed) output:", out)
print("PyTorch (reference) output: ", reference_out)
print("max abs diff:", np.max(np.abs(out - reference_out)))
assert np.allclose(out, reference_out, atol=1e-4), "MISMATCH between reconstructed ONNX and reference PyTorch output"
print("MATCH: reconstructed model is numerically equivalent to the original")
