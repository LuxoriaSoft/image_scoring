import numpy as np

positive = np.load("onnx_export/positive.npy")
negative = np.load("onnx_export/negative.npy")

np.savetxt("onnx_export/positive.txt", positive)
np.savetxt("onnx_export/negative.txt", negative)
