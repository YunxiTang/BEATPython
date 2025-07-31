import onnx
import onnxruntime
import torch
import numpy as np
import time


class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        for i in range(self.n):
            x = self.conv(x)
        return x


# Load the ONNX model
model = onnx.load("./res/model_3_script.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))

native_model = Model(3).cuda()

image = torch.randn(1, 3, 10, 10).cuda()

session = onnxruntime.InferenceSession("./res/model_3_script.onnx")
session.get_modelmeta()

tmp1 = []
for i in range(100):
    ts = time.time()
    output1 = native_model(image)
    tmp1.append(time.time() - ts)
print(np.average(tmp1))

tmp2 = []
for i in range(100):
    ts = time.time()
    output2 = session.run(["output1"], {"actual_input": image.cpu().numpy()})
    tmp2.append(time.time() - ts)
print(np.average(tmp2))

print(np.average(tmp1) / np.average(tmp2))
