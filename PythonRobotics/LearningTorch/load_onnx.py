import onnx

# Load the ONNX model
model = onnx.load("./res/model_2_script.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))