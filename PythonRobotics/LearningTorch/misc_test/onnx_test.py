import torch


class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        for i in range(self.n):
            x = self.conv(x)
        return x


if __name__ == "__main__":
    models = [Model(2), Model(3)]
    model_names = ["model_2", "model_3"]
    for model, model_name in zip(models, model_names):
        dummy_input = torch.randn(1, 3, 10, 10)
        dummy_output = model(dummy_input)

        model_trace = torch.jit.trace(model, dummy_input)
        model_script = torch.jit.script(model)

        # trace is equavilent to torch.onnx.export(model, ...)
        torch.onnx.export(
            model_trace,
            dummy_input,
            "./res/{}_trace.onnx".format(model_name),
            verbose=True,
            input_names=["actual_input"],
            output_names=["output1"],
        )

        # call torch.jit.sciprt firstly
        torch.onnx.export(
            model_script,
            dummy_input,
            f"./res/{model_name}_script.onnx",
            verbose=True,
            input_names=["actual_input"],
            output_names=["output1"],
        )
