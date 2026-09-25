"""
Create a simple CNN model with dynamic spatial dimensions for benchmarking.
The model has enough intermediate tensors to make memory planning observable.
"""

import torch
import torch.nn as nn
import os


class BenchmarkCNN(nn.Module):
    """A model with ~50 intermediate tensors to stress memory planning."""

    def __init__(self):
        super().__init__()
        # Stack enough layers to create many intermediate tensors
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def export_model(output_path: str):
    model = BenchmarkCNN()
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Model exported to {output_path}")
    print(f"  Dynamic axes: batch, height, width")
    print(f"  ~50 intermediate tensors for memory planning stress test")


if __name__ == "__main__":
    export_model(os.path.join(os.path.dirname(__file__), "test_model.onnx"))
