import torch
from ComputerVision import CNN 

# Load the trained model
model = CNN()
model.load_state_dict(torch.load("depth_model (8 in channels).pth", map_location="cpu"))
model.eval()


dummy_input = torch.randn(1, 3, 80, 520)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "depth_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=16
)

print("ONNX model exported successfully!")