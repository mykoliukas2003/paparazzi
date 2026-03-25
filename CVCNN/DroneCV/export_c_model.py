import torch
import numpy as np
from ComputerVision import CNN

model = CNN()
model.load_state_dict(torch.load("depth_model.pth", map_location="cpu"))
model.eval()

# Print shapes
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Export as C header
with open("cnn_weights.h", "w") as f:
    f.write("#ifndef CNN_WEIGHTS_H\n#define CNN_WEIGHTS_H\n\n")
    for name, param in model.named_parameters():
        data = param.detach().numpy().flatten()
        c_name = name.replace(".", "_")
        f.write(f"// {name}: {list(param.shape)}\n")
        f.write(f"static const float {c_name}[{len(data)}] = {{\n")
        for i, val in enumerate(data):
            f.write(f"  {val:.8f}f,\n" if (i % 8 == 7) else f"{val:.8f}f, ")
        f.write("};\n\n")
    f.write("#endif\n")

print("Exported cnn_weights.h")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")