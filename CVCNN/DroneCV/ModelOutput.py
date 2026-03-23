import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ComputerVision import CNN
import torch
from PIL import Image
from torchvision import transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import json
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CNN()
model_path = "depth_model (4).pth" 

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

transform = transforms.ToTensor()

# test_path = 'Data/Images/01633.jpg' #Checking image from training data (More debugging than testing)
# image = Image.open(test_path).convert("RGB")
# image = transform(image)
# image = image.unsqueeze(0).to(device)

with open("test_filenames (3).json", "r") as f:
    test_data = json.load(f)

# with torch.no_grad():
#     pred = model(image)

loss_fn = torch.nn.MSELoss()
losses = []
forward_pass_times = []
for file in test_data[30:41]:
    path = 'Data/Images/{}'.format(file)
    print(path)
    img = Image.open(path).convert("RGB")
    img = img.crop((80,0,160,520))
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    original = 'Data/datasets_depth/depth_map/{}_depth.png'.format(file[:-4])
    original = Image.open(original)
    original = original.crop((80,0,160,520))
    target = transform(original)
    start = time.time()
    with torch.no_grad():
        pred = model(img)

    end = time.time()
    loss = loss_fn(pred,target)
    losses.append(loss)
    forward_pass_times.append(end-start)
    depth = pred.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    

    plt.subplot(1,2,1)
    plt.imshow(depth, cmap = 'gray')

    plt.subplot(1,2,2)
    plt.imshow(original)
    plt.colorbar()

    plt.show()

print("AVG LOSS: ", np.mean(losses))
print("AVG FORWARD PASS TIME: ", np.mean(forward_pass_times) )
summary(model, input_size = (3,80,520))


# plt.imshow(depth, cmap="gray")
# plt.colorbar()
# plt.show()
#Re



