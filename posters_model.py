import torchvision.models as models
import torch
import os

## I am getting a strange [SSL: CERTIFICATE_VERIFY_FAILED] error when trying to download weights. If you want to download the weights uncomment the next line and comment the next three lines.
# mobilenet = models.mobilenet_v3_small(models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model_path = "./models/mobilenet.pth"
mobilenet = models.mobilenet_v3_small()
mobilenet.load_state_dict(torch.load(model_path))
model = torch.nn.Sequential(mobilenet.features, mobilenet.avgpool, torch.nn.Flatten())# .cuda()
