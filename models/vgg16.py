import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import numpy as np

# model = models.vgg16(pretrained=True)

# hoge = torchvision.models.vgg16(pretrained=True)

model = models.vgg16(pretrained=True)
# model = models.alexnet(pretrained=True)

print(model)

# for x in list(vgg16.children()):
#     print(x, '\n')
