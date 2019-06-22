import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Alexnetの実装確認用
net = models.alexnet(pretrained=True)

for param in net.parameters():
    #print(param.size())
    param.requires_grad = False
    #pass

# 転移学習においては適時レイヤーのパラメータを手動で調整する必要がある
# children: Returns an iterator over immediate children modules.
# for x in list(net.children()):
#   print(x, '\n')

# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#   (1): ReLU(inplace)
#   (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   (4): ReLU(inplace)
#   (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (7): ReLU(inplace)
#   (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (9): ReLU(inplace)
#   (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace)
#   (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
# )
#
# Sequential(
#   (0): Dropout(p=0.5)
#   (1): Linear(in_features=9216, out_features=4096, bias=True)
#   (2): ReLU(inplace)
#   (3): Dropout(p=0.5)
#   (4): Linear(in_features=4096, out_features=4096, bias=True)
#   (5): ReLU(inplace)
#   (6): Linear(in_features=4096, out_features=1000, bias=True)
# )

# classifierは単なる全結合層
#for i in range(7):
#    print(net.classifier[i])

# Dropout(p=0.5)
# Linear(in_features=9216, out_features=4096, bias=True)
# ReLU(inplace)
# Dropout(p=0.5)
# Linear(in_features=4096, out_features=4096, bias=True)
# ReLU(inplace)
# Linear(in_features=4096, out_features=1000, bias=True)

net.classifier[6] = nn.Linear(4096, 2)
#print(net.classifier[6].parameters().requires_grad)

for param in net.parameters():
    print(param.size(), param.requires_grad)