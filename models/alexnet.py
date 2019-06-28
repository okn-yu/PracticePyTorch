import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

import matplotlib.pyplot as plt

# Alexnetの実装確認用
net = models.alexnet(pretrained=True)

for param in net.parameters():
    #print(param.size())
    param.requires_grad = False
    #pass

# 転移学習においては適時レイヤーのパラメータを手動で調整する必要がある
# 転移学習の学習先データと学習用データのサイズが違いすぎるとモデルの差分が大きくなり不便
# AlexNetは特徴量抽出までのfeaturesと，それを使って分類するclassifeirに分けて実装
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

# net.classifier[6] = nn.Linear(4096, 2)
# print(net.classifier[6].parameters().requires_grad)

# for param in net.parameters():
#     print(param.size(), param.requires_grad)

# torch.Size([64, 3, 11, 11]) False
# torch.Size([64]) False
# torch.Size([192, 64, 5, 5]) False
# torch.Size([192]) False
# torch.Size([384, 192, 3, 3]) False
# torch.Size([384]) False
# torch.Size([256, 384, 3, 3]) False
# torch.Size([256]) False
# torch.Size([256, 256, 3, 3]) False
# torch.Size([256]) False
# torch.Size([4096, 9216]) False
# torch.Size([4096]) False
# torch.Size([4096, 4096]) False
# torch.Size([4096]) False
# torch.Size([2, 4096]) True
# torch.Size([2]) True

# for key, value in net.__dict__.items():
#   print(key, ':', value)

# netの要素は大きく分けてfeaturesとclassifier

# _backend : <torch.nn.backends.thnn.THNNFunctionBackend object at 0x1151c0128>
# _parameters : OrderedDict()
# _buffers : OrderedDict()
# _backward_hooks : OrderedDict()
# _forward_hooks : OrderedDict()
# _forward_pre_hooks : OrderedDict()
# _state_dict_hooks : OrderedDict()
# _load_state_dict_pre_hooks : OrderedDict()
# _modules : OrderedDict([('features', Sequential(
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
# )), ('classifier', Sequential(
#   (0): Dropout(p=0.5)
#   (1): Linear(in_features=9216, out_features=4096, bias=True)
#   (2): ReLU(inplace)
#   (3): Dropout(p=0.5)
#   (4): Linear(in_features=4096, out_features=4096, bias=True)
#   (5): ReLU(inplace)
#   (6): Linear(in_features=4096, out_features=2, bias=True)
# ))])
# training : True

# print(net.features[0]) # Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))

# 重み可視化用サンプルコード
# これが論文再現の第一歩とする!!

test = net.features[3].weight[0][0].numpy()
plt.imshow(test.reshape(5,5))
plt.show()

test = net.features[3].weight[1][0].numpy()
plt.imshow(test.reshape(5,5))
plt.show()

# test = net.features[10].weight[2][0].numpy()
# plt.imshow(test.reshape(11,11))
# plt.show()
#
# test = net.features[10].weight[3][0].numpy()
# plt.imshow(test.reshape(11,11))
# plt.show()
#
# test = net.features[10].weight[4][0].numpy()
# plt.imshow(test.reshape(11,11))
# plt.show()

# test = net.features[0].weight[4][1].numpy()
# plt.imshow(test.reshape(11,11))
# plt.show()
#
# test = net.features[0].weight[4][2].numpy()
# plt.imshow(test.reshape(11,11))
# plt.show()
