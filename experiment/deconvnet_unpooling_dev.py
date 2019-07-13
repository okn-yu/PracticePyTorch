# Sample code for convolution -> ReLu -> pooling -> pooling -> unpooling -> ReLu -> deconvolution deconvolution.

import cv2
import torch
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

def resize_img(img, rate):
    tensor_list = []

    for i in range(img.shape[0]):
        tensor_list.append(_resize_img(img[i], rate))

    return torch.stack(tensor_list, dim=0)


def _resize_img(img, rate):
    img = img.numpy()
    #img = img.detuch().numpy()
    img = np.transpose(img, (1, 2, 0))
    H, W, C = img.shape
    # print(H, W, C)

    H = int(rate * H)
    W = int(rate * W)

    y = np.arange(H).repeat(W).reshape(W, -1)
    x = np.tile(np.arange(W), (H, 1))

    y = np.floor(y / rate).astype(np.uint8)
    x = np.floor(x / rate).astype(np.uint8)

    resized_img = img[y, x]
    resized_img = np.transpose(resized_img, (2, 0, 1))

    # print(resized_img.shape)

    return torch.tensor(resized_img)


def imshow(img):
    print(img.shape)
    print(type(img))

    # 入力範囲を[-1, 1] から [0, 1] に変更
    img = img / 2 + 0.5
    npimg = img.numpy()
    #npimg = img

    # 要素の順番を(RGB, H, W) から (H, W, RGB)に変更
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()

# AlexNetの第1層
net = models.alexnet(pretrained=True)
first_conv_layer = net.features[0]
first_pool_layer = net.features[1]
relu_function = net.features[4]

train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms.ToTensor(),
                                             download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=2)

# CIFAR10から適当な画像を1枚取得 -> (16 * 3 * 32 * 32)

dataiter = iter(train_loader)
images, labels = dataiter.next()

# 7倍に拡大する -> (16 * 3 * 224 * 224)
images = resize_img(images, 7)
imshow(torchvision.utils.make_grid(images, nrow=4))

# 第1層を用いて畳み込みを行う
img_input = Variable(images)
raw_result = first_conv_layer(img_input)

# ReLu関数
relu_result = relu_function(img_input)
imshow(torchvision.utils.make_grid(relu_result[0][1].data, nrow=4))

# MaxPooling
pool_result = first_pool_layer(relu_result)
imshow(torchvision.utils.make_grid(pool_result[0][1].data, nrow=4))

# MaxUnPooling
unpool = torch.nn.MaxUnpool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)


# 逆畳み込みを実施してみる
deconv_layer = torch.nn.ConvTranspose2d(64, 3, (11, 11), stride=(4, 4), padding=(2, 2))
deconv_layer.weight = first_conv_layer.weight
result = deconv_layer(raw_result)

print(deconv_layer.weight)
print(result.shape)
print(type(result)) # -> <class 'torch.Tensor'>

# Variable化されたTensorの可視化ではdataをつける必要がある
# https://discuss.pytorch.org/t/how-to-transform-variable-into-numpy/104
# Variable's can’t be transformed to numpy, because they’re wrappers around tensors that save the operation history, and numpy doesn’t have such objects. You can retrieve a tensor held by the Variable, using the .data attribute. Then, this should work: var.data.numpy().

imshow(torchvision.utils.make_grid(result.data, nrow=4))
