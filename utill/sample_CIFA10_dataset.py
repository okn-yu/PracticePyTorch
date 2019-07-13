import os
import cv2
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torchvision.models as models


class CustomDataset(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CustomDataset, self).__init__(root, train=train, transform=transform, target_transform=target_transform,
                                            download=download)

    def __getitem__(self, index):
        img, target = super(CustomDataset, self).__getitem__(index)
        img = _resize_img(img, 7)

        return img, target


def resize_img(img, rate):
    tensor_list = []

    for i in range(img.shape[0]):
        tensor_list.append(_resize_img(img[i], rate))

    return torch.stack(tensor_list, dim=0)


def _resize_img(img, rate):
    img = img.numpy()
    # print(img.shape)
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


train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms.ToTensor(),
                                             download=False)
test_dataset = CustomDataset(root='./data/', train=True, transform=transforms.ToTensor(),
                             download=False)

CIFAR10_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=5, shuffle=True)


# print(train_dataset.__dict__.keys())
# print(test_dataset.__dict__.keys())


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


net = models.alexnet(pretrained=True)


for i, (images, labels) in enumerate(CIFAR10_loader):
    # print(type(images))
    # print(labels.numpy())
    # images = resize_img(images, 7)
    # show(torchvision.utils.make_grid(images, padding=1))
    # plt.axis('off')

    outputs = net(images)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    # loss.backward()
    print(loss, loss.backward())
    break
