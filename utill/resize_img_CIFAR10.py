import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms.ToTensor(),
                                             download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=2)


# 参照元
# https://qiita.com/kazetof/items/6a72926b9f8cd44c218e
# CIFAR10:画像サイズは32*32
# CNN利用時の4次元テンソルのサイズの拡大縮小を行う
# ただし学習途中で画像サイズを変更した場合は処理が失敗する
# datasetおよびloaderの作り直しが必要らしい

def resize_img(img, rate):
    tensor_list = []

    for i in range(img.shape[0]):
        tensor_list.append(_resize_img(img[i], rate))

    return torch.stack(tensor_list, dim=0)


def _resize_img(img, rate):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    H, W, C = img.shape
    print(H, W, C)

    H = int(rate * H)
    W = int(rate * W)

    y = np.arange(H).repeat(W).reshape(W, -1)
    x = np.tile(np.arange(W), (H, 1))

    y = np.floor(y / rate).astype(np.uint8)
    x = np.floor(x / rate).astype(np.uint8)

    resized_img = img[y, x]
    resized_img = np.transpose(resized_img, (2, 0, 1))

    print(resized_img.shape)

    return torch.tensor(resized_img)


def imshow(img):
    # 入力範囲を[-1, 1] から [0, 1] に変更
    img = img / 2 + 0.5
    npimg = img.numpy()

    # 要素の順番を(RGB, H, W) から (H, W, RGB)に変更
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()


# iter()でDataLoaderで定義された__iter__が呼ばれ，DataLoaderIterを返す
# dataiter.next()を呼ぶごとにnバッチ目，n+1バッチ目と繰り返しデータを取得
dataiter = iter(train_loader)

# print(type(train_loader))
# <class 'torch.utils.data.dataloader.DataLoader'>
# print(type(dataiter))
# <class 'torch.utils.data.dataloader.DataLoaderIter'>
images, labels = dataiter.next()

# print(labels)
# tensor([5, 8, 6, 2, 5, 1, 0, 6, 0, 1, 6, 3, 7, 2, 7, 2, 7, 1, 1, 8, 9, 1, 9, 7,
#         7, 3, 7, 5, 4, 4, 8, 7, 2, 5, 8, 7, 3, 3, 5, 1, 5, 0, 0, 9, 4, 2, 8, 8,
#         6, 1, 1, 4, 0, 2, 7, 3, 6, 6, 2, 0, 2, 2, 2, 8])

# torch.Size([16, 3, 32, 32])
# print(images.size())

resized_images = resize_img(images, 7)
imshow(torchvision.utils.make_grid(resized_images, nrow=4))
# imshow(torchvision.utils.make_grid(images, nrow=4))
