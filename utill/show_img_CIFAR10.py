import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=2)


# 参照元
# https://qiita.com/kazetof/items/6a72926b9f8cd44c218e
# CIFAR10:画像サイズは32*32
# CNN利用時には4次元テンソルを用いるのでこの方法を直接は利用できない

def resize_img(img, rate):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    H, W, C = img.shape

    H = int(rate * H)
    W = int(rate * W)

    y = np.arange(H).repeat(W).reshape(W, -1)
    x = np.tile(np.arange(W), (H, 1))

    y = np.floor(y / rate).astype(np.uint8)
    x = np.floor(x / rate).astype(np.uint8)

    resized_img = img[y, x]
    resized_img = np.transpose(resized_img, (2, 0, 1))

    return torch.tensor(resized_img)


def imshow(img):
    # batch_sizeが1の場合:
    print("before resize: " + str(img.size()))
    # torch.Size([3, 34, 34])

    img = resize_img(img, 6.6)
    print("after resize: " + str(img.size()))

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

# 8x8の格子状に画像を表示
# imshow(images[0])
imshow(torchvision.utils.make_grid(images, nrow=4))
