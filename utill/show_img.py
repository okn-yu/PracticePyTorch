import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)

# 参照元
# https://qiita.com/kazetof/items/6a72926b9f8cd44c218e

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# iter()でDataLoaderで定義された__iter__が呼ばれ，DataLoaderIterを返す
# dataiter.next()を呼ぶごとにnバッチ目，n+1バッチ目と繰り返しデータを取得
dataiter = iter(train_loader)

# print(type(train_loader))
# <class 'torch.utils.data.dataloader.DataLoader'>
# print(type(dataiter))
# <class 'torch.utils.data.dataloader.DataLoaderIter'>
images, labels = dataiter.next()

# 8x8の格子状に画像を表示
#imshow(images[0])
imshow(torchvision.utils.make_grid(images))
