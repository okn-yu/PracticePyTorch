import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt

# transformsはデータ拡張・リサイズ・正規化などの前処理を実施
train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

# datasetsは画像とラベルのペアを返却
image, label = train_dataset[0]
# print(image.size()) # size()はnumpyでのshape相当
# print(label)

# DataSet -> DatasetLoaderの順に処理
# num_workersはデータのロードを行うプロセス数
# 0の場合はmainプロセスにてロード
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle = True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle = False, num_workers=2)

# バッチサイズの個数に相当する画像とラベルを取得
for images, labels in train_loader:
    print(image.size())
    print(images[0].size())
    print(labels.size())
    break

num_classes = 10

# nn.Module: Base class for all neural network modules.
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html
# https://pytorch.org/docs/stable/nn.html
# APIとしてforwardは実装されているがbackwardは実装されていないことに注意
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        # nn.Linearは線形変換
        # 引数には入力次元と出力次元
        # バイアスもデフォルトで設定される
        self.fc1 = nn.Linear(32 * 32 * 3, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, num_classes)
        # nn.Dropout2dは2次元ドロップアウトレイヤー
        # 1次元ドロップアウトや3次元ドロップアウトも存在
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return F.relu(self.fc3(x))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = MLPNet().to(device)

criterion = nn.CrossEntropyLoss()

# https://pytorch.org/docs/stable/optim.html
# 第1引数: iterable of parameters to optimize or dicts defining parameter groups
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay =5e-4)

print(net.parameters())

num_epocs = 50

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoc in range(num_epocs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    net.train()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.view(-1, 32 * 32 * 3).to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc += ((outputs.max(1))[1] == labels).sum().item()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)

    net.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.view(-1, 32 * 32 * 3).to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)

    print('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
      .format(epoc + 1, num_epocs, i + 1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)