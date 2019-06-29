import os
import cv2
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

to_tensor_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

img = cv2.imread('./data/hymenoptera_data/train/ants/0013035.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.imshow(img)
# plt.show()

root = './data/hymenoptera_data'


class CustomDataset(torch.utils.data.Dataset):
    classes = ['ant', 'bee']

    def __init__(self, root, transform=None, train=True):
        self.transform = transform
        self.images = []
        self.labels = []
        root = './data/hymenoptera_data'

        if train == True:
            root_ants_path = os.path.join(root, 'train', 'ants')
            root_bees_path = os.path.join(root, 'train', 'bees')
        else:
            root_ants_path = os.path.join(root, 'val', 'ants')
            root_bees_path = os.path.join(root, 'val', 'bees')

        ant_images = os.listdir(root_ants_path)
        ant_labels = [0] * len(ant_images)

        bee_images = os.listdir(root_bees_path)
        bee_labels = [1] * len(bee_images)

        for image, label in zip(ant_images, ant_labels):
            self.images.append(os.path.join(root_ants_path, image))
            self.labels.append(label)

        for image, label in zip(bee_images, bee_labels):
            self.images.append(os.path.join(root_bees_path, image))
            self.labels.append(label)


    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):

        return len(self.images)


custom_dataset = CustomDataset(root, to_tensor_transform, train=True)
custom_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=5, shuffle=True)


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


for i, (images, labels) in enumerate(custom_loader):
    # print(type(images))
    print(labels.numpy())
    show(torchvision.utils.make_grid(images, padding=1))
    plt.axis('off')

    break
