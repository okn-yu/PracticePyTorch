import cv2
import torch
from matplotlib import pyplot as plt

img = cv2.imread('./data/hymenoptera_data/train/ants/0013035.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.imshow(img)
# plt.show()

root = 'hymonoptera_data'

class CustomDataset(torch.util.data.Dataset):
    classes = ['ant', 'bee']

    def __init__(self, root, transform=None, train=True):
        pass
