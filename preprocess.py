import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_classes = len(classes)
transformer = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
class PartialLabelCifarData(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        trainset = CIFAR10(root='./data', train=True, download=True, transform=transformer)
        validation_size = 10000

        # get uniform index
        def next_index(last_index = -1):
            ix = last_index
            while ix == last_index:
                ix = np.random.randint(validation_size, len(trainset))
            return ix

        self.validation_data = []
        for i in range(validation_size):
            label = torch.zeros(n_classes)
            label[trainset[i][1]] = 1
            self.validation_data.append((trainset[i][0], label))
        self.data = []
        for i in range(5000):
            label = torch.zeros(n_classes)
            ix1 = next_index()
            ix2 = next_index(ix1)
            while trainset[ix1][1] == trainset[ix2][1]:
                ix2 = next_index(ix1)
            label[trainset[ix1][1]] = 1
            label[trainset[ix2][1]] = 1
            self.data.append(( trainset[ix1][0], label))
            self.data.append(( trainset[ix2][0], label))
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
