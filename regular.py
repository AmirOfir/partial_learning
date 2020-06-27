import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from preprocess import PartialLabelCifarData, transformer, classes, n_classes
from net import create_net

def train_regular(epochs, net, optimizer, criterion, lr_scheduler, learning_size, validation_size, batch_size, early_stop):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    dataset = CIFAR10(root='./data', train=True, download=True, transform=transformer)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - validation_size, validation_size])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            labels = torch.zeros(batch_size, n_classes, device=device)
            inputs, groundTruth = data[0].to(device), data[1].to(device)
            for i in range(batch_size):
                labels[i][groundTruth[i]] = 1
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            labels.to(device)
            

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1000))
                if (early_stop != 0):
                    if (running_loss / 1000 <= early_stop):
                        return
                running_loss = 0.0

        if (not lr_scheduler is None):
            lr_scheduler.step()