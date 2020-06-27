import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from regular import train_regular
from preprocess import OneHotLabelCifarData, PartialLabelCifarData, transformer, classes, n_classes, get_class_performance, test_performance
from net import create_net
from partial import partialStep, optimizeTrainingData

def createOptimizer(net, learning_rate, hyperparameters):
    optimizer_name = hyperparameters.get("optimizer", "SGD")
    if optimizer_name == "Adam":
        return optim.Adam(net.parameters(), lr=learning_rate)
    else:
        return optim.SGD(net.parameters(), lr=learning_rate, momentum=hyperparameters.get("SGD_momentun", 0.9))

hyperparameters, model = create_net()
epochs = hyperparameters.get("epochs", 5)
batch_size = hyperparameters.get("batch_size", 5)
learning_rate = hyperparameters.get("learning_rate", 0.1)
early_stop = hyperparameters.get("early_stop", 0.02)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = CIFAR10(root='./data', train=True, download=True, transform=transformer)
validation_size = 10000
train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - validation_size, validation_size])
valset = OneHotLabelCifarData(val_data)
trainset = PartialLabelCifarData(train_data)

# Get max classes
# print("regular training")
# regular_optimizer = createOptimizer(model, learning_rate, hyperparameters)
# train_regular(train_data, epochs, model, regular_optimizer, nn.L1Loss(), \
#     optim.lr_scheduler.StepLR(optimizer=regular_optimizer, step_size=1, gamma=0.9), batch_size, early_stop)
# get_class_performance(model, valset)
# test_performance(model)

# Reset
hyperparameters, model = create_net()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
model.to(device)
criterion = nn.L1Loss()
optimizer = createOptimizer(model, learning_rate, hyperparameters)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.9)

total = 0
partialStep(model, trainloader, epochs, optimizer, criterion, early_stop, lr_scheduler)
count = optimizeTrainingData(model, valset, trainset)
total += count
while (count > 0 and total < 5000):
    partialStep(model, trainloader, epochs, optimizer, criterion, early_stop, lr_scheduler)
    count = optimizeTrainingData(model, valset, trainset)
    total += count
test_performance(model)    

print('Finished Training')