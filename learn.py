import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from regular import train_regular
from preprocess import PartialLabelCifarData, transformer, classes, n_classes, get_class_performance, test_performance
from net import create_net

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

trainset = PartialLabelCifarData()

# Get max classes
print("regular training")
regular_optimizer = createOptimizer(model, learning_rate, hyperparameters)
train_regular(epochs, model, regular_optimizer, nn.L1Loss(), \
    optim.lr_scheduler.StepLR(optimizer=regular_optimizer, step_size=1, gamma=0.9),
    len(trainset.data), len(trainset.validation_data), batch_size, early_stop)
get_class_performance(model, trainset.validation_data)
test_performance(model)

exit()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

model.to(device)

criterion = nn.L1Loss()
optimizer = createOptimizer()

lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.9)

def train(epochs, trainloader, net, optimizer, criterion, lr_scheduler=None, early_stop=0):
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            # if (epoch == 4):
            #     print(outputs, torch.argmax(outputs, dim=1), labels)
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

train(epochs, trainloader, model, optimizer, criterion, lr_scheduler)
print('Finished Training')