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
from partial import partialStep, optimizeTrainingData, notLearnedLabels

def createOptimizer(net, learning_rate, hyperparameters):
    optimizer_name = hyperparameters.get("optimizer", "SGD")
    if optimizer_name == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=hyperparameters.get("SGD_momentun", 0.9))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=hyperparameters.get("learning_rate_change_epochs", 1000), gamma=hyperparameters.get("learning_rate_decay", 0.1))
    return lr_scheduler, optimizer
def adjustLearningRate(optimizer, epoch:int, hyperparameters:dict):
    set_lr = False
    lr = hyperparameters.get("learning_rate")
    if "learning_rate_decay" in hyperparameters.keys():
        d = hyperparameters.get("learning_rate_decay", 0.1)
        n = hyperparameters.get("learning_rate_change_epochs", 10)
        if (epoch % n == 0):
            lr = lr * d * (epoch / n) # For example: 0.1 * 0.1 * (20/10)
            set_lr = True
    if epoch == 0 or set_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

hyperparameters, model = create_net()
epochs = hyperparameters.get("epochs", 5)
batch_size = hyperparameters.get("batch_size", 5)
learning_rate = hyperparameters.get("learning_rate", 0.1)
early_stop = hyperparameters.get("early_stop", 0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
label_learning_epochs = hyperparameters.get("label_learning_epochs", 20)

dataset = CIFAR10(root='./data', train=True, download=True, transform=transformer)
validation_size = 10000
train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - validation_size, validation_size])
valset = OneHotLabelCifarData(val_data)
trainset = PartialLabelCifarData(train_data)

def netMaxResults():
    print("regular training")
    lr_scheduler, optimizer = createOptimizer(model, learning_rate, hyperparameters)
    trainset = OneHotLabelCifarData(train_data)
    train_regular(trainset, epochs, model, optimizer, nn.L1Loss(), lr_scheduler, batch_size, early_stop)
    get_class_performance(model, valset)
    test_performance(model)

# Reset
hyperparameters, model = create_net()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
model.to(device)
criterion = nn.L1Loss().to(device)

label_learning_epoch = 0
not_learned_labels = 5000
while (label_learning_epoch < label_learning_epochs and not_learned_labels > 100):
    print("label learning epoch", label_learning_epoch, "for learning", not_learned_labels, "labels")
    # Train
    lr_scheduler, optimizer = createOptimizer(model, learning_rate, hyperparameters)
    partialStep(model, trainloader, epochs, optimizer, criterion, early_stop, lr_scheduler)

    # Optimize
    count = optimizeTrainingData(model, valset, trainset)

    not_learned_labels = notLearnedLabels(trainset)
    label_learning_epoch += 1

test_performance(model)    

# Train after the correction
lr_scheduler, optimizer = createOptimizer(model, learning_rate, hyperparameters)
train_regular(trainset, epochs, model, optimizer, nn.L1Loss(), lr_scheduler, batch_size, early_stop)
test_performance(model)

print('Finished Training')