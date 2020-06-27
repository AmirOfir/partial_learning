import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from preprocess import OneHotLabelCifarData, PartialLabelCifarData, classes, n_classes, get_class_performance
from net import create_net

def optimizeTrainingData(model : nn.Module, valset: OneHotLabelCifarData, trainset: PartialLabelCifarData, threshold: float=0.1):
    class_perf = get_class_performance(model, valset)
    optimized_class = torch.argmax(class_perf)
    ground_truth_label = torch.zeros(n_classes); ground_truth_label[optimized_class] = 1

    count = 0
    for i in len(trainset):
        if (trainset[i][1] == ground_truth_label):
            output_label = model(trainset[i][0])
            if (torch.argmax(output_label) == optimized_class and output_label[optimized_class] > threshold): # We proably have the right item
                trainset[i][1] = torch.clone(ground_truth_label)
                count += 1
    print('corrected', count, 'labels')
    return count

def partialStep(model, trainloader, epochs, optimizer, criterion, early_stop, lr_scheduler):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            
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