import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from preprocess import OneHotLabelCifarData, PartialLabelCifarData, classes, n_classes, get_class_performance
from net import create_net

def notLearnedLabels(trainset : PartialLabelCifarData):
    count = 0
    for i in range(len(trainset)):
        # If the item is marked as having this class and another
        count += 1 if torch.sum(trainset[i][1]) == 2 else 0
    return count

def optimizeClass(model : nn.Module, trainset : PartialLabelCifarData, optimized_class:int, prediction_threshold:float):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimized_entities = 0
    ground_truth_label = torch.zeros(n_classes); ground_truth_label[optimized_class] = 1
    print ('optimizing class', classes[optimized_class])
    for i in range(len(trainset)):
        # If the item is marked as having this class and another
        if torch.sum(trainset[i][1]) == 2 and torch.sum(torch.mul(trainset[i][1], ground_truth_label) == 1):
            
            # Get the prediction is high for this class
            item : torch.Tensor = trainset[i][0]
            if item.dim() == 3: item = item.unsqueeze(0)
            output_label = model(item.to(device)).squeeze(0)
            output_label = torch.softmax(output_label,0)

            if torch.argmax(output_label).item() == optimized_class:
                if torch.max(output_label).item() >= prediction_threshold:
                    print('Hey,', i, 'is a', classes[trainset[i][2]], 'classified as', classes[torch.argmax(output_label).item()], 'with prob.', torch.max(output_label).item() )
                    trainset.data[i] = (trainset[i][0], torch.clone(ground_truth_label))
                    optimized_entities += 1
            # elif output_label[optimized_class] <= 0.1:
            #     # This is not an example of <class>
            #     trainset.data[i][1][optimized_class] = 0
            #     optimized_entities += 1
            # print('result:', trainset.data[i])
    return optimized_entities
def optimizeTrainingData(model : nn.Module, valset: OneHotLabelCifarData, trainset: PartialLabelCifarData, prediction_threshold: float=0.15, min_perf=0.5):
    model.eval()
    class_perf = get_class_performance(model, valset)
    highest = torch.argmax(torch.Tensor(class_perf))
    optimized_entities = 0
    for c in range(n_classes):
        if (class_perf[c] > min_perf or class_perf[c] == highest):
            optimized_entities += optimizeClass(model, trainset, c, prediction_threshold)

    print('corrected', optimized_entities, 'labels')
    return optimized_entities

def partialStep(model:nn.Module, trainloader:PartialLabelCifarData, epochs:int, optimizer, criterion, early_stop, lr_scheduler):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()

    for epoch in range(epochs):
        print('partial label learning epoch', epoch)
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
            if i % 100 == 99:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1000))
                if (early_stop != 0):
                    if (running_loss / 1000 <= early_stop):
                        return
                running_loss = 0.0

        if (not lr_scheduler is None):
            lr_scheduler.step()