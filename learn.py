import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess import PartialLabelCifarData, transformer, classes, n_classes

epochs = 10
batch_size = 1
learning_rate = 0.1
momentum = 0.9
early_stop = 0.02

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset = PartialLabelCifarData()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

class CNN(nn.Module):
    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x

net = CNN()
net.to(device)

def createOptimizer():
    global net
    return optim.SGD(net.parameters(), lr=learning_rate)
    # return optim.Adam(net.parameters(), lr=learning_rate)
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

train(epochs, trainloader, net, optimizer, criterion, lr_scheduler)
print('Finished Training')

# Test Accuracy
testset = CIFAR10(root='./data', train=False, download=True, transform=transformer)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)

        # print(outputs, torch.argmax(outputs))
        # print(labels)
        # exit()
        # _, predicted = torch.max(outputs.data, 1)
        # print (predicted, labels)
        total += labels.size(0)
        correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))                              