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
    @staticmethod
    # get uniform index
    def next_index(_min, _max, last_index = -1):
        ix = last_index
        while ix == last_index:
            ix = np.random.randint(_min, _max)
        return ix
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dataset = CIFAR10(root='./data', train=True, download=True, transform=transformer)
        validation_size = 10000
        train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - validation_size, validation_size])

        self.validation_data = []
        for i in range(validation_size):
            label = torch.zeros(n_classes)
            label[val_set[i][1]] = 1
            self.validation_data.append((val_set[i][0], label))

        data_indexes = []
        self.data = []
        while (len(self.data) < 5000):
            label = torch.zeros(n_classes)
            ix1 = PartialLabelCifarData.next_index(0, len(train_set))
            ix2 = PartialLabelCifarData.next_index(0, len(train_set), ix1)
            if ix1 not in data_indexes and ix2 not in data_indexes \
                and train_set[ix1][1] != train_set[ix2][1]:
                label[train_set[ix1][1]] = 1
                label[train_set[ix2][1]] = 1
                self.data.append(( train_set[ix1][0], label ))
                self.data.append(( train_set[ix2][0], label ))
                data_indexes.append(ix1)
                data_indexes.append(ix2)
        # for i in range(5000):
        #     label = torch.zeros(n_classes)
        #     ix1 = next_index(0, len(train_set))
        #     ix2 = next_index(0, len(train_set), ix1)

        #     while train_set[ix1][1] == train_set[ix2][1]:
        #         ix2 = next_index(ix1)
        #     label[trainset[ix1][1]] = 1
        #     label[trainset[ix2][1]] = 1
        #     self.data.append(( trainset[ix1][0], label))
        #     self.data.append(( trainset[ix2][0], label))
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def get_class_performance(net, data_set):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0)
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))

    with torch.no_grad():
        for data in loader:
            image, label = data[0].to(device), data[1].to(device)
            output = net(image)
            predicted = torch.argmax(output, dim=1)
            predicted_label = torch.zeros(n_classes)
            predicted_label[predicted] = 1
            if (label == predicted_label):
                class_correct[predicted] += 1
            class_total[predicted] += 1

    for i in range(n_classes):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    
def test_performance(net):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testset = CIFAR10(root='./data', train=False, download=True, transform=transformer)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            total += labels.size(0)
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))     