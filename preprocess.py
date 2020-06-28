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

def softmaxToOneHot(tensor):
    label = torch.zeros_like(tensor)
    for i in range(tensor.size(0)):
        predicted = torch.argmax(tensor, dim=1)
        label[i][predicted] = 1
    return label

class OneHotLabelCifarData(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data = []
        for i in range(len(dataset)):
            label = torch.zeros(n_classes)
            label[dataset[i][1]] = 1
            self.data.append((dataset[i][0], label))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class PartialLabelCifarData(torch.utils.data.Dataset):
    @staticmethod
    # get uniform index
    def next_index(_min, _max, last_index = -1):
        ix = last_index
        while ix == last_index:
            ix = np.random.randint(_min, _max)
        return ix
    def __init__(self, train_set):
        super().__init__()

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
                self.data.append(( train_set[ix1][0], label, train_set[ix1][1] ))
                self.data.append(( train_set[ix2][0], label, train_set[ix2][1] ))
                data_indexes.append(ix1)
                data_indexes.append(ix2)    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def get_class_performance(net, data_set, print_result=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0)
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))

    net.eval()

    for data in loader:
        image, label = data[0].to(device), torch.argmax(data[1].cpu()).item()
        output = net(image)

        predicted = torch.argmax(output, dim=1).cpu().item()
        
        if label == predicted:
            class_correct[label] += 1
        class_total[label] += 1
    class_perf = []
    print(class_total, class_correct)
    for i in range(n_classes):
        perf = 0 if class_total[i] == 0 else 100 * class_correct[i] / class_total[i]
        class_perf.append(perf)
        if (print_result):
            print( f'Accuracy of {classes[i]} : {perf}%' )
    return class_perf
    
def test_performance(net):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.eval()
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