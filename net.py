import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def parse_model_config(path):
    """Parses the layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('#'):
            pass
        elif line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            #print(value, key.rstrip(), key)
            module_defs[-1][key.rstrip()] = value

    return module_defs
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            if kernel_size == 3:
                modules.add_module(
                    f"conv_{module_i}",
                    nn.Conv2d(
                        in_channels=output_filters[-1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=int(module_def["stride"]),
                        padding=pad,
                        bias=not bn)
               )
            else:
              pad_left = pad
              pad_right = pad if pad == (kernel_size - 1) / 2 else pad + 1
              if (pad_left != pad_right):
                  modules.add_module(
                      f"pad_{module_i}",
                      nn.ZeroPad2d((pad_left, pad_right, pad_left, pad_right))
                  )
              modules.add_module(
                  f"conv_{module_i}",
                  nn.Conv2d(
                      in_channels=output_filters[-1],
                      out_channels=filters,
                      kernel_size=kernel_size,
                      stride=int(module_def["stride"]),
                      bias=not bn,
                  ),
               )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
        elif module_def["type"] == "relu":
            modules.add_module(f"relu_{module_i}", nn.ReLU())
        elif module_def["type"] == "softmax":
            modules.add_module(f"softmax_{module_i}", nn.Softmax())
        elif module_def["type"] == "avgpool":
            modules.add_module(f"avgpool_{module_i}", nn.AvgPool1d())
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)
        elif module_def["type"] == "dropout":
            p = float(module_def["p"])
            modules.add_module(f"dropout{module_i}", nn.Dropout2d(p))
        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)
        elif module_def["type"] == "linear":
            i = int(module_def["in"])
            o = int(module_def["out"])
            modules.add_module(f"linear{module_i}", nn.Linear(i,o))
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
        elif module_def["type"] == "flatten":
            modules.add_module(f"flatten_{module_i}", Flatten())
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list    