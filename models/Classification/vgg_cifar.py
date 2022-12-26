import math
import torch
import torch.nn as nn
from torch.autograd import Variable
#from .masklayer import *
#from newlayers import *

__all__ = ['vgg', 'vgg16_cifar']

cfg = {
    '16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    }

class vgg(nn.Module):
    def __init__(self, dataset, batch_norm=True, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg == None:
            cfg = globals()['cfg']['16']
        print(cfg)
        n_f = 0
        for f in cfg:
            if type(f) == int:
                n_f += f
        print('total pruned filter ratio: ', 1.0-(n_f/4224))

        self.feature = self.make_layers(cfg, batch_norm) #prune

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Linear(cfg[-1], num_classes) #nn.Sequential(nn.Linear(cfg[-1], num_classes))
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=True): #prune
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def vgg16_cifar(dataset, **kwargs):
    model = vgg(dataset=dataset, batch_norm=True, cfg=cfg['16'], **kwargs)
    return model
