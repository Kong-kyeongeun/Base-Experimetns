import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['densenet','densenet40']

"""
densenet with basic block.
"""

class BasicBlock(nn.Module):
    def __init__(self, inplanes, cfg, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, cfg, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class densenet(nn.Module):
    def __init__(self, depth=40, 
        dropRate=0, dataset='cifar10', growthRate=12, compressionRate=1, cfg = None):
        super(densenet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3
        block = BasicBlock

        self.growthRate = growthRate
        self.dropRate = dropRate

        if cfg == None:
            cfg = [growthRate]*(3*n)

        assert len(cfg) == 3*n, 'length of config variable cfg should be 3n+3'

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n, cfg[0:n])
        self.trans1 = self._make_transition(compressionRate,1)
        self.dense2 = self._make_denseblock(block, n, cfg[n:2*n])
        self.trans2 = self._make_transition(compressionRate,2)
        self.dense3 = self._make_denseblock(block, n, cfg[2*n:3*n])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(self.inplanes, 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(self.inplanes, 100)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, cfg):
        layers = []
        assert blocks == len(cfg), 'Length of the cfg parameter is not right.'
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, cfg = cfg[i], growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += cfg[i]

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, trans_num):
        # cfg is a number in this case.
        inplanes = self.inplanes
        outplanes = int(math.floor((self.growthRate*12*trans_num +2*self.growthRate)//compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def densenet40(dataset, cfg=None, **kwargs):
    model = densenet(dataset=dataset, cfg =cfg, **kwargs)
    return model
#-------------------------------------------------------------------#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as cp
# from collections import OrderedDict

# __all__ = ['DenseNet', 'densenet40k12']

# def _bn_function_factory(norm, relu, conv):
#     def bn_function(*inputs):
#         concated_features = torch.cat(inputs, 1)
#         bottleneck_output = conv(relu(norm(concated_features)))
#         return bottleneck_output
#     return bn_function

# class _DenseLayer(nn.Module):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
#         super(_DenseLayer, self).__init__()
#         self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
#         self.add_module('relu1', nn.ReLU(inplace=True)),
#         self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
#                         kernel_size=1, stride=1, bias=False)),
#         self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
#         self.add_module('relu2', nn.ReLU(inplace=True)),
#         self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
#                         kernel_size=3, stride=1, padding=1, bias=False)),
#         self.drop_rate = drop_rate
#         self.efficient = efficient

#     def forward(self, *prev_features):
#         bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
#         if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
#             bottleneck_output = cp.checkpoint(bn_function, *prev_features)
#         else:
#             bottleneck_output = bn_function(*prev_features)
#         new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
#         return new_features

# class _Transition(nn.Sequential):
#     def __init__(self, num_input_features, num_output_features):
#         super(_Transition, self).__init__()
#         self.add_module('norm', nn.BatchNorm2d(num_input_features))
#         self.add_module('relu', nn.ReLU(inplace=True))
#         self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
#                                           kernel_size=1, stride=1, bias=False))
#         self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

# class _DenseBlock(nn.Module):
#     def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
#         super(_DenseBlock, self).__init__()
#         for i in range(num_layers):
#             layer = _DenseLayer(
#                     num_input_features + i * growth_rate,
#                     growth_rate=growth_rate,
#                     bn_size=bn_size,
#                     drop_rate=drop_rate,
#                     efficient=efficient,
#             )
#             self.add_module('denselayer%d' %(i + 1), layer)

#     def forward(self, init_features):
#         features = [init_features]
#         for name, layer in self.named_children():
#             new_features = layer(*features)
#             features.append(new_features)
#         return torch.cat(features, 1)

# class DenseNet(nn.Module):
#     def __init__(self, dataset, growth_rate=12, block_config=[6, 6, 6], compression=0.5,
#                  num_init_features=24, bn_size=4, drop_rate=0,
#                  small_inputs=True, efficient=False, KD=False):
#         super(DenseNet, self).__init__()
#         if dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError("Dataset conflict!!..")

#         assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
#         self.avgpool_size = 8 if small_inputs else 7
#         self.KD = KD

#         if small_inputs:
#             self.features = nn.Sequential(OrderedDict([
#                 ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
#             ]))
#         else:
#             self.features = nn.Sequential(OrderedDict([
#                 ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
#             ]))
#             self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
#             self.features.add_module('relu0', nn.ReLU(inplace=True))
#             self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
#                                                            ceil_mode=False))

#         num_features = num_init_features
#         for i, num_layers in enumerate(block_config):
#             block = _DenseBlock(
#                 num_layers=num_layers,
#                 num_input_features=num_features,
#                 bn_size=bn_size,
#                 growth_rate=growth_rate,
#                 drop_rate=drop_rate,
#                 efficient=efficient,
#             )
#             self.features.add_module('denseblock%d' %(i + 1), block)
#             num_features = num_features + num_layers * growth_rate
#             if i != len(block_config) - 1:
#                 trans = _Transition(num_input_features=num_features,
#                                     num_output_features=int(num_features * compression))
#                 self.features.add_module('transition%d' % (i + 1), trans)
#                 num_features = int(num_features * compression)

#         self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
#         self.classifier = nn.Linear(num_features, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         features = self.features(x)
#         # D40K12  B x 132 x 8 x 8
#         # D100K12 B x 342 x 8 x 8
#         # D100K40 B x 1126 x 8 x 8
#         x = F.relu(features, inplace=True)
#         x_f = F.avg_pool2d(x, kernel_size=self.avgpool_size).view(features.size(0), -1) # B x 132
#         x = self.classifier(x_f)
#         return x

# def densenet40k12(dataset):
#     model = DenseNet(dataset=dataset, growth_rate=12, block_config=[6, 6, 6])
#     return model
