import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet_Cifar', 'resnet20', 'resnet32', 'resnet56', 'resnet110']

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, cfg, dropRate, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, cfg[0], stride)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv2 = conv3x3(cfg[0], cfg[1])
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride
        self.dropRate = dropRate

        if stride != 1 or inplanes != cfg[1] * BasicBlock.expansion:
            self.convShortcut = nn.Conv2d(inplanes, cfg[1] * BasicBlock.expansion,
                                    kernel_size=1, stride=stride, bias=False)
            self.bnShortcut = nn.BatchNorm2d(cfg[1] * BasicBlock.expansion)
            self.downsample = True


    def forward(self, x):
        x = F.relu(x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.convShortcut(x)
            residual = self.bnShortcut(residual)

        out += residual

        return out

class ResNet_Cifar(nn.Module):
    def __init__(self, dataset, depth, dropRate, block=BasicBlock, cfg=None):
        super(ResNet_Cifar, self).__init__()
        n = (depth - 2) // 6
        block = BasicBlock
        self.block = block
        filters = [[16], [16, 16]*n, [32, 32]*n, [64, 64]*n]
        filters = [item for sub_list in filters for item in sub_list]

        if cfg is None:
            cfg = [[16], [16, 16]*n, [32, 32]*n, [64, 64]*n]
            cfg = [item for sub_list in cfg for item in sub_list]
        print(cfg, len(cfg))
        print('ResNet_Cifar: Depth {}, Layers for each block: {}, dropRate: {}, Original Filters: {}, Pruned Filters: {}, Pruned Rate: {}'.format(depth, n, dropRate, sum(filters) ,sum(cfg), 1-(sum(cfg)/sum(filters))))
        
        #self.inplanes = 16
        self.inplanes = cfg[0]
        self.dropRate = dropRate

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise ValueError("No valid dataset is given.")

        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1, self.inplanes = self._make_layer(block, n, cfg[1:2*n+1], self.dropRate)
        self.layer2, self.inplanes = self._make_layer(block, n, cfg[2*n+1:4*n+1], self.dropRate, stride=2)
        self.layer3, self.inplanes = self._make_layer(block, n, cfg[4*n+1:6*n+1], self.dropRate, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(cfg[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, blocks, cfg, dropRate, stride=1):
        downsample = None
        reg = 0

        layers = []
        layers.append(block(self.inplanes, cfg[0:2], dropRate, stride))
        reg = cfg[1]
        self.inplanes = cfg[1] * block.expansion
        for i in range(1, blocks):
            layers.append(block(reg, cfg[2*i: 2*(i+1)], dropRate))
            reg = cfg[2*(i+1)-1]

        return nn.Sequential(*layers), cfg[-1]
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet20(dataset, prune=None, cfg=None):
    model = ResNet_Cifar(dataset=dataset, dropRate=0.0, block=BasicBlock, depth=20, cfg=cfg)
    return model

def resnet32(dataset, prune=None, cfg=None):
    model = ResNet_Cifar(dataset=dataset, dropRate=0.0, block=BasicBlock, depth=32, cfg=cfg)
    return model

def resnet56(dataset, prune=None, cfg=None):
    model = ResNet_Cifar(dataset=dataset, dropRate=0.0, block=BasicBlock, depth=56, cfg=cfg)
    return model

def resnet110(dataset, prune=None, cfg=None):
    model = ResNet_Cifar(dataset=dataset, dropRate=0.0, block=BasicBlock, depth=110, cfg=cfg)
    return model

if __name__=='__main__':
    model = resnet110(dataset='cifar10')
    print(model)
    x = torch.FloatTensor(1, 3, 32, 32)
    y = model(x)
    print(y.shape)
