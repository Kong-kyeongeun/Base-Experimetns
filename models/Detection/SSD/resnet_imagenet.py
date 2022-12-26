import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from thop import profile

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, cfg, dropRate, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg[0], stride)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(cfg[0], cfg[1])
        self.bn2 = nn.BatchNorm2d(cfg[1])

        self.downsample = None
        self.stride = stride
        self.dropRate = dropRate

        if stride != 1 or inplanes != cfg[1] * BasicBlock.expansion:
            self.convShortcut = nn.Conv2d(inplanes, cfg[1] * BasicBlock.expansion,
                                    kernel_size=1, stride=stride, bias=False)
            self.bnShortcut = nn.BatchNorm2d(cfg[1] * BasicBlock.expansion)
            self.downsample = True

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
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
        out = self.relu(out)

        return out

def fixed_expansion(n_block):
    if n_block is 1:
        o_ch = 64
    elif n_block is 2:
        o_ch = 128
    elif n_block is 3:
        o_ch = 256
    elif n_block is 4:
        o_ch = 512
    else:
        raise ValueError("Out of range for the resnet channels")
    return o_ch


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, cfg, n_block, dropRate, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv3 = nn.Conv2d(cfg[1], fixed_expansion(n_block) * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(fixed_expansion(n_block) * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride
        self.dropRate = dropRate

        if stride != 1 or inplanes != fixed_expansion(n_block) * Bottleneck.expansion:
            self.convShortcut = nn.Conv2d(inplanes, fixed_expansion(n_block) * Bottleneck.expansion,
                                    kernel_size=1, stride=stride, bias=False)
            self.bnShortcut = nn.BatchNorm2d(fixed_expansion(n_block) * Bottleneck.expansion)
            self.downsample = True

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        #if self.dropRate > 0:
        #    out = F.dropout(out, p=self.dropRate, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.convShortcut(residual)
            residual = self.bnShortcut(residual)

        out += residual
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, dropRate, cfg=None, num_classes=1000):
        self.inplanes = 64
        self.dropRate = dropRate

        super(ResNet, self).__init__()
        if cfg == None:
            #cfg = [[64]*layers[0], [128]*layers[1], [256]*layers[2], [512]*layers[3]]
            cfg = [[64], [64, 64]*layers[0], [128, 128]*layers[1], [256, 256]*layers[2], [512, 512]*layers[3]]
            cfg = [item for sub_list in cfg for item in sub_list]
        print(cfg)
        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        if block.__name__ == 'BasicBlock':
            ## for only resnet18
            
            self.layer1, self.inplanes = self._make_layer_1(block, layers[0], cfg[1:2*layers[0]+1], self.dropRate)
            self.layer2, self.inplanes = self._make_layer_1(block, layers[1], cfg[2*layers[0]+1:4*layers[1]+1], self.dropRate, stride=2)
            self.layer3, self.inplanes = self._make_layer_1(block, layers[2], cfg[4*layers[1]+1:6*layers[2]+1], self.dropRate, stride=2)
            self.layer4, self.inplanes = self._make_layer_1(block, layers[3], cfg[6*layers[2]+1:8*layers[3]+1], self.dropRate, stride=2)
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(cfg[-1] * block.expansion, num_classes) # resnet18
            
            ## for only resnet34
            '''
            self.layer1, self.inplanes = self._make_layer_1(
                    block, layers[0],
                    cfg[1:2*layers[0]+1], self.dropRate)
            self.layer2, self.inplanes = self._make_layer_1(
                    block, layers[1],
                    cfg[2*layers[0]+1 : 2*(layers[0]+layers[1])+1], self.dropRate, stride=2)
            self.layer3, self.inplanes = self._make_layer_1(
                    block, layers[2],
                    cfg[2*(layers[0]+layers[1])+1 : 2*(layers[0]+layers[1]+layers[2])+1],
                    self.dropRate, stride=2)
            self.layer4, self.inplanes = self._make_layer_1(
                block, layers[3],
                cfg[2*(layers[0]+layers[1]+layers[2])+1 : 2*(layers[0]+layers[1]+layers[2]+layers[3])+1],
                self.dropRate, stride=2)
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(cfg[-1] * block.expansion, num_classes) # resnet18
            '''
        if block.__name__ == 'Bottleneck':
            ## for only resnet50
            self.layer1, self.inplanes = self._make_layer_2(
                block, layers[0],
                cfg[1:2*layers[0]+1], 1, self.dropRate)
            self.layer2, self.inplanes = self._make_layer_2(
                block, layers[1],
                cfg[2*layers[0]+1 : 2*(layers[0]+layers[1])+1], 2, self.dropRate, stride=2)
            self.layer3, self.inplanes = self._make_layer_2(
                block, layers[2],
                cfg[2*(layers[0]+layers[1])+1 : 2*(layers[0]+layers[1]+layers[2])+1],
                3, self.dropRate, stride=2)
            self.layer4, self.inplanes = self._make_layer_2(
                block, layers[3],
                cfg[2*(layers[0]+layers[1]+layers[2])+1 : 2*(layers[0]+layers[1]+layers[2]+layers[3])+1],
                4, self.dropRate, stride=2)
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(fixed_expansion(4) * block.expansion, num_classes) # resnet50

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_1(self, block, blocks, cfg, dropRate, stride=1):
        downsample = None
        reg = 0
        layers = []
        layers.append(block(self.inplanes, cfg[0:2], dropRate, stride))
        reg = cfg[1]
        self.inplanes = int(cfg[1] * block.expansion)
        for i in range(1, blocks):
            layers.append(block(reg, cfg[2*i: 2*(i+1)], dropRate))
            reg = cfg[2*(i+1)-1]
        return nn.Sequential(*layers), cfg[-1]

    def _make_layer_2(self, block, blocks, cfg, n_block, dropRate, stride=1):
        downsample = None
        reg = 0
        layers = []
        layers.append(block(self.inplanes, cfg[0:2], n_block, dropRate, stride))
        reg = fixed_expansion(n_block) * block.expansion
        for i in range(1, blocks):
            layers.append(block(reg, cfg[2*i: 2*(i+1)], n_block, dropRate))
            reg = fixed_expansion(n_block) * block.expansion
        return nn.Sequential(*layers), fixed_expansion(n_block) * block.expansion

    def forward(self, x):
         x = self.conv1(x)
         x = self.bn1(x)
         x = self.relu(x)
         x = self.maxpool(x) 
         
         x = self.layer1(x)
         x = self.layer2(x)
         x = self.layer3(x)
         x = self.layer4(x)

         x = self.avgpool(x)
         x = x.view(x.size(0), -1)
         x = self.fc(x)

         return x


def resnet18(pretrained=False, dropRate=0.0, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], dropRate, **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        print('ResNet-18 Use pretrained model for initialization')
    return model

def resnet34(pretrained=False, dropRate=0.0, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], dropRate, **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        print('ResNet-34 Use pretrained model for initialization')
    return model

def resnet50(pretrained=False, dropRate=0.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], dropRate, **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        print('ResNet-50 Use pretrained model for initialization')
    return model

if __name__ == '__main__':
    model = resnet18(pretrained=False, dropRate=0.1)
    print(model)
    x = torch.FloatTensor(1, 3, 224, 224)
    #flops, params = profile(model, inputs=(input,))
    #print(flops, params)
    y = model(x)
    print(y.shape)
