import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn'
]

model_urls = {
    'vgg11' : 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13' : 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, batch_norm=False, init_weights=True, cfg=None, num_classes=1000):
        super(VGG, self).__init__()
        print(cfg)
        self.features = self.make_layers(cfg, batch_norm)
        self.classifier = nn.Sequential(
                nn.Linear(cfg[-2] * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers=[]
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def vgg11(pretrained=False, **kwargs):
    model = VGG(cfg=cfg['A'], **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model

def vgg11_bn(pretrained=False, **kwargs):
    model = VGG(batch_norm=True, cfg=cfg['A'], **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model

def vgg13(pretrained=False, **kwargs):
    model = VGG(cfg=cfg['B'], **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model

def vgg13_bn(pretrained=False, **kwargs):
    model = VGG(batch_norm=True, cfg=cfg['B'], **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model

def vgg16(pretrained=False, **kwargs):
    model = VGG(cfg=cfg['D'], **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

def vgg16_bn(pretrained=False, **kwargs):
    model = VGG(batch_norm=True, cfg=cfg['D'], **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model

def vgg19(pretrained=False, **kwargs):
    model = VGG(cfg=cfg['E'], **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

def vgg19_bn(pretrained=False, **kwargs):
    model = VGG(batch_norm=True, cfg=cfg['E'], **kwargs)
    if pretrained == True:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


if __name__ == '__main__':
    model = vgg19_bn(pretrained=True)
    x = Variable(torch.FloatTensor(1, 3, 224, 224))
    y = model(x)
    print(y.data.shape)
    #for index, item in enumerate(model.parameters()):
    #    print(item.data)
