import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Double_conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, leaky=0, bias=True):
        super(Double_conv, self).__init__()
        self.layer = []
        self.layer.append(torch.nn.Conv2d(in_ch, out_ch, 3, 1, padding=1, bias=bias))
        self.layer.append(torch.nn.BatchNorm2d(out_ch))
        self.layer.append(torch.nn.ReLU(inplace=True) if not leaky else torch.nn.LeakyReLU(leaky, inplace=True))
        self.layer.append(torch.nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=bias))
        self.layer.append(torch.nn.BatchNorm2d(out_ch))
        self.layer.append(torch.nn.ReLU(inplace=True) if not leaky else torch.nn.LeakyReLU(leaky, inplace=True))
        self.layer = torch.nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResUNet34_2task(torch.nn.Module):

    # 编码器使用resnet的原始模块
    def __init__(self, in_ch, num_classes, bilinear=False, layer_return=False, bias=False):
        super(ResUNet34_2task, self).__init__()
        self.layer_return = layer_return
        self.bias = bias
        self.base_channel = 16
        block = BasicBlock
        self.filters = [self.base_channel, self.base_channel * 2, self.base_channel * 4, self.base_channel * 8,
                        self.base_channel * 16]
        self.firstconv = torch.nn.Sequential(*[torch.nn.Conv2d(in_ch, self.base_channel, 3, padding=1),
                                               torch.nn.BatchNorm2d(self.base_channel),
                                               torch.nn.ReLU(inplace=True)])
        self.enc1 = self._make_layer(block, self.filters[0], self.filters[0], 3, 1)
        self.enc2 = self._make_layer(block, self.filters[0], self.filters[1], 4, 2)
        self.enc3 = self._make_layer(block, self.filters[1], self.filters[2], 6, 2)
        self.enc4 = self._make_layer(block, self.filters[2], self.filters[3], 3, 2)

        #self.centerblock = Double_conv(self.filters[3], self.filters[3], bias=bias)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.filters[3], num_classes)


    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        # encoder
        out_first = self.firstconv(x)
        out_enc1 = self.enc1(out_first)
        out_enc2 = self.enc2(out_enc1)
        out_enc3 = self.enc3(out_enc2)
        out_enc4 = self.enc4(out_enc3)
       
        #center = self.centerblock(out_enc4)
       
        pool = self.avgpool(out_enc4)
        fla = torch.flatten(pool, 1)
        fc = self.fc(fla)

        return fc



