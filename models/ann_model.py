import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, last_relu=False, downsample=None, stride2=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2 if stride2 else 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.last_relu = last_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x) if self.downsample is not None else x
        out += residual
        out = out[:, :, 1:-1, 1:-1].contiguous()
        return out if not self.last_relu else self.relu(out)


class ResNet2Stage(nn.Module):
    def __init__(self, firstchannels=64, channels=(64, 128), inchannel=3, block_num=(3, 4)):
        self.inplanes = firstchannels
        super(ResNet2Stage, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, firstchannels, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(firstchannels)
        self.relu = nn.ReLU(inplace=True)
        self.stage1 = self._make_layer(channels[0], block_num[0], last_relu=True, stride2=True)
        self.stage2 = self._make_layer(channels[1], block_num[1], last_relu=True, stride2=True)
        self.conv_out = nn.Conv2d(channels[1] * 4, channels[1] * 4, kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, planes, blocks, last_relu, stride2=False):
        block = Bottleneck
        downsample = None
        if self.inplanes != planes * block.expansion or stride2:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=3,
                          stride=2 if stride2 else 1, padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, last_relu=True, downsample=downsample, stride2=stride2)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, last_relu=(last_relu if i == (blocks-1) else True)))

        return nn.Sequential(*layers)

    def step(self, x):
        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)
        x = self.stage1(x)  
        x = self.stage2(x)  
        x = self.conv_out(x)
        return x

    def forward(self, net_in):
        return torch.stack([self.step(net_in[..., step]) for step in range(net_in.shape[-1])], -1)
