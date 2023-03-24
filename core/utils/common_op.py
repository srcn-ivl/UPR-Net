import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def conv_bn_noact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
    )


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def conv2(in_planes, out_planes, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride, 1),
        nn.PReLU(out_planes),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.PReLU(out_planes)
    )


def conv3(in_planes, out_planes, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride, 1),
        nn.PReLU(out_planes),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.PReLU(out_planes),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.PReLU(out_planes)
    )


def conv4(in_planes, out_planes, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride, 1),
        nn.PReLU(out_planes),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.PReLU(out_planes),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.PReLU(out_planes),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes))


def deconv2(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.PReLU(out_planes)
    )


def deconv3(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.PReLU(out_planes),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.PReLU(out_planes)
    )



class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, has_bn=False):
        super(ResBlock, self).__init__()

        if in_planes == out_planes and stride == 1:
            self.conv0 = nn.Identity()
        else:
            self.conv0 = nn.Conv2d(in_planes, out_planes,
                                   3, stride, 1, bias=False)

        if has_bn:
            self.conv1 = conv_bn_act(in_planes, out_planes, 3, stride, 1)
            self.conv2 = conv_bn(out_planes, out_planes, 3, 1, 1)
        else:
            self.conv1 = conv_act(in_planes, out_planes, 3, stride, 1)
            self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

        self.relu1 = nn.PReLU(1)
        self.relu2 = nn.PReLU(out_planes)

        self.fc1 = nn.Conv2d(out_planes, 16, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(16, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        w = x.mean(3, True).mean(2, True)
        w = self.relu1(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        x = self.relu2(x * w + y)
        return x
