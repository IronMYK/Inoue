import torch.nn as nn
from torchvision.models import resnet50

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        res_net_50 =  resnet50(pretrained=True)
        pretrained = nn.Sequential(*list(res_net_50.children())[:6])
        for p in pretrained.parameters():
            p.requires_grad = False
        self.model = nn.Sequential(
            pretrained,
            nn.Dropout2d(),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvLayer(512, 512),
            ConvLayer(512, 256),
            ConvLayer(256, 128),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvLayer(128, 128),
            ConvLayer(128, 64),
            ConvLayer(64, 32),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvLayer(32, 32),
            ConvLayer(32, 16),
            ConvLayer(16, 1, relu=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Restorer(nn.Module):
    def __init__(self):
        super(Restorer, self).__init__()
        self.model = nn.Sequential(
            ConvLayer(1, 16, 5, 2),
            ConvLayer(16, 32, 3, 2),
            ConvLayer(32, 64, 3),
            ConvLayer(64, 128, 3, 2),
            ConvLayer(128, 256),
            ConvLayer(256, 256),
            ConvLayer(256, 128),
            ConvLayer(128, 64),
            nn.Dropout2d(),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvLayer(64, 64),
            ConvLayer(64, 32),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvLayer(32, 16),
            ConvLayer(16, 16),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvLayer(16, 16),
            ConvLayer(16, 8),
            ConvLayer(8, 1, relu=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, pad=1, relu=True):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=pad)
        self.conv2d_bn = nn.BatchNorm2d(out_channels)
        self.relu = relu

    def forward(self, x):
        out = self.conv2d(x)
        out = self.conv2d_bn(out)
        if self.relu:
            out = nn.functional.relu(out)
        return out