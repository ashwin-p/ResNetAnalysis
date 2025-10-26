import torch.nn as nn

import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        return self.stem(x)


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(in_features // 2, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class ShortcutProjection(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.GroupNorm(16, out_channels),
        )

    def forward(self, x):
        return self.projection(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, negative_slope=0.01):
        super().__init__()
        self.preact1 = nn.Sequential(
            nn.GroupNorm(16, in_channels),
            nn.LeakyReLU(negative_slope)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)

        self.preact2 = nn.Sequential(
            nn.GroupNorm(16, out_channels),
            nn.LeakyReLU(negative_slope)
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(16, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.preact1(x)
        out = self.conv1(out)
        out = self.preact2(out)
        out = self.conv2(out)
        return out + shortcut

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bottleneck_ratio=4, negative_slope=0.01):
        super().__init__()
        mid_channels = out_channels // bottleneck_ratio

        self.preact1 = nn.Sequential(
            nn.GroupNorm(16, in_channels),
            nn.LeakyReLU(negative_slope)
        )
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

        self.preact2 = nn.Sequential(
            nn.GroupNorm(16, mid_channels),
            nn.LeakyReLU(negative_slope)
        )
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.preact3 = nn.Sequential(
            nn.GroupNorm(16, mid_channels),
            nn.LeakyReLU(negative_slope)
        )
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(16, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.preact1(x)
        out = self.conv1(out)
        out = self.preact2(out)
        out = self.conv2(out)
        out = self.preact3(out)
        out = self.conv3(out)
        return out + shortcut

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu")
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

