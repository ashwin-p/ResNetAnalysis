import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.stem(x)


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
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
            nn.GroupNorm(32, out_channels),
        )

    def forward(self, x):
        return self.projection(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
        )
        self.act = nn.ReLU()

        if in_channels != out_channels or stride != 1:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act(self.block(x) + shortcut)
        return x

