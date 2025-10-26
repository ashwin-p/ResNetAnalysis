import torch
import torch.nn as nn
from models.components import Backbone, BottleneckBlock, ClassificationHead, initialize_weights

class BottleneckResNet(nn.Module):
    def __init__(self, num_classes=100, num_extra_blocks=0, bottleneck_ratio=4):
        super().__init__()
        self.backbone = Backbone()

        self.stage1 = BottleneckBlock(64, 64, stride=1, bottleneck_ratio=bottleneck_ratio)

        self.stage2 = nn.ModuleList()
        self.stage2.append(BottleneckBlock(64, 128, stride=2, bottleneck_ratio=bottleneck_ratio))
        for _ in range(num_extra_blocks):
            self.stage2.append(BottleneckBlock(128, 128, stride=1, bottleneck_ratio=bottleneck_ratio))

        self.stage3 = nn.ModuleList()
        self.stage3.append(BottleneckBlock(128, 256, stride=2, bottleneck_ratio=bottleneck_ratio))
        for _ in range(num_extra_blocks):
            self.stage3.append(BottleneckBlock(256, 256, stride=1, bottleneck_ratio=bottleneck_ratio))

        self.stage4 = BottleneckBlock(256, 512, stride=2, bottleneck_ratio=bottleneck_ratio)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = ClassificationHead(512, num_classes)

        self.apply(initialize_weights)

        for m in self.modules():
            if isinstance(m, BottleneckBlock):
                if hasattr(m, "preact3"):
                    for n in m.preact3:
                        if isinstance(n, nn.GroupNorm):
                            nn.init.constant_(n.weight, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.stage1(x)
        for block in self.stage2:
            x = block(x)
        for block in self.stage3:
            x = block(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

