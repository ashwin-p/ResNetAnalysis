import torch
import torch.nn as nn
from models.components import Backbone, ResidualBlock, ClassificationHead, initialize_weights

class StandardResNet(nn.Module):
    def __init__(self, num_classes=100, num_extra_resblocks=0):
        super().__init__()
        self.backbone = Backbone()
        
        self.stage1 = ResidualBlock(64, 64)
        
        self.stage2 = nn.ModuleList()
        self.stage2.append(ResidualBlock(64, 128, stride=2))
        for _ in range(num_extra_resblocks):
            self.stage2.append(ResidualBlock(128, 128, stride=1))
        
        self.stage3 = nn.ModuleList()
        self.stage3.append(ResidualBlock(128, 256, stride=2))
        for _ in range(num_extra_resblocks):
            self.stage3.append(ResidualBlock(256, 256, stride=1))
        
        self.stage4 = ResidualBlock(256, 512, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = ClassificationHead(512, num_classes)
        
        self.apply(initialize_weights)

        for m in self.modules():
            if isinstance(m, ResidualBlock):
                if hasattr(m, "conv2"):
                    for n in m.preact2:
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

