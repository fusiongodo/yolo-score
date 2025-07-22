from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as c
from torchvision.models import resnet18



class _ConvBlock(nn.Module):
    """Conv‑BN‑LeakyReLU block used throughout Darknet‑like backbone."""
    def __init__(self, c_in: int, c_out: int, k: int, s: int, p: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))




class YOLOv2Heavy(nn.Module):
    """YOLO v1 backbone + detection head (fully‑connected)."""
    def __init__(self): #SxS Anzahl Grid cells, B Anzahl Anchors
        super().__init__()
        self.S, self.A, self.C = c.S, c.A, c.C

        self.model = nn.Sequential(

            #240x240x1 (config.RES = 240)
            _ConvBlock(1, 8, 7, 1, 3),
            nn.MaxPool2d(2, 2),
            #120x120x8
            _ConvBlock(8, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            #60x60x16
            _ConvBlock(32, 48, 1, 1, 0),
            #224x224x24
            _ConvBlock(48, 64, 3, 1, 1),
            #224x224x48
            _ConvBlock(64, 128, 1, 1, 0),
            #224x224x48
            nn.MaxPool2d(2, 2),
            _ConvBlock(128, 256, 3, 1, 1),
            #112x112x256
            _ConvBlock(256, 512, 3, 1, 1),
            
            #112x112x96

            # usually four times
            *[nn.Sequential(
                _ConvBlock(512, 256, 1, 1, 0),
                _ConvBlock(256, 512, 3, 1, 1)
            ) for _ in range(4)],

            #112x112x96

            _ConvBlock(512, 512, 1, 1, 0),
            _ConvBlock(512, 1024, 3, 1, 1),
            #112x112x128

            _ConvBlock(1024, 1024, 3, 1, 1),
            _ConvBlock(1024,  self.A * (5 + self.C), 1, 1, 0 ) #6 anchors
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        N, _, S1, S2 = out.shape
        assert S1 == self.S and S2 == self.S, "Grid mismatch"
        # reshape to [N, S, S, B*5 + C]
        out = out.view(N, self.A, 5 + self.C, self.S, self.S) \
                 .permute(0, 3, 4, 1, 2)
        return out
    







class YOLOResNet(nn.Module):
    """ResNet18 backbone adapted for YOLO detection head."""
    def __init__(self):
        super().__init__()
        self.N = c.N  # 40, grid size for crop
        self.A = c.A  # 2, number of anchors
        self.C = c.C  # 135, number of classes

        backbone = resnet18(weights=None)
        # Adapt for grayscale input and adjust initial stride for desired downsampling
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=3, padding=3, bias=False)

        # Prevent further downsampling by setting strides to 1 in subsequent layers
        for layer in [backbone.layer2, backbone.layer3, backbone.layer4]:
            for block in layer:
                if hasattr(block, 'conv1'):
                    block.conv1.stride = (1, 1)
                if hasattr(block, 'downsample') and block.downsample is not None:
                    block.downsample[0].stride = (1, 1)

        # Backbone up to the feature map
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # YOLO detection head: conv to desired output channels
        num_features = 512  # Output channels from ResNet18 layer4
        self.head = nn.Conv2d(num_features, self.A * (5 + self.C), kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        out = self.head(features)
        N_batch, _, S1, S2 = out.shape
        assert S1 == self.N and S2 == self.N, "Grid mismatch"
        # Reshape to [N_batch, N, N, A, 5 + C]
        out = out.view(N_batch, self.A, 5 + self.C, self.N, self.N).permute(0, 3, 4, 1, 2)
        return out