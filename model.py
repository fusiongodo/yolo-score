from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as c



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

            #896x896x3
            _ConvBlock(3, 16, 7, 1, 3),
            nn.MaxPool2d(2, 2),
            #448x448x16
            _ConvBlock(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            #224x224x32
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
    





class YOLOv2Tiny(nn.Module):
    """YOLO v1 backbone + detection head (fully‑connected)."""
    def __init__(self): #SxS Anzahl Grid cells, B Anzahl Anchors
        super().__init__()
        self.S, self.A, self.C = c.S, c.A, c.C

        self.model = nn.Sequential(
            #896x896x3
            _ConvBlock(3, 8, 7, 1, 3),
            nn.MaxPool2d(2, 2),
            #448x448x16
            _ConvBlock(8, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            #224x224x32
            _ConvBlock(16, 24, 1, 1, 0),
            #224x224x24
            _ConvBlock(24, 48, 3, 1, 1),
            #224x224x48
            _ConvBlock(48, 48, 1, 1, 0),
            #224x224x48
            _ConvBlock(48, 96, 3, 1, 1),
            #224x224x96
            nn.MaxPool2d(2, 2),
            #112x112x96

            # usually four times
            *[nn.Sequential(
                _ConvBlock(96, 48, 1, 1, 0),
                _ConvBlock(48, 96, 3, 1, 1)
            ) for _ in range(1)],

            #112x112x96

            _ConvBlock(96, 96, 1, 1, 0),
            _ConvBlock(96, 128, 3, 1, 1),
            #112x112x128

            _ConvBlock(128, 128, 3, 1, 1),
            _ConvBlock(128,  self.A * (5 + self.C), 1, 1, 0 ) #6 anchors
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        N, _, S1, S2 = out.shape
        assert S1 == self.S and S2 == self.S, "Grid mismatch"
        # reshape to [N, S, S, B*5 + C]
        out = out.view(N, self.A, 5 + self.C, self.S, self.S) \
                 .permute(0, 3, 4, 1, 2)
        return out

