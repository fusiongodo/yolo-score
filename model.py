from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as c
from torchvision.models import resnet18
import loss as l
import random







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

         #Prevent further downsampling by setting strides to 1 in subsequent layers
        for layer in [backbone.layer2, backbone.layer3]:
            for block in layer:
                if hasattr(block, 'conv1'):
                    block.conv1.stride = (1, 1)
                if hasattr(block, 'downsample') and block.downsample is not None:
                    block.downsample[0].stride = (1, 1)

        #backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

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
        out = self.head(features)   #[Q, A*(5+C), N, N]
        N_batch, _, S1, S2 = out.shape 
        
        assert S1 == self.N and S2 == self.N, "Grid mismatch"
        # Reshape to [N_batch, N, N, A, 5 + C]
        out = out.view(N_batch, self.A, 5 + self.C, self.N, self.N).permute(0, 3, 4, 1, 2)
        return out
    
    def dummy_forward(self, batch_size: int, mode: str = 'random') -> torch.Tensor:
        """
        Generate dummy output tensor of shape [batch_size, N, N, A, 5 + C].
        
        Args:
            batch_size (int): Number of samples in the batch.
            mode (str): 'zero' for all zeros, 'one' for all ones, 'random' to choose randomly.
        
        Returns:
            torch.Tensor: Dummy output tensor.
        """
        shape = (batch_size, self.N, self.N, self.A, 5 + self.C)
        if mode == 'random':
            mode = random.choice(['zero', 'one'])
        if mode == 'one':
            return torch.ones(shape, dtype=torch.float32)
        return torch.zeros(shape, dtype=torch.float32)
