import torch
import torch.nn as nn

from .L2_norm_conv import L2NormConv2d
from .conv_block import ConvBlock
class ResidualBlock(nn.Module):
    def __init__(self, 
                 channels, 
                 depth=2,
                 bias=True,
                 groups=8,
                 residual = True):
        
        super().__init__()
        self.residual = residual
        # 1. Safety check for GroupNorm
        # If out_channels is smaller than groups, reduce groups to match out_channels
        actual_groups = groups if channels % groups == 0 else channels
        self.blocks = nn.ModuleList()
        for i in range(depth-1):
            self.blocks.append(
                ConvBlock(
                    in_channels=channels, 
                    out_channels=channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=bias,
                    groups=actual_groups,
                    transpose=False
                )
            )
        self.blocks.append(nn.Conv2d(channels, channels, 3, 1, 1, bias=bias))
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.norm = nn.GroupNorm(num_groups=actual_groups, num_channels=channels)
    def forward(self, x):

        identity = x
        for block in self.blocks:
            x = block(x)
        
        if self.residual:
            x = x + identity

        x = self.norm(x)
        x = self.activation(x)
        return x