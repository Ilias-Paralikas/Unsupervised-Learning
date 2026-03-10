import torch
import torch.nn as nn

from .convs import CustomConv2d
from .conv_block import ConvBlock
from .helpers import GroupNormOrNone

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
                    groups=groups,
                    transpose=False
                )
            )
        self.blocks.append(CustomConv2d(channels, channels, 3, 1, 1, bias=bias))
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.norm = GroupNormOrNone(channels, groups)
    def forward(self, x):

        identity = x
        for block in self.blocks:
            x = block(x)
        
        if self.residual:
            x = x + identity

        x = self.norm(x)
        x = self.activation(x)
        return x