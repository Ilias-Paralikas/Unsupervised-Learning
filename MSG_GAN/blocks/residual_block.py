import torch
import torch.nn as nn

from .convs import CustomConv2d
from .conv_block import ConvBlock
from .normalizations import PixelNorm

class ResidualBlock(nn.Module):
    def __init__(self, 
                 channels, 
                 depth=2,
                 bias=True,
                 use_norm=True,
                 residual = True):
        
        super().__init__()
        self.residual = residual
        self.use_norm = use_norm
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
                    use_norm=self.use_norm,
                    transpose=False
                )
            )
        self.blocks.append(CustomConv2d(channels, channels, 3, 1, 1, bias=bias))
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        if self.use_norm:
            self.norm = PixelNorm()
    def forward(self, x):

        identity = x
        for block in self.blocks:
            x = block(x)
   
        if self.use_norm:
            x = self.norm(x)

             
        if self.residual:
            x = x + identity

        x = self.activation(x)
        return x