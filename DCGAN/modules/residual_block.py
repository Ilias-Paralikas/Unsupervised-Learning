import torch
import torch.nn as nn

from .blocks.eq_lr_layers import EQLRConv2d
from .blocks import ConvBlock
from .blocks.normalizations import PixelNorm

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 depth=2,
                 bias=True,
                 use_norm=True,
                 residual = True):
        
        super().__init__()
        self.residual = residual
        self.use_norm = use_norm
        if in_channels != out_channels:
            self.channel_adaptor = EQLRConv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        else:
            self.channel_adaptor = nn.Identity()

        
        self.blocks = nn.ModuleList()
        for i in range(depth-1):
            self.blocks.append(
                ConvBlock(
                    in_channels=in_channels, 
                    out_channels=in_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=bias,
                    use_norm=self.use_norm
                )
            )
        self.blocks.append(EQLRConv2d(in_channels, in_channels, 3, 1, 1, bias=bias))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.norm = PixelNorm() if use_norm else nn.Identity()

    def forward(self, x):

        identity = x
        for block in self.blocks:
            x = block(x)

        if self.residual:
            x = x + identity

        x = self.norm(x)

        x = self.activation(x)

        x = self.channel_adaptor(x)
        return x