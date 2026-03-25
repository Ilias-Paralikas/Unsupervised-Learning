import torch 
import torch.nn as nn
import torch.nn.functional as F

from .blocks import EQLRConv2d, EQLEConvBlock
from .blocks import PixelNorm


class EQLRDeepConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 depth=2,
                 kernel_size=3, 
                 stride=1, 
                 padding=1,
                 bias=True,
                 residual=True,
                 use_norm=False):
        super().__init__()
        self.residual = residual
        self.blocks = nn.ModuleList([
                EQLEConvBlock(in_channels=in_channels, 
                                   out_channels=in_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=bias,
                                     use_norm=use_norm)
                for _ in range(depth-1)
            ])
        self.final_block = EQLRConv2d(in_channels=in_channels, 
                                   out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=bias)
        
        self.lrelu = nn.LeakyReLU(0.2)

        self.residual_scale = 1/ (2**0.5)

        self.use_norm = use_norm
        if use_norm:
            self.norm = PixelNorm(out_channels)

    def forward(self, x):
        identity = x
        for block in self.blocks:
            x = block(x)
        if self.residual:
            x = self.residual_scale *(x +identity)
        x = self.final_block(x) 
        
        if self.use_norm:
            x = self.norm(x)
  
        x = self.lrelu(x)
        return x