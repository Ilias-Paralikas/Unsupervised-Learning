# --- DCGAN/modules/residualblock.py ---
import torch
import torch.nn as nn

from .blocks.eq_lr_layers import EQLRConv2d
from .blocks.normalizations import PixelNorm

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 depth=2, 
                 bias=True, 
                 use_norm=True, 
                 residual=True):
        super().__init__()
        self.residual = residual
        self.use_norm = use_norm
        
        if in_channels != out_channels:
            self.channel_adaptor = EQLRConv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        else:
            self.channel_adaptor = nn.Identity()
            
        self.blocks = nn.ModuleList()
        
        for i in range(depth ):
            c_in = in_channels if i == 0 else out_channels

            # Using EQLRConv2d directly instead of ConvBlock so we can cleanly insert SPADE
            self.blocks.append(EQLRConv2d(c_in, out_channels, kernel_size=3, stride=1, padding=1, bias=bias))
        
        if self.use_norm:
            self.norm = PixelNorm()
        else:
            self.norm = nn.Identity()
                
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.residual_scale = 1/ (2**0.5)

    def forward(self, x):
        identity = x
        for i, block in enumerate(self.blocks):
            # 1. Convolution
            x = block(x)
            
            # 2. Normalization
            x = self.norm(x)
                
            # 3. Activation
            if i != len(self.blocks) - 1:
                x = self.activation(x)
        if self.residual:
            x = self.residual_scale * (x + self.channel_adaptor(identity))
        x = self.activation(x)

        return x