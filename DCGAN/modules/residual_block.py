# --- DCGAN/modules/residualblock.py ---
import torch
import torch.nn as nn
from .blocks.eq_lr_layers import EQLRConv2d
from .blocks import ConvBlock
from .blocks.normalizations import PixelNorm
from .blocks.normalizations.spade import SPADE # Add this import

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, bias=True, use_norm=True, residual=True, use_spade=False, label_nc=None):
        super().__init__()
        self.residual = residual
        self.use_norm = use_norm
        self.use_spade = use_spade
        
        if in_channels != out_channels:
            # We must set bias=False if using SPADE
            self.channel_adaptor = EQLRConv2d(in_channels, out_channels, 1, 1, 0, bias=(not use_spade))
        else:
            self.channel_adaptor = nn.Identity()
            
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList() # Store norms separately so we can pass routed_styles
        
        for i in range(depth ):
            c_in = in_channels if i == 0 else out_channels
            
            # Using EQLRConv2d directly instead of ConvBlock so we can cleanly insert SPADE
            self.blocks.append(EQLRConv2d(c_in, out_channels, kernel_size=3, stride=1, padding=1, bias=(not use_spade)))
            
            if self.use_spade:
                self.norms.append(SPADE(out_channels, label_nc))
            elif self.use_norm:
                self.norms.append(PixelNorm())
            else:
                self.norms.append(nn.Identity())
                
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, routed_styles=None):
        identity = x
        for i, block in enumerate(self.blocks):
            # 1. Convolution
            x = block(x)
            
            # 2. Normalization / SPADE
            if self.use_spade:
                x = self.norms[i](x, routed_styles)
            else:
                x = self.norms[i](x)
                
            # 3. Activation
            x = self.activation(x)
            
        if self.residual:
            x = x + self.channel_adaptor(identity)
        return x