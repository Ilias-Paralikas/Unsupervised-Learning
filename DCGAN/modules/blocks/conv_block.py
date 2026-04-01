import torch
import torch.nn as nn

from .eq_lr_layers import EQLRConv2d
from .normalizations import PixelNorm



class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 stride=1, 
                 padding=1, 
                 bias=True,
                 use_norm  =True,
                 transpose=False):
        super().__init__()
        self.use_norm = use_norm
   
      
        self.conv = EQLRConv2d(
                in_channels, out_channels, kernel_size, 
                stride=stride, padding=padding, bias=bias
            )
        
        # 3. Normalization and Activation
        if self.use_norm:
            self.norm = PixelNorm()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        return self.activation(x)