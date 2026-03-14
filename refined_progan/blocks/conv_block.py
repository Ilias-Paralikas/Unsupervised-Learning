import torch
import torch.nn as nn

from .convs import CustomConv2d
from .convs import CustomConvTranspose2d
from .normalizations import PixelNorm



class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 stride=1, 
                 padding=1, 
                 bias=True,
                 norm  =True,
                 transpose=False):
        super().__init__()
        self.norm = norm
        # 1. Safety check for GroupNorm
        # If out_channels is smaller than groups, reduce groups to match out_channels

        
        # 2. Define the convolution layer with explicit keywords
        if transpose:
            self.conv = CustomConvTranspose2d(
                in_channels, out_channels, kernel_size, 
                stride=stride, padding=padding, bias=bias
            )
        else:
            self.conv = CustomConv2d(
                in_channels, out_channels, kernel_size, 
                stride=stride, padding=padding, bias=bias
            )
        
        # 3. Normalization and Activation
        if self.norm:
            self.norm = PixelNorm()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        return self.activation(x)