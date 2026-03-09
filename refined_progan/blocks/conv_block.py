import torch
import torch.nn as nn

from .L2_norm_conv import L2NormConv2d
from .L2_norm_trans_conv import L2NormConvTranspose2d


# Assuming these are in the same directory or correctly imported
# from .L2_norm_conv import L2NormConv2d
# from .L2_norm_trans_conv import L2NormConvTranspose2d

class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 bias=True,
                 groups=8,
                 transpose=False):
        super().__init__()
        
        # 1. Safety check for GroupNorm
        # If out_channels is smaller than groups, reduce groups to match out_channels
        actual_groups = groups if out_channels % groups == 0 else out_channels
        
        # 2. Define the convolution layer with explicit keywords
        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, 
                stride=stride, padding=padding, bias=bias
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, 
                stride=stride, padding=padding, bias=bias
            )
        
        # 3. Normalization and Activation
        self.norm = nn.GroupNorm(num_groups=actual_groups, num_channels=out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)