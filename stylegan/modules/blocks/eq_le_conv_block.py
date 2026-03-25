import torch 
import torch.nn as nn
import torch.nn.functional as F

from .eq_lr_conv2d import EQLRConv2d
from .pixel_norm import PixelNorm

class EQLEConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size=3, 
                 stride=1, 
                 padding=1,
                 bias=True,
                 use_norm=False):
        super().__init__()
        self.conv =  EQLRConv2d(in_channels=in_channels, 
                                   out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=bias)
        self.lrelu = nn.LeakyReLU(0.2)
        self.use_norm = use_norm
        if use_norm:
            self.norm = PixelNorm()
    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.lrelu(x)
        return x
