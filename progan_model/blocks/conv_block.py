import torch
import torch.nn as nn

from .wsconv import WSConv2d
from .pixel_norm import PixelNorm
class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 use_pixel_norm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2,inplace=True)
        self.pixel_norm = PixelNorm()
        self.use_pixel_norm = use_pixel_norm
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        if self.use_pixel_norm:
            x = self.pixel_norm(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        if self.use_pixel_norm:
            x = self.pixel_norm(x)
        return x
       
