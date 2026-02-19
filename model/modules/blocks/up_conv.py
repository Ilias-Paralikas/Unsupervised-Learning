import torch.nn as nn
import torch.nn.functional as F

from .conv_block import ConvBlock
from .residual_double_conv import ResidualDoubleConv


class UpConv(nn.Module):
    def __init__(self, in_channels, 
                 out_channels,
                 norm,
                 activation,
                 scale=2,
                 mode='bilinear',
                 double_conv=True):
        super().__init__()
        self.scale = scale
        self.mode  = mode
        self.channel_conv =  ConvBlock(in_channels, 
                                    out_channels, 
                                    kernel_size=3,
                                    padding=1,
                                    norm=norm,
                                    activation=activation)
        if double_conv:
            self.double_conv = ResidualDoubleConv(out_channels,norm=norm,activation=activation)
        else:
            self.double_conv = None
       
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        x = self.channel_conv(x)

        if self.double_conv is not None:
            x = self.double_conv(x) 

        return x
    