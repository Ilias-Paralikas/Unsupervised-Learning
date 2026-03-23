import torch
import torch.nn as nn

from .modulated_conv import ModulatedConv2d

class ToRGB(nn.Module):
    def __init__(self, in_channels, w_dim,img_channels=3):
        super().__init__()
        # 1x1 modulated convolution, NO demodulation
        self.conv = ModulatedConv2d(
            in_channels=in_channels, 
            out_channels=img_channels,
            kernel_size=1, 
            w_dim=w_dim, 
            demodulate=False
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, w):
        x = self.conv(x, w)
        x = x + self.bias
        return x