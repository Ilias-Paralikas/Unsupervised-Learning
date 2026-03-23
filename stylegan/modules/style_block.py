import torch 
import torch.nn as nn

from .modulated_conv import ModulatedConv2d
from .blocks import NoiseInjection

class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels,w_dim,kernel_size=3):
        super().__init__()
        self.conv =ModulatedConv2d(in_channels=in_channels, 
                                   out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     w_dim=w_dim)
        self.noise_injection = NoiseInjection(channels=out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x,w,noise=None):
        x = self.conv(x,w)
        x = self.noise_injection(x,noise)
        x = x + self.bias
        return self.lrelu(x)