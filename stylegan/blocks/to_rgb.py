import torch
import torch.nn as nn

from .modulate_conv_2d import ModulatedConv2d
class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        # Demodulate is False for ToRGB!
        self.w_conv = ModulatedConv2d(in_channels, out_channels, kernel_size=1, w_dim=w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
    def forward(self, x, w):
        x = self.w_conv(x, w)
        return x + self.bias
