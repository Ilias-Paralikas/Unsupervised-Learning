import torch
import torch.nn as nn
import torch.nn.functional as F



from .modulate_conv_2d import ModulatedConv2d
from .noise_injection import NoiseInjection

class StyleConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                out_channels,
                kernel_size, 
                w_dim, 
                demodulate=True):
        super().__init__()
        self.w_conv = ModulatedConv2d(in_channels, 
                out_channels,
                kernel_size, 
                w_dim, 
                demodulate=demodulate)
        
        self.noise_injection = NoiseInjection(out_channels)

        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
    def forward(self, x, w):
        x = self.w_conv(x,w)
        x = self.noise_injection(x)
        x = x + self.bias
        x = self.activate(x)
        return x 