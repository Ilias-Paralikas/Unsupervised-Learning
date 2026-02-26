import torch
import torch.nn as nn
import torch.nn.functional as F

class WSConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=0, gain=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        # He's constant for Transpose Conv: sqrt(gain / fan_in)
        # fan_in here is in_channels * kernel_size^2
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights to N(0, 1)
        nn.init.normal_(self.conv.weight)

    def forward(self, x):
        scaled_weight = self.conv.weight * self.scale
        out = F.conv_transpose2d(
            x, 
            scaled_weight, 
            bias=None, 
            stride=self.conv.stride, 
            padding=self.conv.padding
        )
        return out + self.bias.view(1, self.bias.shape[0], 1, 1)