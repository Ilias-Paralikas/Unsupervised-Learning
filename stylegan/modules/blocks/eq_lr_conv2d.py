import torch
import torch.nn as nn
import torch.nn.functional as F

class EQLRConv2d(nn.Module):
    def __init__(self,
                in_channels, 
                out_channels, 
                kernel_size,
                stride=1, 
                padding=0, 
                bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding

        self.fan_in = in_channels * kernel_size ** 2
        self.scale = (2 / self.fan_in) ** 0.5
        
    def forward(self, x):
        weight = self.weight * self.scale
        return F.conv2d(x, weight, self.bias, self.stride, self.padding)
    
