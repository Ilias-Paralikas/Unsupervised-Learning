import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EQLRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1):
        super().__init__()
        # if isinstance(kernel_size, int):
        #     kernel_size = (kernel_size, kernel_size)
            
        # self.stride = stride
        # self.padding = padding
        # self.groups = groups
        
        # # 1. For grouped convolutions, the weight shape is (out_channels, in_channels // groups, kH, kW)
        # self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        
        # if bias:
        #     self.bias = nn.Parameter(torch.zeros(out_channels))
        # else:
        #     self.bias = None
            
        # # 2. Robust fan-in calculation must account for the divided input channels
        # fan_in = (in_channels // groups) * kernel_size[0] * kernel_size[1]
        # self.scale = math.sqrt(2.0 / fan_in)


        self.conv =nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
    def forward(self, x):
        return self.conv(x)
        # weight = self.weight * self.scale
        # # 3. Pass the groups parameter to F.conv2d
        # return F.conv2d(x, weight, self.bias, self.stride, self.padding, dilation=1, groups=self.groups)