import math
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
        
        # 1. Handle both integer and tuple kernel sizes safely
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
            
        self.stride = stride
        self.padding = padding

        # 2. Use unpacking to support tuple dimensions
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # 3. Robust fan-in calculation directly from the kernel dimensions
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        
        # 4. Math.sqrt is generally preferred for clarity 
        self.scale = math.sqrt(2.0 / fan_in)
        
    def forward(self, x):
        # Dynamic scaling works perfectly as you wrote it
        weight = self.weight * self.scale
        return F.conv2d(x, weight, self.bias, self.stride, self.padding)