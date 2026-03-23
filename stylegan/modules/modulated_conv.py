import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


from .blocks import EQLRLinear
# Assuming EQLRLinear is the class we perfected earlier
# from eq_lr import EQLRLinear 

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, w_dim, demodulate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.demodulate = demodulate
        
        # 1. Base weights strictly N(0, 1) for Equalized LR
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        
        # 2. Equalized Learning Rate scale factor (He Constant)
        fan_in = in_channels * kernel_size ** 2
        self.eq_lr_scale = (2 / fan_in) ** 0.5
        
        self.style_proj = EQLRLinear(in_features=w_dim,
                                      out_features=in_channels)

    def forward(self, x, w):
        batch, in_c, height, width = x.shape
        
      
        style = self.style_proj(w) + 1.0
        style = style.view(batch, 1, in_c, 1, 1)
      
        weight = self.eq_lr_scale * self.weight * style
        
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * demod
      
        weight = weight.view(batch * self.out_channels, in_c, self.kernel_size, self.kernel_size)
        
        x = x.view(1, batch * in_c, height, width)
        
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        
        out = out.view(batch, self.out_channels, height, width)
        
        return out