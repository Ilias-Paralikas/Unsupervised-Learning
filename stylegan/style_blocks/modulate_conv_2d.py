import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .equalised_linear_layer import EqualizedLinear


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, w_dim, demodulate=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.style_fc =EqualizedLinear(w_dim, in_channels, lr_mul=0.01)
        self.demodulate = demodulate
        self.padding = kernel_size // 2
        
 # 3. Calculate runtime scale for the convolution
        fan_in = in_channels * (kernel_size ** 2)
        self.scale = math.sqrt(2 / fan_in)

    def forward(self, x, w):
        batch_size = x.shape[0]
        # Affine transform for style
        style = self.style_fc(w) + 1.0 
        
        # Modulate weights
        w_prime = (self.weight *self.scale) * style.view(batch_size, 1, -1, 1, 1)
        
        # Demodulate weights (StyleGAN2 specific)
        if self.demodulate:
            norm = torch.rsqrt((w_prime ** 2).sum(dim=(2, 3, 4)) + 1e-8)
            w_prime = w_prime * norm.view(batch_size, -1, 1, 1, 1)
            
        # Reshape for grouped convolution to handle batch-specific weights
        # FIX: Use .reshape() instead of .view() to handle non-contiguous memory from .expand()
        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        w_prime = w_prime.reshape(-1, w_prime.shape[2], w_prime.shape[3], w_prime.shape[4])
        
        out = F.conv2d(x, w_prime, padding=self.padding, groups=batch_size)
        
        return out.view(batch_size, -1, out.shape[2], out.shape[3])