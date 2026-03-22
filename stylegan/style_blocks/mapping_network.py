import torch 
import torch.nn as nn
import torch.nn.functional as F

from .modules.equalised_linear_layer import EqualizedLinear

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers=8, lr_mul=0.01):
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(num_layers):
            layers.append(EqualizedLinear(in_dim, w_dim, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = w_dim
        self.mapping = nn.Sequential(*layers)
        
    def forward(self, z):
        # Normalize z first (PixelNorm equivalent for flat vectors)
        z = z / torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-8)
        return self.mapping(z)