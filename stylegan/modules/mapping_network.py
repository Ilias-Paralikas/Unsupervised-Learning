import torch 
import torch.nn as nn

from .blocks import PixelNorm, EQLRLinear


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, depth=8, lr_mul=0.01):
        super().__init__()
        
        layers = [PixelNorm()]
        
        for i in range(depth):
            in_features = z_dim if i == 0 else w_dim
            
            layers.append(EQLRLinear(in_features=in_features,
                                     out_features=w_dim,
                                     lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)