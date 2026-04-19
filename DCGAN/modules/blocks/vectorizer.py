import torch
import torch.nn as nn

from .eq_lr_layers import EQLRLinear
from .normalizations import PixelNorm

class Vectorizer(nn.Module):
    def __init__(self,
                 in_neuroes,
                 vector_dim,
                 degrees_of_freedom,
                 use_matrix_multiplication=False,
                 linear_layer_multipliers=[]):
        super().__init__()
        self.in_neuroes = in_neuroes
        self.vector_dim = vector_dim
        self.degrees_of_freedom = degrees_of_freedom
        self.linear_layer_multipliers = linear_layer_multipliers.copy()
        self.use_matrix_multiplication = use_matrix_multiplication

        self.linear_layer_dim = [int(m * self.degrees_of_freedom) for m in linear_layer_multipliers]
        self.linear_layer_dim.append(self.degrees_of_freedom)

        # 1. Start with Flatten and PixelNorm
        linear_layer = nn.ModuleList([
            nn.Flatten(),
            PixelNorm(), # Only applied once at the start!
            EQLRLinear(self.in_neuroes, self.linear_layer_dim[0])
        ])
        

        # 3. Hidden layers (No LayerNorm!)
        for i in range(len(self.linear_layer_dim)-1):
            linear_layer.append(nn.LeakyReLU(0.2, inplace=True)) # LeakyReLU is standard for GANs
            linear_layer.append(EQLRLinear(self.linear_layer_dim[i], self.linear_layer_dim[i+1]))


        if not self.use_matrix_multiplication:
            linear_layer.append(nn.LeakyReLU(0.2, inplace=True))
            linear_layer.append(EQLRLinear(self.linear_layer_dim[-1], self.vector_dim))
            
        self.linear = nn.Sequential(*linear_layer)
        self.vectors = nn.Parameter(torch.randn(self.degrees_of_freedom, self.vector_dim),
                                     requires_grad=True)
        self.pixel_norm = PixelNorm() # Add a standalone PixelNorm

    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear(x)
        if self.use_matrix_multiplication:
            x = x.view(batch_size, self.degrees_of_freedom)
            x = torch.matmul(x, self.vectors)
        x = self.pixel_norm(x)
        x = x.view(batch_size, 1, x.shape[1])
        return x