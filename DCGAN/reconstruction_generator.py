import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ResidualBlock
from .modules.blocks.eq_lr_layers import EQLRConv2d
from .modules.blocks import ConvBlock, Vectorizer


class ReconstructionGenerator(nn.Module):
    def __init__(self, 
                channels,
                z_dim,
                number_of_components=2,
                vector_dim=128,
                vectorizer_linear_layer_multipliers=[4],
                degrees_of_freedom=12):
        super().__init__()
        self.channels = channels.copy()

        self.flatten_layer = EQLRConv2d(in_channels=z_dim,
                                        out_channels=z_dim,
                                        kernel_size=4,
                                        stride=1,
                                        padding=0,
                                        bias=True)

        self.vectorizers = nn.ModuleList([
            Vectorizer(in_neuroes=z_dim,
                       vector_dim=vector_dim,
                       degrees_of_freedom=degrees_of_freedom,
                       linear_layer_multipliers=vectorizer_linear_layer_multipliers)
                       for _ in range(number_of_components)
        ])
        
     

    def forward(self, x):
        x  = self.flatten_layer(x)
        vectors = []
        for vectorizer in self.vectorizers:
            vectors.append(vectorizer(x))
        vectors = torch.cat(vectors, dim=1)
        return vectors