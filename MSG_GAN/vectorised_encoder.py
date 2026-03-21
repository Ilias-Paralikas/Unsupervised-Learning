import torch 
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .blocks import Vectorizer



class VectorisedEncoder(nn.Module):
    def __init__(self, 
                z_dim,
                channels,
                number_of_vectorizers,
                vector_dim,
                degrees_of_freedom,
                vectorizer_linear_dim,
                img_channels=3,
                block_depth=2,
                residual=True,
                use_norm=True):
        super().__init__()
        self.channels = channels.copy()

        self.vectorizer_linear_dim = vectorizer_linear_dim.copy()
        self.encoder = Encoder(z_dim=z_dim,
                               channels=self.channels,
                               img_channels=img_channels,
                               block_depth=block_depth,
                               residual=residual,
                               use_norm=use_norm)

        self.vectorizers = nn.ModuleList([
            Vectorizer(in_neuroes=z_dim,
                       vector_dim=vector_dim,
                       degrees_of_freedom=degrees_of_freedom,
                       linear_layer_dim=self.vectorizer_linear_dim)
            for i in range(number_of_vectorizers)
        ])

    def forward(self, x):
        x = self.encoder(x)
        vectors= []
        for i in range(len(self.vectorizers)):
            vectors.append(self.vectorizers[i](x))
        return vectors