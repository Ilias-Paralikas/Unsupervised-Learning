import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import Generator    



class SegmentationGenerator(nn.Module):
    def __init__(self, 
                z_dim,
                channels,
                number_of_vectorizers,
                img_channels=3,
                block_depth=2,
                residual=True,
                use_norm=True,
                final_activation=None):
        super().__init__()
        self.channels = channels.copy()
        self.number_of_vectorizers = number_of_vectorizers
        self.generator = Generator(z_dim=z_dim,
                                   channels=self.channels,
                                   img_channels=img_channels,
                                   block_depth=block_depth,
                                   residual=residual,
                                   use_norm=use_norm)
        self.final_activation = final_activation
    
    def forward(self, x):
        number_of_vectorizers = x.shape[0]
        assert number_of_vectorizers == self.number_of_vectorizers
        b_size = x.shape[1]
        x = x.view(b_size*number_of_vectorizers,x.shape[1:])
        x = self.generator(x)
        x = x.view(self.number_of_vectorizers,b_size,x.shape[1:])
        return x