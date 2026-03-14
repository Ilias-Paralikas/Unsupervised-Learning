import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_block import ConvBlock
from .residual_block import ResidualBlock

class DownsampleBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 kernel_size = 3,
                 depth=2,
                 bias=True,
                 residual = True,
                 use_norm=False):
        
        super().__init__()
        self.residual = residual
        self.use_norm = use_norm

        self.channel_block = ConvBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, 
                    stride=1, 
                    padding=1, 
                    bias=bias,
                    use_norm=self.use_norm,
                ) 

        self.res_block =ResidualBlock(
                    channels=out_channels, 
                    depth=depth,
                    bias=bias,
                    use_norm=self.use_norm)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.channel_block(x)
        x = self.res_block(x)
        x = self.avg_pool(x)
        return x