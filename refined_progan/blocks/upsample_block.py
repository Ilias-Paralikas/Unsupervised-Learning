import torch
import torch.nn as nn
import torch.nn.functional as F


from .conv_block import ConvBlock
from .residual_block import ResidualBlock

class UpsampleBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 depth=2,
                 bias=True,
                 use_norm=True,
                 residual = True):
        
        super().__init__()
        self.residual = residual
        self.use_norm = use_norm
        # 1. Safety check for GroupNorm
        # If out_channels is smaller than groups, reduce groups to match out_channels
        self.channel_block = ConvBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=bias,
                    use_norm=self.use_norm
                ) 

        self.res_block =ResidualBlock(
                    channels=out_channels, 
                    depth=depth,
                    bias=bias,
                    use_norm= self.use_norm,
                    residual=self.residual
            )
        
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.channel_block(x)
        x = self.res_block(x)
        return x