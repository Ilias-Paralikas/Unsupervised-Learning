import torch
import torch.nn as nn
import torch.nn.functional as F

from .style_conv_block import StyleConvBlock

class StackedStyleBlocks(nn.Module):
    def __init__(self,  
                 in_channels, 
                out_channels,
                kernel_size, 
                w_dim, 
                depth=2,
                demodulate=True):
        
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(depth-1):
            self.blocks.append(
                StyleConvBlock(
                    in_channels=in_channels, 
                    out_channels=in_channels, 
                    kernel_size=kernel_size, 
                    w_dim=w_dim,
                    demodulate=demodulate
                )
            )
        self.blocks.append( StyleConvBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_size, 
                    w_dim=w_dim,
                    demodulate=demodulate
                ))
    def forward(self, x, w):
        for block in self.blocks:
            x = block(x,w)
        return x