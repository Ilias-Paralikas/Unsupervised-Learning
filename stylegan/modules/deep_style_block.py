import torch 
import torch.nn as nn


from .style_block import StyleBlock

class DeepStyleBlock(nn.Module):    
    def __init__(self, in_channels, out_channels,w_dim,kernel_size=3,depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            StyleBlock(in_channels=in_channels, 
                       out_channels=in_channels,
                       w_dim=w_dim,
                       kernel_size=kernel_size)
            for _ in range(depth-1)
        ])
        self.blocks.append(StyleBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       w_dim=w_dim,
                                       kernel_size=kernel_size))
    def forward(self, x,w,noise=None):
        for block in self.blocks:
            x = block(x,w,noise)
        return x