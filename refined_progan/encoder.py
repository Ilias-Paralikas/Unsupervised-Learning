import torch
import torch.nn as nn
import torch.nn.functional as F


from .blocks.convs import CustomConv2d
from .blocks import DownsampleBlock
from .blocks import ConvBlock

class Encoder(nn.Module):
    def __init__(self, 
                z_dim,
                channels,
                img_channels=3,
                block_depth=2,
                residual=True,
                use_norm=True):
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels.copy()
        self.channels.reverse()
        self.img_channels = img_channels
        self.block_depth =block_depth
        self.residual =residual
        self.use_norm = use_norm 

        self.initial_block = ConvBlock(self.img_channels,
                                             self.channels[0],
                                             use_norm=self.use_norm)
        self.blocks = nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.blocks.append(
               DownsampleBlock(self.channels[i],
                               self.channels[i+1],
                               use_norm=self.use_norm,
                               depth=self.block_depth,
                               residual=self.residual)
            )

        self.final_block = nn.Sequential(
            ConvBlock(self.channels[-1] ,
                      self.channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      use_norm=self.use_norm),
            # 2. 4x4 Valid Conv to collapse the 4x4 spatial dimensions to 1x1
            ConvBlock(self.channels[-1],
                      self.channels[-1],
                      kernel_size=4,
                      stride=1,
                      padding=0,  # Valid padding avoids pooling
                      use_norm=self.use_norm),
            CustomConv2d(self.channels[-1],self.z_dim,1,1,0)

        )
   
    
    def forward(self, x):
        y = self.initial_block(x[-1])

        for i in range(len(self.channels)-1):
            y = self.blocks[i](y)
        

        y = self.final_block(y)
        return y.view(y.shape[0],-1,1,1)
