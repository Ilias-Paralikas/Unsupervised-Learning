import torch
import torch.nn as nn
import torch.nn.functional as F


from .blocks.convs import CustomConv2d
from .blocks import ConvBlock
from .blocks import UpsampleBlock

class Generator(nn.Module):
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
        self.img_channels = img_channels
        self.block_depth = block_depth
        self.residual = residual
        self.use_norm = use_norm

        # 1. Define the initial convolution layer
        self.initial_conv = nn.Sequential(
                            ConvBlock(in_channels=z_dim, 
                                out_channels=channels[0], 
                                kernel_size=4, 
                                stride=1, 
                                padding=0, 
                                bias=True,
                                transpose=True,
                                use_norm=self.use_norm
                            ),
                            ConvBlock(in_channels=channels[0], 
                                out_channels=channels[0], 
                                kernel_size=3, 
                                stride=1, 
                                padding=1, 
                                bias=True,
                                use_norm=self.use_norm
                            )
        )

        self.blocks = nn.ModuleList([self.initial_conv])
        for i in range(1,len(self.channels)):
            self.blocks.append(
               UpsampleBlock(channels[i-1],
                             channels[i],
                             depth=self.block_depth,
                             residual=self.residual,
                             use_norm=self.use_norm)
            )


        self.rgb_layers = nn.ModuleList()
        for i in range(len(self.channels)):
            self.rgb_layers.append(
               CustomConv2d(channels[i],self.img_channels,1,1,0)
            )
        self.activation = nn.Tanh()
     
    
    def forward(self, x):
        out  = []
        for i in range(len(self.channels)):
            x = self.blocks[i](x)
            y = self.rgb_layers[i](x)
            y = self.activation(y)
            out.append(y)
            
        return out