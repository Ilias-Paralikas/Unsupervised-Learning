
import torch
import torch.nn as nn
import torch.nn.functional as F


from .conventional_blocks.convs import CustomConv2d
from .conventional_blocks import ConvBlock
from .conventional_blocks import UpsampleBlock

class SegmentationGenerator(nn.Module):
    def __init__(self, 
                 z_dim,
                channels,
                number_of_vectorizers,
                block_depth=2,
                residual=True,
                use_norm=True):
        super().__init__()
        self.z_dim = z_dim  
        self.channels = channels.copy()
        self.block_depth = block_depth
        self.residual = residual
        self.use_norm = use_norm
        self.number_of_vectorizers = number_of_vectorizers

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
            
        self.final_block = CustomConv2d(channels[-1],self.number_of_vectorizers,1,1,0)


        self.final_activation = nn.Softmax(dim=1)
    
    def forward(self, x):
        for i in range(len(self.channels)):
            x = self.blocks[i](x)
        x = self.final_block(x)
        x = self.final_activation(x)
        return x