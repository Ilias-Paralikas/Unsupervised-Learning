import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ResidualBlock
from .modules.blocks.eq_lr_layers import EQLRConv2d
from .modules.blocks import ConvBlock


class SegmentationGenerator(nn.Module):
    def __init__(self, 
                channels,
                z_dim,
                number_of_components,
                block_depth=2,
                residual=True,
                use_norm=True):
        super().__init__()
        self.channels = channels.copy()
        
        self.from_noise = ResidualBlock(in_channels=z_dim,
                                        out_channels=self.channels[0],
                                        depth=block_depth,
                                        residual=residual,
                                        use_norm=use_norm)
        self.blocks = nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.blocks.append(
               ResidualBlock(in_channels=self.channels[i],
                               out_channels=self.channels[i+1],
                               depth=block_depth,
                               residual=residual,
                               use_norm=use_norm)
            )


        self.final_block =nn.Sequential(
            ConvBlock(in_channels=self.channels[-1],
                      out_channels=self.channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      use_norm=use_norm),
            EQLRConv2d(in_channels=self.channels[-1],
                                    out_channels=number_of_components,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True)
        )

    def forward(self, x):
      
        x = self.from_noise(x)
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = block(x)
        x = self.final_block(x)
    
        return x