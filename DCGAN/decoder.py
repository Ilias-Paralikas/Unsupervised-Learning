import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ResidualBlock
from .modules.blocks.eq_lr_layers import EQLRConv2d
from .modules.blocks import ConvBlock


class Decoder(nn.Module):
    def __init__(self, 
                channels,
                z_dim,
                skip_channels=[],
                block_depth=2,
                residual=True,
                img_channels=3,
                use_norm=False):
        super().__init__()
        self.channels = channels.copy()
        self.skip_channels = skip_channels.copy()

        self.from_noise = nn.Sequential(
            EQLRConv2d(in_channels=z_dim,
                                   out_channels=self.channels[0],
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True),
            ResidualBlock(in_channels=self.channels[0],
                               out_channels=self.channels[0],
                               depth=block_depth,
                               residual=residual,
                               use_norm=use_norm)
                 
        )
        
        self.blocks = nn.ModuleList()
        for i in range(len(self.channels)-1):
            if i <len(self.skip_channels):
                in_channels = self.skip_channels[i]+self.channels[i]
            else:
                in_channels = self.channels[i]

            self.blocks.append(
               ResidualBlock(in_channels=in_channels,
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
                                    out_channels=img_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True)
        )
    
    def forward(self, x, skip_connections=[]):
        
        assert len(skip_connections) == len(self.skip_channels) # becase the first is the x
        x = self.from_noise(x)
        
        for i, block in enumerate(self.blocks):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i <= len(self.skip_channels)-1:
                x = torch.cat([x, skip_connections[-(i + 1)]], dim=1)
            x = block(x)


        x = self.final_block(x)
        
        return x