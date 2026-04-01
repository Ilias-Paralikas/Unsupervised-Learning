import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ResidualBlock
from .modules.blocks.eq_lr_layers import EQLRConv2d
from .modules.blocks import ConvBlock


class Encoder(nn.Module):
    def __init__(self, 
                channels,
                z_dim,
                block_depth=2,
                residual=True,
                img_channels=3,
                use_norm=False):
        super().__init__()
        self.channels = channels.copy()
        self.channels.reverse()

        self.from_rgb = EQLRConv2d(in_channels=img_channels,
                                   out_channels=self.channels[0],
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True)
        
        self.blocks = nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.blocks.append(
               ResidualBlock(in_channels=self.channels[i],
                               out_channels=self.channels[i+1],
                               depth=block_depth,
                               residual=residual,
                               use_norm=use_norm)
            )

        self.downsample =nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block =nn.Sequential(
            ConvBlock(in_channels=self.channels[-1],
                      out_channels=self.channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      use_norm=use_norm),
            EQLRConv2d(in_channels=self.channels[-1],
                                    out_channels=z_dim,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True)
        )

    def forward(self, x):
        
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
            x = self.downsample(x)
        x = self.final_block(x)
        return x