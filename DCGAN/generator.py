import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ResidualBlock
from .modules.blocks.eq_lr_layers import EQLRConv2d
from .modules.blocks import ConvBlock


class Generator(nn.Module):
    def __init__(self, 
                channels,
                z_dim,
                block_depth=2,
                residual=True,
                img_channels=3,
                use_norm=True):
        super().__init__()
        self.channels = channels.copy()
        
        self.from_noise = EQLRConv2d(in_channels=z_dim,
                                   out_channels=self.channels[0],
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True)
        
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.blocks.append(
               ResidualBlock(in_channels=self.channels[i],
                               out_channels=self.channels[i+1],
                               depth=block_depth,
                               residual=residual,
                               use_norm=use_norm)
            )
            self.to_rgb.append(EQLRConv2d(in_channels=self.channels[i+1],
                                   out_channels=img_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True))


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

    def forward(self, x):

        x = self.from_noise(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            rgb_acc = self.to_rgb[0](x) if i == 0 else rgb_acc + self.to_rgb[i](x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            rgb_acc  = F.interpolate(rgb_acc, scale_factor=2, mode='nearest')
            
        x = self.final_block(x)
        x = torch.tanh(x)
        return x