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
        
        # 4x4 base features
        self.from_noise = ResidualBlock(in_channels=z_dim,
                                        out_channels=self.channels[0],
                                        depth=block_depth,
                                        residual=residual,
                                        use_norm=use_norm)
        
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

        # RESTORE the final blocks for proper activation before the last RGB projection
        self.final_feature = ConvBlock(in_channels=self.channels[-1],
                                       out_channels=self.channels[-1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       use_norm=use_norm)
                                       
        self.final_to_rgb = EQLRConv2d(in_channels=self.channels[-1],
                                       out_channels=img_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=True)

    def forward(self, x):
        x = self.from_noise(x)
        rgb_acc = None

        # Accumulate RGB from intermediate blocks
        for i, block in enumerate(self.blocks):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = block(x)
            
            if rgb_acc is None:
                rgb_acc = self.to_rgb[i](x)
            else:
                rgb_acc = F.interpolate(rgb_acc, scale_factor=2, mode='nearest')
                rgb_acc = rgb_acc + self.to_rgb[i](x)
            
        # Add the final activated refinement
        x = self.final_feature(x)
        rgb_acc = rgb_acc + self.final_to_rgb(x)
        
        # CRITICAL FIX: Scale down the accumulated value so Tanh doesn't saturate!
        # We added len(self.blocks) + 1 total RGB maps together.
        rgb_acc = rgb_acc / (len(self.blocks) + 1)

        return torch.tanh(rgb_acc)