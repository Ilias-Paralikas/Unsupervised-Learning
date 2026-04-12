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
            # U-Net concatenation: upsampled features (channels[i]) 
            # + skip connection (channels[i+1])
            in_ch = self.channels[i] + self.channels[i+1]
            
            self.blocks.append(
               ResidualBlock(in_channels=in_ch,
                               out_channels=self.channels[i+1],
                               depth=block_depth,
                               residual=residual,
                               use_norm=use_norm)
            )

        self.final_block = nn.Sequential(
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

    def forward(self, x, skips):
        """
        x: bottleneck feature from encoder (B, z_dim, H, W)
        skips: list of intermediate features from the encoder
        """
        x = self.from_noise(x)
        
        for i, block in enumerate(self.blocks):
            # 1. Upsample
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            # 2. Get the corresponding skip connection
            # skips is ordered [H, H/2, H/4]. We need them in reverse order for decoding.
            skip = skips[-(i + 1)] 
            
            # 3. Concatenate along the channel dimension (dim=1)
            x = torch.cat([x, skip], dim=1)
            
            # 4. Process
            x = block(x)
            
        x = self.final_block(x)
    
        return x