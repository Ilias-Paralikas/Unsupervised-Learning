import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ResidualBlock
from .modules.blocks.eq_lr_layers import EQLRConv2d
from .modules.blocks import ConvBlock
from .encoder import Encoder

class UNet(nn.Module):
    def __init__(self,
                 channels,
                 number_of_components,
                 block_depth=2,
                 residual=True,
                 img_channels=3,
                 use_norm=False,
                 cut_connections=0):
        super().__init__()
        self.cut_connections = cut_connections
        self.channels  = channels.copy()
        # ── Encoder ────────────────────────────────────────────────────
        self.encoder = Encoder(channels=self.channels,
                               z_dim=self.channels[0],
                               block_depth=block_depth,
                               residual=residual,
                               img_channels=img_channels,
                               use_norm=use_norm)   
        # ── Decoder ────────────────────────────────────────────────────
        # channels is [512, 256, 128, 64] (largest to smallest)
        dec_channels = self.channels

        
        self.from_noise = ResidualBlock(
            in_channels=self.channels[0], out_channels=dec_channels[0],
            depth=block_depth, residual=residual, use_norm=use_norm)

        num_dec = len(dec_channels) - 1
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_dec):
            # i=0 is coarsest (near bottleneck), i=num_dec-1 is finest (near image)
            # cut the last cut_connections steps = highest i = closest to image
            is_cut = i >= (num_dec - cut_connections)
            in_ch = dec_channels[i] if is_cut else dec_channels[i] + dec_channels[i + 1]
            self.decoder_blocks.append(
                ResidualBlock(in_channels=in_ch,
                              out_channels=dec_channels[i + 1],
                              depth=block_depth, residual=residual,
                              use_norm=use_norm))

        self.decoder_final = nn.Sequential(
            ConvBlock(in_channels=dec_channels[-1], out_channels=dec_channels[-1],
                      kernel_size=3, stride=1, padding=1, use_norm=use_norm),
            EQLRConv2d(in_channels=dec_channels[-1], out_channels=number_of_components,
                       kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        # ── Encode ─────────────────────────────────────────────────────
        x,skips = self.encoder(x, return_features=True)
        # ── Decode ─────────────────────────────────────────────────────
        x = self.from_noise(x)
        num_dec = len(self.decoder_blocks)

        for i, block in enumerate(self.decoder_blocks):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            is_cut = i >= (num_dec - self.cut_connections)
            if not is_cut:
                skip = skips[-(i + 1)]
                x = torch.cat([x, skip], dim=1)

            x = block(x)

        return self.decoder_final(x)