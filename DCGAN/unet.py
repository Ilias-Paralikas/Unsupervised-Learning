import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder

class UNet(nn.Module):
    def __init__(self,
                 channels,
                 block_depth=2,
                 residual=True,
                 in_channels=3,
                 out_channels=3,
                 use_norm=False,
                 cut_connections=0):
        super().__init__()
        self.cut_connections = cut_connections
        self.encoder_channels  = channels.copy()
        # ── Encoder ────────────────────────────────────────────────────
        self.encoder = Encoder(channels=self.encoder_channels,
                               z_dim=self.encoder_channels[0],
                               block_depth=block_depth,
                               residual=residual,
                               img_channels=in_channels,
                               use_norm=use_norm)   
        
        # ── Decoder ────────────────────────────────────────────────────
        self.decoder_channels = channels.copy()
        self.skip_channels = self.decoder_channels[:-(cut_connections+1)] # the 1 is the first input x
        self.decoder = Decoder(channels=self.decoder_channels,
                               skip_channels=self.skip_channels,
                               z_dim=self.decoder_channels[0],
                               block_depth=block_depth,
                               residual=residual,
                               img_channels=out_channels,
                               use_norm=use_norm)
        


    def forward(self, x):
        x, skip_connections = self.encoder(x, return_features=True)
        x = self.decoder(x, 
                        skip_connections=skip_connections[self.cut_connections:])
        return x