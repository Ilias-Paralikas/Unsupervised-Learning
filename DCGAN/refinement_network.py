import torch
import torch.nn as nn

from .modules import ResidualBlock
from .modules.blocks.eq_lr_layers import EQLRConv2d


class RefinementNetwork(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 img_channels=3,
                 input_shape = (256,256),
                 pos_embedding_dim=4,
                 depth=2,
                 residual=True,
                 use_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels.copy()
        self.img_channels = img_channels

        self.pos_embedding = nn.Parameter(torch.randn(1, pos_embedding_dim, input_shape[0], input_shape[1]))
        self.refinement_blocks = nn.ModuleList([
            ResidualBlock(in_channels=in_channels+pos_embedding_dim,
                          out_channels=channels[0],
                          depth=depth,
                          use_norm=use_norm,
                          residual=residual)
        ])

        for i in range(1, len(channels)):
            self.refinement_blocks.append(
                ResidualBlock(in_channels=channels[i-1],
                              out_channels=channels[i],
                              depth=depth,
                              use_norm=use_norm,
                              residual=residual)
            )

        self.refinement_blocks.append(
            EQLRConv2d(in_channels=channels[-1],
                       out_channels=img_channels,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=True)
        )

    def forward(self, x):
        batch_pos_embedding = self.pos_embedding.repeat(x.shape[0], 1, 1, 1)
        x = torch.cat([x, batch_pos_embedding], dim=1)
        for block in self.refinement_blocks:
            x = block(x)
        return x