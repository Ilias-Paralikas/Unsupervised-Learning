import torch.nn as nn

from .conv_block import ConvBlock
    
class ResidualDoubleConv(nn.Module):
    def __init__(self, 
                 in_channels,
                 norm,
                 activation,
                 depth=2):
        super().__init__()
        self.double_conv = nn.Sequential(
            *[ConvBlock(in_channels, in_channels, kernel_size=3, padding=1,norm=norm, activation=activation) for _ in range(depth)]
        )
    def forward(self, x):
        return self.double_conv(x) + x