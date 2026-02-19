import torch.nn as nn
from .conv_block import ConvBlock

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, norm, activation, depth=2):
        super().__init__()
        self.activation = activation
        
        # We need to build the layers manually to control the activation of the last one
        layers = []
        
        # First (depth-1) layers have activation
        for _ in range(depth - 1):
            layers.append(
                ConvBlock(in_channels, in_channels, 
                          kernel_size=3, padding=1, 
                          norm=norm, activation=activation)
            )
            
        # The LAST layer must have NO activation (Identity)
        # We pass nn.Identity as the activation to the ConvBlock
        layers.append(
            ConvBlock(in_channels, in_channels, 
                      kernel_size=3, padding=1, 
                      norm=norm, activation=nn.Identity())
        )
        
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        # Apply the sequence (Conv -> ReLU -> Conv)
        residual = self.double_conv(x)
        
        # Add the original input
        out = residual + x
        
        # Apply the final activation here
        return self.activation(out)
