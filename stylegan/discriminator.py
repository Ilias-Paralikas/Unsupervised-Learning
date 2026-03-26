import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import EQLRDeepConvBlock
from .modules.blocks import EQLRConv2d, EQLEConvBlock

class Discriminator(nn.Module):
    def __init__(self, 
                channels,
                block_depth=2,
                residual=True,
                img_channels=3):
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
        self.channe_adaptors = nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.blocks.append(
               EQLRDeepConvBlock(in_channels=self.channels[i],
                               out_channels=self.channels[i+1],
                               depth=block_depth,
                               residual=residual,
                               use_norm=False)
            )
            self.channe_adaptors.append(EQLRConv2d(in_channels=self.channels[i],
                                   out_channels=self.channels[i+1],
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True))
        self.downsample =nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block =nn.Sequential(
            EQLEConvBlock(in_channels=self.channels[-1] +1,
                      out_channels=self.channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      use_norm=False),
            # 2. 4x4 Valid Conv to collapse the 4x4 spatial dimensions to 1x1
            EQLEConvBlock(in_channels=self.channels[-1],
                      out_channels=self.channels[-1],
                      kernel_size=4,
                      stride=1,
                      padding=0,  # Valid padding avoids pooling
                      use_norm=False),
            EQLRConv2d(in_channels=self.channels[-1],
                                    out_channels=1,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True)
        )

        self.residual_scale = 1/ (2**0.5)

    def minibatch_std(self, x, group_size=4, eps=1e-8):
        b, c, h, w = x.shape
        # 1. Fallback for batch size of 1
        if b == 1:
            zeros = torch.zeros(b, 1, h, w, device=x.device, dtype=x.dtype)
            return torch.cat([x, zeros], dim=1)
        # 2. Adjust group size if it doesn't divide the batch evenly
        group_size = min(group_size, b)
        if b % group_size != 0:
            group_size = b
            
        # 3. Grouped statistics with epsilon for numerical stability
        y = x.view(b // group_size, group_size, c, h, w)
        var = torch.var(y, dim=1, unbiased=False)
        std = torch.sqrt(var + eps)
        
        # 4. Average over channels and spatial dimensions
        mean_std = std.mean(dim=[1, 2, 3], keepdim=True)
        mean_std = mean_std.unsqueeze(1) # Shape: (G, 1, 1, 1, 1)

        mean_std = mean_std.expand(-1, group_size, -1, h, w) # Shape: (G, group_size, 1, h, w)
            
            # Flatten back to (batch, channels, height, width)
        mean_std = mean_std.reshape(b, 1, h, w)    
        return torch.cat([x, mean_std], dim=1)
    
    def forward(self, x,return_features=False):
        features = []
        
        residual = self.from_rgb(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](residual)
            if return_features :
                features.append(x)
            residual = self.channe_adaptors[i](residual)
            residual = self.residual_scale * (residual + x)
            
            # Downsample EVERY time to ensure the final spatial map is 4x4
            residual = self.downsample(residual)

        residual = self.minibatch_std(residual)
        residual = self.final_block(residual)
        residual = residual.view(residual.shape[0], -1)
        if return_features:
            return residual, features
        return residual