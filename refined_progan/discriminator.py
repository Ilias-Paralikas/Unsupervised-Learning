import torch
import torch.nn as nn
import torch.nn.functional as F


from .blocks.convs import CustomConv2d
from .blocks import DownsampleBlock
from .blocks import ConvBlock

class Discriminator(nn.Module):
    def __init__(self, 
                channels,
                img_channels=3,
                block_depth=2,
                residual=True):
        super().__init__()
        self.channels = channels.copy()
        self.channels.reverse()
        self.img_channels = img_channels
        self.block_depth =block_depth
        self.residual =residual

        self.initial_block = ConvBlock(self.img_channels,
                                             self.channels[0],
                                             use_norm=False)
        self.blocks = nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.blocks.append(
               DownsampleBlock(self.channels[i]+self.img_channels+1 if i>0 else self.channels[i]+1,
                               self.channels[i+1],
                               use_norm=False,
                               depth=self.block_depth,
                               residual=self.residual)
            )

        self.final_block = nn.Sequential(
            ConvBlock(self.channels[-2] + self.img_channels + 1,
                      self.channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      use_norm=False),
            # 2. 4x4 Valid Conv to collapse the 4x4 spatial dimensions to 1x1
            ConvBlock(self.channels[-1],
                      self.channels[-1],
                      kernel_size=4,
                      stride=1,
                      padding=0,  # Valid padding avoids pooling
                      use_norm=False),
            CustomConv2d(self.channels[-1],1,1,1,0)

        )
   
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


    def combine_features_rgb(self,rgb,features):
        return torch.cat([features,rgb],dim=1)
    
    def forward(self, x):
        y = self.initial_block(x[-1])
        y = self.minibatch_std(y)

        for i in range(len(self.channels)-1):
            y = self.blocks[i](y)
            y = self.combine_features_rgb(x[-i-2],y)
            y = self.minibatch_std(y)

        y = self.final_block(y)
        return y.view(y.shape[0],-1)
