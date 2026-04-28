import torch
import torch.nn as nn
import torch.nn.functional as F


from .modules.blocks.eq_lr_layers import EQLRConv2d
from .modules import ResidualBlock
from .modules.blocks import ConvBlock

class Discriminator(nn.Module):
    def __init__(self, 
                channels,
                img_channels=3,
                block_depth=2,
                residual=True,
                use_norm=False):
        super().__init__()
        self.channels = channels.copy()
        self.channels.reverse()
        self.img_channels = img_channels
        self.block_depth =block_depth
        self.residual =residual
        self.use_norm = use_norm

        self.initial_block = ConvBlock(self.img_channels,
                                             self.channels[0],
                                             use_norm=self.use_norm)
        self.blocks = nn.ModuleList()
        for i in range(len(self.channels)-1):
            in_ch = self.channels[i] if i == 0 else self.channels[i] + self.img_channels
            self.blocks.append(
               ResidualBlock(in_channels=in_ch,
                             out_channels=self.channels[i+1],
                             use_norm=self.use_norm,
                             depth=self.block_depth,
                             residual=self.residual)
                )
            


        self.final_block = nn.Sequential(
            ConvBlock(self.channels[-1] + self.img_channels,
                      self.channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      use_norm=self.use_norm),
            # 2. 4x4 Valid Conv to collapse the 4x4 spatial dimensions to 1x1
            ConvBlock(self.channels[-1],
                      self.channels[-1],
                      kernel_size=4,
                      stride=1,
                      padding=0,  # Valid padding avoids pooling
                      use_norm=self.use_norm),
            EQLRConv2d(self.channels[-1],1,1,1,0)

        )
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

   


    def combine_features_rgb(self,rgb,features):
        return torch.cat([features,rgb],dim=1)
    
    def forward(self, x,return_features=False):
        y = self.initial_block(x[-1])


        features = []
        for i in range(len(self.channels)-1):
            y = self.blocks[i](y)
            y = self.avg_pool(y)
            if return_features :
                features.append(y)
            y = self.combine_features_rgb(x[-i-2],y)

        y = self.final_block(y)

        if return_features:
            return y.view(y.shape[0],-1),features
        else:
            return y.view(y.shape[0],-1)