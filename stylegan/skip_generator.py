

import torch
import torch.nn as nn
import torch.nn.functional as F

from .style_blocks import StyleConvBlock, ToRGB, MappingNetwork

class SkipGenerator(nn.Module):
    def __init__(self, 
                 z_dim,
                 w_dim, 
                 channels,
                 mapping_network_depth,
                 number_of_vectorizers=None,
                img_channels=3):
        super().__init__()
        self.w_dim = w_dim
        self.number_of_vectorizers = number_of_vectorizers

        self.mapping_network = MappingNetwork(z_dim, w_dim,num_layers=mapping_network_depth)
        # 4x4 Constant Input
        self.constant_input = nn.Parameter(torch.ones(1, channels[0], 4, 4))
        
        # Initial 4x4 blocks
        self.initial_block = StyleConvBlock(channels[0], channels[0], kernel_size=3, w_dim=w_dim)
        self.initial_rgb = ToRGB(channels[0], img_channels, w_dim=w_dim)

        self.blocks = nn.ModuleList()
        self.rgb_blocks = nn.ModuleList()
        
        # Upsampling blocks (starts at 8x8)
        for i in range(len(channels) - 1):            
            self.blocks.append(StyleConvBlock(channels[i], channels[i+1], kernel_size=3, w_dim=w_dim))
            self.rgb_blocks.append(ToRGB(channels[i+1], img_channels, w_dim=w_dim))
        self.activation = nn.Tanh()
    
    def forward(self, z):
        batch_size = z.shape[0]
        # add the option for non vectorized forward pass, mainly for testing
        if self.number_of_vectorizers is not None :
            number_of_vectorizers = x.shape[1]
            assert number_of_vectorizers == self.number_of_vectorizers
            effective_batch_size = batch_size*number_of_vectorizers

            z = z.view(effective_batch_size,z.shape[2])
        else:
            effective_batch_size = batch_size

        # w = self.mapping_network(z)
        w = z
        x = self.constant_input.expand(effective_batch_size, -1, -1, -1)
        x = self.initial_block(x, w)
        rgb = self.initial_rgb(x, w)

    
        for i in range(len(self.blocks)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)

            x = self.blocks[i](x, w)
            
            new_rgb = self.rgb_blocks[i](x, w)
            rgb = rgb + new_rgb
            
        # if not vectorized, no need to untagle the batch size
        if self.number_of_vectorizers is not None :
            rgb = rgb.view(batch_size,number_of_vectorizers,*rgb.shape[1:])

        rgb = self.activation(rgb)
        return rgb 
    