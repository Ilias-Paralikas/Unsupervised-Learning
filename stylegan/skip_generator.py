import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, ToRGB



class SkipGenerator(nn.Module):
    def __init__(self, z_dim, w_dim, img_channels, channels):
        super().__init__()
        self.w_dim = w_dim
        
        # 4x4 Constant Input
        self.constant_input = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        
        # Initial 4x4 blocks
        self.initial_block = ConvBlock(channels[0], channels[0], kernel_size=3, w_dim=w_dim)
        self.initial_rgb = ToRGB(channels[0], img_channels, w_dim=w_dim)

        self.blocks = nn.ModuleList()
        self.rgb_blocks = nn.ModuleList()
        
        # Upsampling blocks (starts at 8x8)
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i+1]
            
            self.blocks.append(ConvBlock(in_ch, out_ch, kernel_size=3, w_dim=w_dim))
            self.rgb_blocks.append(ToRGB(out_ch, img_channels, w_dim=w_dim))

    def forward(self, w):
        batch_size = w.shape[0]
        
        # 1. Expand the constant input for the batch
        x = self.constant_input.expand(batch_size, -1, -1, -1)
        
        # 2. Process the initial 4x4 block
        x = self.initial_block(x, w)
        rgb = self.initial_rgb(x, w)
        
        # out_rgbs = [rgb] # UNCOMMENT THIS IF USING MSG-GAN
        
        # 3. Loop through the upsampling blocks
        for i in range(len(self.blocks)):
            # Upsample both the feature map AND the accumulated RGB image
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Apply convolution block to feature map
            x = self.blocks[i](x, w)
            
            # Get new RGB details from the feature map and add to the upscaled RGB
            new_rgb = self.rgb_blocks[i](x, w)
            rgb = rgb + new_rgb
            
            # out_rgbs.append(rgb) # UNCOMMENT THIS IF USING MSG-GAN
            
        # Return final high-res image (or out_rgbs if training with MSG-GAN loss)
        return rgb 