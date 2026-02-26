import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import WSConvTranspose2d, WSConv2d, PixelNorm, ConvBlock
class Generator(nn.Module):
    def __init__(self,
                 z_dim,
                 channels,
                 img_channels):
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels.copy()
        self.img_channels = img_channels

        self.initial_block = nn.Sequential(
            WSConvTranspose2d(self.z_dim, self.channels[0], kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2,inplace=True),
            WSConv2d(self.channels[0], self.channels[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            PixelNorm(),
        )

        self.initial_rgb = WSConv2d(self.channels[0], img_channels, kernel_size=1, stride=1, padding=0)

        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb])

        for i in range(len(self.channels) - 1):
            self.prog_blocks.append(ConvBlock(self.channels[i],self.channels[i+1]))
            self.rgb_layers.append(WSConv2d(self.channels[i+1], img_channels, kernel_size=1, stride=1, padding=0))

        
    def fade_in(self, alpha,upscaled,generated):
        return (alpha*generated + (1-alpha)*upscaled)
    
    def forward(self, x,alpha,steps):
        out = self.initial_block(x)
        if steps ==0:
            out = self.initial_rgb(out)
        else:
            for step in range(steps):
                upscaled = F.interpolate(out, scale_factor=2, mode='nearest')
                out = self.prog_blocks[step](upscaled)

            out = self.rgb_layers[steps](out)

            
            if alpha <1.0:
                final_upscaled = self.rgb_layers[steps-1](upscaled)
                out = self.fade_in(alpha,final_upscaled,out)
        
        return torch.tanh(out)

    
