import torch
import torch.nn as nn
import torch.nn.functional as F


from .modules import MappingNetwork, ToRGB, StyleBlock, DeepStyleBlock
class StyleGenerator(nn.Module):
    def __init__(self,
                 z_dim,
                 w_dim,
                 mapping_network_depth,
                 mapping_network_lr_mul,
                 channels,
                 block_depth=2,
                 img_channels=3,
                 const_input_shape=(512,4,4)):
        super().__init__()

        self.register_buffer("pl_mean", torch.zeros(1))

        self.channels = channels.copy()
        self.mapping_network = MappingNetwork(z_dim=z_dim,
                                               w_dim=w_dim, 
                                                depth=mapping_network_depth, 
                                                lr_mul=mapping_network_lr_mul)

        self.contant_input = nn.Parameter(torch.randn(1,*const_input_shape))

        self.blocks = nn.ModuleList([StyleBlock(in_channels=const_input_shape[0],
                                                 out_channels=channels[0],
                                                 w_dim=w_dim)])
        self.rgb_blocks = nn.ModuleList([ToRGB(in_channels=channels[0],
                                               img_channels=img_channels,
                                               w_dim=w_dim)])
    
        for i in range(1,len(channels)):
            self.blocks.append(DeepStyleBlock(in_channels=channels[i-1],
                                                out_channels= channels[i],
                                                w_dim=w_dim,
                                                depth=block_depth))
            self.rgb_blocks.append(ToRGB(in_channels=channels[i], 
                                         w_dim=w_dim,
                                         img_channels=img_channels))
        
    def forward(self, 
                z,
                style_mixing_prob=0,
                return_w=False,
                skip_mapping_network=False):
        batch_size = z.shape[0]

        # sometimes we want to pass the W instead of the Z vector. 
        # If that is the case, we dont want style mixing
        if skip_mapping_network:
            w = z
            style_mixing_prob = 0.0
        else:
            w = self.mapping_network(z)
       

        # for PLR we want constant W vector se we disable style mixing
        if return_w:
            style_mixing_prob = 0.0

        style_mixing = torch.rand(1).item() < style_mixing_prob
        if style_mixing:
            z_2 = torch.randn_like(z)
            w_2 = self.mapping_network(z_2)

            style_mixing_layer = torch.randint(1,len(self.blocks),size=(1,)).item()
        
        x = self.contant_input.repeat(batch_size, 1, 1, 1)
        x = self.blocks[0](x, w)
        rgb = self.rgb_blocks[0](x, w)

        for i in range(1,len(self.blocks)):
            if style_mixing: 
                if i == style_mixing_layer:
                    w = w_2
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.blocks[i](x, w)
    
            rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
            rgb += self.rgb_blocks[i](x, w)

        if return_w:
            return rgb,w
        return rgb
        
