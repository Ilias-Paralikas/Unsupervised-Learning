import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ResidualBlock
from .modules.blocks.eq_lr_layers import EQLRConv2d
from .modules.blocks import ConvBlock, Vectorizer


class ReconstructionGenerator(nn.Module):
    def __init__(self, 
                channels,
                z_dim,
                number_of_components=2,
                vector_dim=128,
                vectorizer_linear_layer_dim=[1024],
                degrees_of_freedom=12,
                block_depth=2,
                residual=True,
                img_channels=3,
                use_norm=True):
        super().__init__()
        self.channels = channels.copy()

        self.flatten_layer = EQLRConv2d(in_channels=z_dim,
                                        out_channels=z_dim,
                                        kernel_size=4,
                                        stride=1,
                                        padding=0,
                                        bias=True)

        self.vectorizers = nn.ModuleList([
            Vectorizer(in_neuroes=z_dim,
                       vector_dim=vector_dim,
                       degrees_of_freedom=degrees_of_freedom,
                       linear_layer_dim=vectorizer_linear_layer_dim)
                       for _ in range(number_of_components)
        ])
        
        self.first_transpose = nn.ConvTranspose2d(in_channels=vector_dim,
                                                  out_channels=self.channels[0],
                                                  kernel_size=4,
                                                  stride=1,
                                                  padding=0,
                                                  bias=True)
        # 4x4 base features
        self.from_noise = ResidualBlock(in_channels=self.channels[0],
                                        out_channels=self.channels[0],
                                        depth=block_depth,
                                        residual=residual,
                                        use_norm=use_norm)

        
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.blocks.append(
               ResidualBlock(in_channels=self.channels[i],
                               out_channels=self.channels[i+1],
                               depth=block_depth,
                               residual=residual,
                               use_norm=use_norm)
            )
            self.to_rgb.append(EQLRConv2d(in_channels=self.channels[i+1],
                                   out_channels=img_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True))

     

    def forward(self, x):
        x  = self.flatten_layer(x)
        vectors = []
        for vectorizer in self.vectorizers:
            vectors.append(vectorizer(x))
        vectors = torch.cat(vectors, dim=1)

        x = vectors.view(vectors.shape[0], vectors.shape[1], vectors.shape[2], 1, 1)
        batch_size = x.shape[0]
        number_of_components = x.shape[1]

        effective_batch_size = batch_size * number_of_components


        x  = x.view(effective_batch_size, *x.shape[2:])

        x = self.first_transpose(x)
        x = self.from_noise(x)
        rgb_acc = None

        # Accumulate RGB from intermediate blocks
        for rgb, block in zip(self.to_rgb,self.blocks):
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = block(x)
            
            if rgb_acc is None:
                rgb_acc = rgb(x)
            else:
                rgb_acc = F.interpolate(rgb_acc, scale_factor=2, mode='bilinear')
                rgb_acc = rgb_acc + rgb(x)

      
        rgb_acc = rgb_acc / (len(self.blocks) + 1)
        rgb_acc  = torch.tanh(rgb_acc)

        rgb_acc = rgb_acc.view(batch_size, number_of_components, *rgb_acc.shape[1:])

        return rgb_acc