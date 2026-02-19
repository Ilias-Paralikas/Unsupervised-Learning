import torch.nn as nn

from .blocks import ConvBlock, UpConv

class Decoder(nn.Module):
    def __init__(self,
                bottleneck_dim,
                out_channels,
                output_layer_activation,
                channels,
                first_conv_size,
                norm,
                activation,
                double_conv=True):
 
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.out_channels =out_channels
        self.channels = channels.copy()
        self.first_conv_size=first_conv_size
        self.norm = norm
        self.double_conv = double_conv
        self.activation = activation
        self.output_layer_activation = output_layer_activation
        dec_layers = nn.ModuleList([ConvBlock(self.bottleneck_dim, 
                                              self.channels[0], 
                                              kernel_size=self.first_conv_size, 
                                              stride=1, 
                                              padding=0, 
                                              bias=False,
                                              norm=nn.Identity,
                                              activation=self.activation,
                                              Transpose=True)])
        for i in range(len(channels)-1):
            dec_layers.append(UpConv(self.channels[i], 
                                     self.channels[i+1],
                                     norm=self.norm,
                                    double_conv=self.double_conv,
                                    activation=self.activation))

        dec_layers.append(nn.Conv2d(self.channels[-1],
                                     self.out_channels, 
                                     kernel_size=1, 
                                     stride=1, 
                                     padding=0))
        self.decoder = nn.Sequential(*dec_layers)
    def forward(self, x):
        x = self.decoder(x)
        x = self.output_layer_activation(x)
        return x
