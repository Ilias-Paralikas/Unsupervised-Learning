import torch
import torch.nn as nn

from .modules import Decoder
from .modules.blocks import LinearNeuralNetwork

class SegmentationDecoder(nn.Module):
    def __init__(self,
                 out_channels=5,
                 linear_in_neurons=1024,
                 linear_out_neurons=256,
                 linear_layer_dim=[],
                 linear_norm=nn.LayerNorm,
                 linear_activation=nn.ReLU(inplace=True),
                 decoder_channels=[2048,1024,512,256,128,64,32,16],
                 first_conv_size= 4,
                 double_conv=True,
                 decoder_norm=nn.BatchNorm2d,
                 decoder_activation=nn.ReLU(inplace=True),
                 decoder_output_layer_activation=nn.Sigmoid()):
        super().__init__()
        # linear layer parameters
       
        self.linear_in_neurons = linear_in_neurons  
        self.linear_out_neurons = linear_out_neurons
        self.linear_layer_dim = linear_layer_dim.copy()
        self.linear_norm = linear_norm
        self.linear_activation = linear_activation
        # decoder parameters
        self.out_channels = out_channels
        self.decoder_channels = decoder_channels
        self.first_conv_size = first_conv_size
        self.double_conv = double_conv
        self.decoder_norm = decoder_norm
        self.decoder_activation = decoder_activation
        self.decoder_output_layer_activation = decoder_output_layer_activation

        self.linear = LinearNeuralNetwork(in_neurons=self.linear_in_neurons,
                                            out_neurons=self.linear_out_neurons,
                                            layer_dims=self.linear_layer_dim,
                                            norm=self.linear_norm,
                                            activation=self.linear_activation)

     
        self.decoder = Decoder(bottleneck_dim=self.linear_out_neurons,
                               out_channels=self.out_channels,
                               channels=self.decoder_channels,
                               output_layer_activation=self.decoder_output_layer_activation,
                               double_conv=self.double_conv,
                               first_conv_size=self.first_conv_size,
                               activation=self.decoder_activation,
                               norm=self.decoder_norm)
          
    def forward(self, x):
        x = self.linear(x)
        x = x.view(*x.shape,-1,1)
        x = self.decoder(x)
        return x