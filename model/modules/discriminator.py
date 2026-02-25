import torch
import torch.nn as nn
from .encoder import Encoder
from .blocks import LinearNeuralNetwork

class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels,
        encoder_channels,
        encoder_norm,
        encoder_activation,
        input_size,
        linear_out_neurons,
        linear_layer_dims,
        linear_norm,
        linear_activation,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.encoder_norm = encoder_norm
        self.encoder_activation = encoder_activation
        self.input_size = input_size
        
        self.linear_out_neurons = linear_out_neurons
        self.linear_layer_dims = linear_layer_dims
        self.linear_norm = linear_norm
        self.linear_activation = linear_activation

        # 1. Reuse the Encoder module
        self.encoder = Encoder(
            in_channels=self.in_channels,
            channels=self.encoder_channels,
            norm=self.encoder_norm,
            activation=self.encoder_activation,
            input_size=self.input_size,
        )

        # Calculate input neurons for the linear part dynamically
        with torch.no_grad():
            dummy_x = torch.randn(1, self.in_channels, *self.input_size)
            # Encoder output is (B, F) because Encoder has nn.Flatten() at the end
            dummy_out = self.encoder(dummy_x) 
            linear_in_neurons = dummy_out.shape[1]

        # 2. Reuse the LinearNeuralNetwork block
        self.classifier = LinearNeuralNetwork(
            in_neurons=linear_in_neurons,
            out_neurons=self.linear_out_neurons,
            layer_dims=self.linear_layer_dims,
            norm=self.linear_norm,
            activation=self.linear_activation,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        features = self.encoder(x)
        # features: (B, F)
        logits = self.classifier(features)
        # logits: (B, 1)
        return logits
