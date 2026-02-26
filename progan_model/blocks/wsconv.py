import torch
import torch.nn as nn
import torch.nn.functional as F

class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        # Initialize the convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Calculate the He constant (scale factor)
        # Formula: sqrt(gain / fan_in)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        
        # Pull the bias out so we can apply it AFTER the scaled convolution
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights to N(0, 1). 
        # The scale factor will handle the variance.
        nn.init.normal_(self.conv.weight)
        self.conv.bias = None # Remove original bias to prevent double-biasing

    def forward(self, x):
        # Scale weights on the fly (Equalized Learning Rate)
        # This keeps gradients healthy and the optimizer stable.
        scaled_weight = self.conv.weight * self.scale
        
        # Use functional conv2d to apply the scaled weights
        out = F.conv2d(
            x, 
            scaled_weight, 
            bias=None, 
            stride=self.conv.stride, 
            padding=self.conv.padding
        )
        
        # Add the bias manually
        return out + self.bias.view(1, self.bias.shape[0], 1, 1)