

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # ProGAN runtime scaling factor (He initialization)
        fan_in = in_channels * (kernel_size ** 2)
        self.scale = (2 / fan_in) ** 0.5 

    def forward(self, x):
        # Scale weights dynamically at runtime
        return F.conv2d(x, self.weight * self.scale, self.bias, stride=self.stride, padding=self.padding)


# class CustomConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
#         super().__init__()
        
#         self.stride = stride
#         self.padding = padding
        
#         # 1. Define the raw weight parameter
#         # Shape: (out_channels, in_channels, kernel_height, kernel_width)
#         self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        
#         # 2. Define the bias (left unscaled, as discussed!)
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_channels))
#         else:
#             self.register_parameter('bias', None)
            
#         # 3. Proper initialization
#         # Kaiming/He initialization helps start the weights in a good variance range
#         nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, x):
#         # Calculate the L2 norm. 
#         # We calculate it across dim=(1, 2, 3) to treat each output filter independently.
#         # keepdim=True ensures the shape remains (out_channels, 1, 1, 1) for broadcasting.
#         weight_norm = torch.linalg.vector_norm(self.weight, ord=2, dim=(1, 2, 3), keepdim=True)
        
#         # Scale the weights by dividing by the L2 norm.
#         # We add a tiny epsilon (1e-8) to prevent division by zero if a filter dies completely.
#         weight_norm =1

#         scaled_weight = self.weight / (weight_norm + 1e-8)
        
#         # Perform the standard convolution operation using the scaled weights
#         return F.conv2d(x, scaled_weight, self.bias, self.stride, self.padding)



