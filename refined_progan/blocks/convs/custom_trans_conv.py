# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CustomConvTranspose2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
#         super().__init__()
        
#         self.stride = stride
#         self.padding = padding
#         self.output_padding = output_padding
        
#         # 1. Weight shape for Transpose Conv is (In, Out, K, K)
#         self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_channels))
#         else:
#             self.register_parameter('bias', None)
            
#         # 2. Initialization 
#         # For transpose conv, we usually use fan_in for upsampling stability
#         nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')

#     def forward(self, x):
#         # 3. Calculate L2 norm
#         # For Transpose Conv, each filter is associated with dim 1 (the output channel dim).
#         # We normalize across (0, 2, 3) so each out_channel has a unit-norm filter.
#         # weight_norm = torch.linalg.vector_norm(self.weight, ord=2, dim=(0, 2, 3), keepdim=True)
        
#         # # 4. Scale
#         weight_norm =1
#         scaled_weight = self.weight / (weight_norm + 1e-8)
        
#         # 5. Execute Transposed Convolution
#         return F.conv_transpose2d(
#             x, 
#             scaled_weight, 
#             self.bias, 
#             stride=self.stride, 
#             padding=self.padding, 
#             output_padding=self.output_padding
#         )

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConvTranspose2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.ConvTranspose2d(*args, **kwargs)

    def forward(self, x):
        return self.conv(x)