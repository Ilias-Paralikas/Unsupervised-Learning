import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class EQLRLinear(nn.Module):
    def __init__(self, in_features, out_features, lr_mul=1.0):
        super().__init__()
        self.lr_mul = lr_mul
        
        # Divide by lr_mul during init so the effective learning rate changes during backprop
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # He initialization constant
        fan_in = in_features
        
        # Multiply by lr_mul so the forward pass magnitudes cancel out the division above
        self.scale = (math.sqrt(2.0 / fan_in)) * lr_mul 
    
    def forward(self, x):
        weight = self.weight * self.scale
        
        # Bias is directly scaled by lr_mul at runtime
        bias = self.bias * self.lr_mul
        
        return F.linear(x, weight, bias)