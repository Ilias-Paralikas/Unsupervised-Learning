import torch 
import torch.nn as nn
import torch.nn.functional as F


class EQLRLinear(nn.Module):
    def __init__(self, in_features, out_features, lr_mul=1.0):
        super().__init__()
        self.lr_mul = lr_mul
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features))

        fan_in = in_features
        self.scale = ((2 / fan_in) ** 0.5) * lr_mul 
    
    def forward(self, x):
        weight = self.weight * self.scale
        bias = self.bias * self.lr_mul
        
        return F.linear(x, weight, bias)