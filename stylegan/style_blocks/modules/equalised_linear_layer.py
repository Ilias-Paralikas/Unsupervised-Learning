import torch 
import torch.nn as nn
import torch.nn.functional as F

class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, lr_mul=0.01):
        super().__init__()
        # FIX: Divide the standard normal initialization by lr_mul!
        # This gives the weight a large initial value (std=100) so that when 
        # it is multiplied by the tiny self.scale (which includes * 0.01),
        # the effective variance during the forward pass is exactly 1.0 (He Initialization).
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.scale = (2 / in_features) ** 0.5 * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)