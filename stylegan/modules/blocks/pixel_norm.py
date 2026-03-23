
import torch 
import torch.nn as nn

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        norm = torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
        return x * norm