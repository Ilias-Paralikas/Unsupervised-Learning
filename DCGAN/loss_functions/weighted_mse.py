import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, input, target):
        return self.weight * torch.mean((input - target) ** 2)