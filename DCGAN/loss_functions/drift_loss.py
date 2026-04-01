import torch
import torch.nn as nn

class DriftLoss(nn.Module):
    def __init__(self, drift_weight=1.0):
        super().__init__()
        self.drift_weight = drift_weight

    def forward(self, x):
        return self.drift_weight * torch.mean((x) ** 2)