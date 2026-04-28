import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.tv_weight = weight

    def forward(self, masks):
        # masks shape: (B, N, H, W)
        # Calculate the absolute difference between adjacent pixels horizontally and vertically
        tv_h = torch.mean(torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1]))
        
        return self.tv_weight * (tv_h + tv_w)