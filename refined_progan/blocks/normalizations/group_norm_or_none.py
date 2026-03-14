import torch
import torch.nn as nn



class GroupNormOrNone(nn.Module):
    def __init__(self, out_channels, groups=8):
        super().__init__()
        if groups is not None:
            actual_groups = groups if out_channels % groups == 0 else out_channels
            self.norm = nn.GroupNorm(num_groups=actual_groups, num_channels=out_channels)
        else:
            self.norm =nn.Identity()
    
    def forward(self, x):
        return self.norm(x)
