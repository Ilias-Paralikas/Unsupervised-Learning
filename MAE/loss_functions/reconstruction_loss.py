import torch
import torch.nn as nn
import torch.nn.functional as F


class MAEReconstructionLoss(nn.Module):
    """Standard MAE masked-patch MSE (unchanged)."""

    def __init__(
        self,
        patch_size:       int   = 16,
        in_channels:      int   = 3,
        normalize_target: bool  = True,
        weight:           float = 1.0,
    ):
        super().__init__()
        self.patch_size       = patch_size
        self.in_channels      = in_channels
        self.normalize_target = normalize_target
        self.weight           = weight

    def patchify(self, imgs):
        B, C, H, W = imgs.shape
        P = self.patch_size
        h = w = H // P
        x = imgs.reshape(B, C, h, P, w, P)
        x = x.permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, h * w, P * P * C)

    def forward(self, pred, target_imgs, mask):
        target = self.patchify(target_imgs)
        if self.normalize_target:
            mean   = target.mean(dim=-1, keepdim=True)
            std    = target.var(dim=-1, keepdim=True, unbiased=False).sqrt() + 1e-6
            target = (target - mean) / std
        loss_per_patch = (pred - target).pow(2).mean(dim=-1)
        loss = (loss_per_patch * mask).sum() / (mask.sum() + 1e-8)
        return self.weight * loss
