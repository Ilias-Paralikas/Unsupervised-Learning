import torch
import torch.nn as nn

class MAEReconstructionLoss(nn.Module):
    """
    MSE loss on masked patches only.

    Accepts full images (B, C, H, W) — patchification and masking are
    handled internally. No need to expose patch tensors in the training loop.

    Args:
        patch_size       : patch height = width P
        in_channels      : image channels C
        normalize_target : per-patch zero-mean / unit-var normalisation (paper default)
        weight           : scalar loss multiplier
    """

    def __init__(
        self,
        patch_size:       int   = 16,
        in_channels:      int   = 3,
        normalize_target: bool  = False,
        weight:           float = 1.0,
    ):
        super().__init__()
        self.patch_size       = patch_size
        self.in_channels      = in_channels
        self.normalize_target = normalize_target
        self.weight           = weight

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, P*P*C)"""
        p        = self.patch_size
        B, C, H, W = imgs.shape
        h, w     = H // p, W // p
        x        = imgs.reshape(B, C, h, p, w, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)

    def forward(
        self,
        pred:   torch.Tensor,   # (B, C, H, W) — full reconstruction
        target: torch.Tensor,   # (B, C, H, W) — original image
        mask:   torch.Tensor,   # (B, N)        — 1 = masked, 0 = visible
    ) -> torch.Tensor:

        pred_patches   = self.patchify(pred)    # (B, N, patch_dim)
        target_patches = self.patchify(target)  # (B, N, patch_dim)

        if self.normalize_target:
            mean           = target_patches.mean(dim=-1, keepdim=True)
            std            = target_patches.var(dim=-1, keepdim=True, unbiased=False).sqrt() + 1e-6
            target_patches = (target_patches - mean) / std

        loss_per_patch = (pred_patches - target_patches).pow(2).mean(dim=-1)  # (B, N)
        loss           = (loss_per_patch * mask).sum() / (mask.sum() + 1e-8)

        return self.weight * loss