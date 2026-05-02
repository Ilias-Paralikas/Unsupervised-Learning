import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentReconstructionLoss(nn.Module):
    """
    Per-component reconstruction loss applied BEFORE blending.

    Plain pixel MSE against the original image, averaged equally over all
    components.  No seg_probs weighting — every decoder gets an identical
    gradient regardless of what the segmentation network assigns to it.

    Parameters
    ----------
    weight : float  — overall loss scale
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        component_imgs: list,               # N_comp x (B, C, H, W)
        target_imgs:    torch.Tensor,        # (B, C, H, W)
        mask:           torch.Tensor = None, # (B, N) optional — restrict to masked patches
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=target_imgs.device)

        for img_n in component_imgs:
            mse = (img_n - target_imgs).pow(2)              # (B, C, H, W)

            if mask is not None:
                B, N = mask.shape
                h = w = int(N ** 0.5)
                P       = target_imgs.shape[-1] // h
                mask_px = mask.reshape(B, 1, h, w).float()
                mask_px = F.interpolate(mask_px, scale_factor=P, mode="nearest")
                mse     = mse * mask_px
                loss_n  = mse.sum() / (mask_px.sum() * target_imgs.shape[1] + 1e-8)
            else:
                loss_n = mse.mean()

            total = total + loss_n

        return self.weight * total / len(component_imgs)
