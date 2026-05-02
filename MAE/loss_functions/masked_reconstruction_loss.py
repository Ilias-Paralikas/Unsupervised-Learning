import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedReconstructionLoss(nn.Module):
    """
    MSE between the blended reconstruction and the target image,
    restricted to the masked patches only.

    Focusing on masked patches gives a cleaner signal: visible patches
    had their encoder features available as context, so reconstruction
    there is trivial and uninformative for both the decoders and the
    segmentation routing.

    Parameters
    ----------
    weight : float
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pred:   torch.Tensor,  # (B, C, H, W)  blended reconstruction
        target: torch.Tensor,  # (B, C, H, W)  original image
        mask:   torch.Tensor,  # (B, N)  1 = masked patch, 0 = visible
    ) -> torch.Tensor:
        B, C, H, W = pred.shape
        N = mask.shape[1]
        h = int(N ** 0.5)
        P = H // h

        mask_px = mask.reshape(B, 1, h, h).float()
        mask_px = F.interpolate(mask_px, scale_factor=float(P), mode="nearest")  # (B, 1, H, W)

        mse  = (pred - target).pow(2)                                  # (B, C, H, W)
        loss = (mse * mask_px).sum() / (mask_px.sum() * C + 1e-8)

        return self.weight * loss
