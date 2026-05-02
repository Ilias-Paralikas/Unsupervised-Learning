import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingLoss(nn.Module):
    """
    Reconstruction-guided routing loss.

    For each pixel, finds which component decoder reconstructed it best
    (lowest MSE against the original image) and uses that as a hard
    pseudo-label to supervise the segmentation network via cross-entropy.

    Only seg_logits receives gradients — component_imgs are detached internally.

    Parameters
    ----------
    weight : float   — overall loss scale
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

    def get_pseudo_mask(
        self,
        component_imgs: list,            # N_comp x (B, C, H, W)
        full_img:       torch.Tensor,    # (B, C, H, W)
    ) -> torch.Tensor:
        with torch.no_grad():
            comp_imgs = torch.stack(
                [img.detach() for img in component_imgs], dim=1
            )  # (B, N_comp, C, H, W)

            pixel_mse   = (comp_imgs - full_img.unsqueeze(1)).pow(2).mean(dim=2)
            pseudo_mask = pixel_mse.argmin(dim=1).long()  # (B, H, W)

        return pseudo_mask

    def forward(
        self,
        seg_logits:     torch.Tensor,    # (B, N_comp, H, W)  raw logits
        component_imgs: list,            # N_comp x (B, C, H, W)
        full_img:       torch.Tensor,    # (B, C, H, W)
    ) -> torch.Tensor:
        pseudo_mask = self.get_pseudo_mask(component_imgs, full_img)
        loss = F.cross_entropy(seg_logits, pseudo_mask)
        return self.weight * loss
