import torch
import torch.nn as nn


class GradientPenaltyLoss(nn.Module):
    """
    WGAN-GP gradient penalty.

    Works with the single-scale Discriminator that takes (B, C, H, W)
    directly — no multi-resolution list needed.
    """

    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight

    def forward(self,
                discriminator: nn.Module,
                real:  torch.Tensor,
                fake:  torch.Tensor) -> torch.Tensor:
        """
        Args:
            discriminator : the Discriminator module
            real          : (B, C, H, W) real images
            fake          : (B, C, H, W) generated images  ← pass detached
        Returns:
            scalar gradient penalty loss
        """
        B      = real.size(0)
        device = real.device

        # Random linear interpolation between real and fake
        alpha  = torch.rand(B, 1, 1, 1, device=device).expand_as(real)
        interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

        # Discriminator score on interpolated samples
        d_interp = discriminator(interp)

        # Gradients w.r.t. interpolated inputs
        gradients = torch.autograd.grad(
            outputs      = d_interp,
            inputs       = interp,
            grad_outputs = torch.ones_like(d_interp),
            create_graph = True,
            retain_graph = True,
        )[0]                                          # (B, C, H, W)

        gradients = gradients.reshape(B, -1)          # (B, C*H*W)
        grad_norm = gradients.norm(2, dim=1)          # (B,)

        penalty   = ((grad_norm - 1.0) ** 2).mean()
        return self.weight * penalty
