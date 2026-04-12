import torch
import torch.nn as nn

class GradientPenaltyLoss(nn.Module):
    def __init__(self, weight=10.0):
        """
        Args:
            gp_loss_weight (float): The weight factor for the gradient penalty. 
                                   Standard WGAN-GP papers use 10.0.
        """
        super().__init__()
        self.lambda_weight = weight

    def forward(self, critic, real, fake):
        batch_size = real.size(0)
        device = real.device
        
        # 1. Create alpha and use expand_as (more memory-efficient than repeat)
        alpha = torch.rand((batch_size, 1, 1, 1), device=device)
        alpha = alpha.expand_as(real)
        
        # 2. Compute interpolated images and explicitly require gradients
        interpolated_images = alpha * real + (1 - alpha) * fake
        interpolated_images.requires_grad_(True)

        # 3. Calculate critic scores
        mixed_scores = critic(interpolated_images)

        # 4. Take the gradient of the scores with respect to the images
        grad_outputs = torch.ones_like(mixed_scores, device=device)
        gradients = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # 5. Flatten and compute the L2 norm
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # 6. Calculate penalty
        gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
        
        return gradient_penalty * self.lambda_weight

