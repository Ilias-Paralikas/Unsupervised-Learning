# import torch
# import torch.nn as nn

# class GradientPenaltyLoss(nn.Module):
#     def __init__(self, weight=10.0):
#         """
#         Args:
#             gp_loss_weight (float): The weight factor for the gradient penalty. 
#                                    Standard WGAN-GP papers use 10.0.
#         """
#         super().__init__()
#         self.lambda_weight = weight

#     def forward(self, critic, real, fake):
#         batch_size = real.size(0)
#         device = real.device
        
#         # 1. Create alpha and use expand_as (more memory-efficient than repeat)
#         alpha = torch.rand((batch_size, 1, 1, 1), device=device)
#         alpha = alpha.expand_as(real)
        
#         # 2. Compute interpolated images and explicitly require gradients
#         interpolated_images = alpha * real + (1 - alpha) * fake
#         interpolated_images.requires_grad_(True)

#         # 3. Calculate critic scores
#         mixed_scores = critic(interpolated_images)

#         # 4. Take the gradient of the scores with respect to the images
#         grad_outputs = torch.ones_like(mixed_scores, device=device)
#         gradients = torch.autograd.grad(
#             inputs=interpolated_images,
#             outputs=mixed_scores,
#             grad_outputs=grad_outputs,
#             create_graph=True,
#             retain_graph=True,
#         )[0]

#         # 5. Flatten and compute the L2 norm
#         gradients = gradients.view(batch_size, -1)
#         gradient_norm = gradients.norm(2, dim=1)
        
#         # 6. Calculate penalty
#         gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
        
#         return gradient_penalty * self.lambda_weight


import torch
import torch.nn as nn

class GradientPenaltyLoss(nn.Module):
    def __init__(self, weight=10.0):
        super().__init__()
        self.lambda_weight = weight

    def forward(self, critic, real, fake, reconstructed):
        batch_size = real.size(0)
        device = real.device

        combined_targets = torch.cat([fake, reconstructed], dim=0)
        combined_real = torch.cat([real, real], dim=0)

        epsilon = torch.rand(2 * batch_size, 1, 1, 1, device=device)
        interpolated = (epsilon * combined_real + (1 - epsilon) * combined_targets).requires_grad_(True)

        critic_interpolates = critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_norms = torch.sqrt((gradients ** 2).sum(dim=1) + 1e-12)
        gradient_penalty = ((gradient_norms - 1.0) ** 2).mean()

        return self.lambda_weight * gradient_penalty