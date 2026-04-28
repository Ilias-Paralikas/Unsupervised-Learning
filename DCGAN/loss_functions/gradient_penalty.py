import torch
import torch.nn as nn

class GradientPenalty(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.gradient_penalty_weight = weight

    def forward(self, critic, real_images, fake_images):
        """
        Calculates the WGAN Gradient Penalty for a Multi-Scale GAN.
        
        Args:
            critic (nn.Module): The discriminator network.
            real_images (list of Tensors): List of real images at different scales.
            fake_images (list of Tensors): List of fake images at different scales.
            
        Returns:
            torch.Tensor: The computed gradient penalty scalar.
        """
        batch_size = real_images[0].shape[0]
        device = real_images[0].device
        
        # 1. Create a single random interpolation weight for the batch
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # 2. Interpolate between real and fake at every scale
        interpolated_images = []
        for real_img, fake_img in zip(real_images, fake_images):
            mixed = (alpha * real_img + (1 - alpha) * fake_img).requires_grad_(True)
            interpolated_images.append(mixed)
            
        # 3. Pass the mixed images through the critic
        mixed_scores = critic(interpolated_images)
        
        # 4. Calculate the gradients of the scores w.r.t the mixed images
        gradients = torch.autograd.grad(
            outputs=mixed_scores,
            inputs=interpolated_images,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        
        # 5. Compute the penalty across ALL scales combined
        grad_flat_list = []
        for grad in gradients:
            # Flatten each scale: [batch_size, channels, H, W] -> [batch_size, -1]
            grad_flat_list.append(grad.reshape(batch_size, -1))
            
        # Concatenate all flattened gradients along the feature dimension (dim=1)
        grad_concat = torch.cat(grad_flat_list, dim=1)
        
        # 6. Calculate L2 norm safely (add epsilon inside the square root to prevent NaN)
        # This prevents autograd from crashing if the gradient is exactly zero
        grad_norm = torch.sqrt(torch.sum(grad_concat ** 2, dim=1) + 1e-8)
        
        # 7. Penalize deviation from 1.0
        gradient_penalty = torch.mean((grad_norm - 1.0) ** 2)
            
        return gradient_penalty