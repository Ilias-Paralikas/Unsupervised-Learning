import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedReconstructionLoss(nn.Module):
    def __init__(self, dilation_kernel_size=3,weight=1.0):
        super().__init__()
        # A 3x3 kernel will expand the mask by 1 pixel in all directions
        # A 5x5 kernel will expand it by 2 pixels, etc.
        self.kernel_size = dilation_kernel_size
        self.padding = dilation_kernel_size // 2
        self.weight = weight


    
    def get_dilated_masks(self, seg_probs):
        """
        independent_reconstructions: (B, N, 3, H, W) - outputs when seg_probs=None
        target_img: (B, 3, H, W) - ground truth image
        seg_probs: (B, N, H, W) - soft segmentation masks
        """
        b, n, h, w = seg_probs.shape
        
        # 1. Get Hard Masks (Argmax)
        # Find the winning component for each pixel
        hard_indices = seg_probs.argmax(dim=1, keepdim=True) # (B, 1, H, W)
        
        # Convert to one-hot format
        hard_masks = torch.zeros_like(seg_probs).scatter_(1, hard_indices, 1.0) # (B, N, H, W)
        
        # 2. Expand/Dilate the Hard Masks
        # We can use Max Pooling with stride=1 to perform morphological dilation!
        # If any pixel in the 3x3 neighborhood is 1, the center becomes 1.
        dilated_masks = F.max_pool2d(
            hard_masks, 
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.padding
        ) # (B, N, H, W)
        
        # Add channel dimension so we can multiply with RGB images
        dilated_masks = dilated_masks.unsqueeze(2) # (B, N, 1, H, W)
        
        return dilated_masks

    def forward(self, independent_reconstructions, target_img, seg_probs):
        dilated_masks = self.get_dilated_masks(seg_probs)
        # 3. Calculate Localized MSE Loss
        # Expand target to match N components
        target_expanded = target_img.unsqueeze(1) # (B, 1, 3, H, W)
        
        # Calculate squared error
        squared_error = (independent_reconstructions - target_expanded).pow(2) # (B, N, 3, H, W)
        
        # Mask the error so we ONLY care about the dilated regions
        masked_error = squared_error * dilated_masks
        
        # 4. Average the error
        # We divide by the sum of the dilated masks so components with larger 
        # masks don't naturally have higher loss sums.
        # Add a tiny epsilon to prevent division by zero if a mask is completely empty.
        sum_error = masked_error.sum()
        pixels_in_masks = dilated_masks.sum() * 3.0 # *3 for RGB channels
        
        loss = sum_error / (pixels_in_masks + 1e-8)
        loss = self.weight * loss
        return loss
    