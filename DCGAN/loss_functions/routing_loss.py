import torch
import torch.nn as nn
import torch.nn.functional as F
class SoftRoutingLoss(nn.Module):
    def __init__(self, temperature=0.1, weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def get_targets(self, reconstructions, target_img):
        """
        reconstructions: (B, N, C, H, W) 
        target_img: (B, C, H, W) 
        seg_logits: (B, N, H, W)
        """
        with torch.no_grad():
            target_expanded = target_img.unsqueeze(1) # (B, 1, C, H, W)
            
            # Mean Squared Error per pixel -> (B, N, H, W)
            pixel_errors = (reconstructions - target_expanded).pow(2).mean(dim=2)
            
            # Convert errors to target probabilities
            pseudo_targets = F.softmax(-pixel_errors / self.temperature, dim=1)
        
        return pseudo_targets
    def forward(self, reconstructions, target_img, seg_logits):
        """
        reconstructions: (B, N, C, H, W) 
        target_img: (B, C, H, W) 
        seg_logits: (B, N, H, W)
        """
        pseudo_targets= self.get_targets(reconstructions, target_img)
        # --- EXPLICIT SOFT CROSS ENTROPY ---
        # 1. Convert logits to log-probabilities
        log_probs = F.log_softmax(seg_logits, dim=1)
        
        # 2. Cross entropy math: -sum(target_prob * log_prob)
        # Sum across the components/classes dimension (dim=1)
        pixel_loss = -torch.sum(pseudo_targets * log_probs, dim=1) # (B, H, W)
        
        # 3. Average over the batch and spatial dimensions
        loss = torch.mean(pixel_loss)
        
        return self.weight * loss
    
        