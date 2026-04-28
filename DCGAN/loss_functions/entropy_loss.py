import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.entropy_weight = weight
        
    def forward(self, logits):
        """
        logits: (B, N, H, W) raw output from the segmentation generator
        """
        # 1. Convert logits to log-probabilities (numerically stable)
        log_probs = F.log_softmax(logits, dim=1)
        
        # 2. Convert logits to probabilities 
        probs = F.softmax(logits, dim=1)
        
        # 3. Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=1) 
        
        # 4. Return the mean entropy scaled by the weight
        return self.entropy_weight * torch.mean(entropy)