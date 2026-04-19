# --- DCGAN/modules/blocks/normalizations/spade.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        """
        Args:
            norm_nc: Number of channels in the feature map (C)
            label_nc: Number of channels in the routed semantic map (n_components * vectordim)
        """
        super().__init__()
        
        # InstanceNorm to wipe out the old style
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # SPADE Routing network - STRICTLY 1x1 convolutions
        hidden_nc = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        # C-channel output to re-inject the specific style properly
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=1, padding=0)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, kernel_size=1, padding=0)

    def forward(self, x, routed_styles):
        # 1. Wipe the slate clean
        normalized = self.param_free_norm(x)

        # 2. Resize routed_styles to match the current generator resolution
        if routed_styles.shape[2:] != x.shape[2:]:
            routed_styles = F.interpolate(routed_styles, size=x.shape[2:], mode='nearest')

        # 3. Generate adaptive parameters
        actv = self.mlp_shared(routed_styles)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # 4. Modulate
        return normalized * (1 + gamma) + beta