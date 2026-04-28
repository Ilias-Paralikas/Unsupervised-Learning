import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MAEEncoder
from .decoder import MAEDecoder
from .modules.patch_embed import PatchEmbed

# reuse EQLRConv2d so the smoother follows the same weight conventions
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from DCGAN.modules.blocks.eq_lr_layers import EQLRConv2d


class SeamSmoother(nn.Module):
    """
    Two learnable 3x3 convs with a residual connection.
    Applied to the composed reconstruction after blending — heals the
    periodic 16-pixel patch-boundary seams before the discriminator sees
    the image.

    Zero-init on the second conv guarantees near-identity at epoch 0,
    so MSE / routing losses warm up without interference.
    """
    def __init__(self, depth: int = 1, channels: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth-1):
            self.layers.append(EQLRConv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.final_conv = EQLRConv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        # Zero-init → identity residual at the start of training
        nn.init.zeros_(self.final_conv.conv.weight)
        nn.init.zeros_(self.final_conv.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x)
        x= self.final_conv(x)
        x = torch.tanh(x)
        return x


class ComponentMAE(nn.Module):
    """
    One shared MAEEncoder  +  N independent shallow MAEDecoders  +  SeamSmoother.

    Each MAEDecoder owns its own mask_token parameter — the per-component prior.

    forward(x, seg_probs=None) → (output_list, mask)

        seg_probs is None  : output_list[0] shape (B, N, C, H, W)
                             raw per-component reconstructions, NO smoother applied
                             (routing / dilated losses use these directly)

        seg_probs provided : output_list[0] shape (B, C, H, W)
                             composed + SeamSmoother applied
                             (MSE loss and discriminator use this)

        mask               : (B, num_patches)  1=masked  0=visible
    """

    def __init__(self,
                 img_size:             int   = 224,
                 patch_size:           int   = 16,
                 in_channels:          int   = 3,
                 encoder_embed_dim:    int   = 768,
                 encoder_depth:        int   = 12,
                 encoder_num_heads:    int   = 12,
                 decoder_embed_dim:    int   = 512,
                 decoder_depth:        int   = 4,
                 decoder_num_heads:    int   = 16,
                 mlp_ratio:            float = 4.0,
                 mask_ratio:           float = 0.75,
                 dropout:              float = 0.0,
                 number_of_components: int   = 4,
                 smoother_depth:       int   = 1):
        super().__init__()

        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

        self.number_of_components = number_of_components
        self.patch_size           = patch_size
        self.in_channels          = in_channels
        self.img_size             = img_size
        self.num_patches          = (img_size // patch_size) ** 2

        # ── Shared encoder ────────────────────────────────────────────────────
        self.encoder = MAEEncoder(
            img_size    = img_size,
            patch_size  = patch_size,
            in_channels = in_channels,
            embed_dim   = encoder_embed_dim,
            depth       = encoder_depth,
            num_heads   = encoder_num_heads,
            mlp_ratio   = mlp_ratio,
            dropout     = dropout,
            mask_ratio  = mask_ratio,
        )

        # ── N independent shallow decoders ────────────────────────────────────
        # Each instance gets its own self.mask_token (the component prior).
        self.decoders = nn.ModuleList([
            MAEDecoder(
                num_patches       = self.num_patches,
                encoder_embed_dim = encoder_embed_dim,
                decoder_embed_dim = decoder_embed_dim,
                depth             = decoder_depth,
                num_heads         = decoder_num_heads,
                mlp_ratio         = mlp_ratio,
                patch_size        = patch_size,
                in_channels       = in_channels,
                dropout           = dropout,
            )
            for _ in range(number_of_components)
        ])

        # ── Seam smoother ─────────────────────────────────────────────────────
        # Applied ONLY to the composed blended output (seg_probs path).
        # Raw per-component outputs are never smoothed so routing / dilated
        # losses receive clean per-component gradients.
        self.seam_smoother = SeamSmoother(channels=in_channels, depth=smoother_depth)

    # ── Patch utilities ───────────────────────────────────────────────────────

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, patch_size² × C)"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * p * p)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_size² × C) → (B, C, H, W)"""
        p = self.patch_size
        B, N, _ = x.shape
        h = w = int(N ** 0.5)
        x = x.reshape(B, h, w, self.in_channels, p, p)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(
            B, self.in_channels, h * p, w * p
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self,
                x:         torch.Tensor,
                seg_probs: torch.Tensor = None):
        """
        Args:
            x         : (B, C, H, W)
            seg_probs : (B, N, H, W) softmax probabilities from UNet, or None.

        Returns:
            output_list : one-element list — use [-1] to access the tensor.
                          (B, C, H, W)     seg_probs provided → smoothed blend
                          (B, N, C, H, W)  seg_probs is None  → raw components
            mask        : (B, num_patches)  1 = masked
        """

        # ── 1. Shared encode ──────────────────────────────────────────────────
        latent, mask, ids_restore = self.encoder(x)

        # ── 2. N independent decodes ──────────────────────────────────────────
        component_imgs = []
        for decoder in self.decoders:
            pred = decoder(latent, ids_restore)   # (B, num_patches, patch_dim)
            img  = self.unpatchify(pred)           # (B, C, H, W)
            img  = torch.tanh(img)
            component_imgs.append(img)

        # (B, N, C, H, W)
        independent_recs = torch.stack(component_imgs, dim=1)

        # Independent mode — return raw components, no smoothing
        if seg_probs is None:
            return [independent_recs], mask

        # ── 3. Compose ────────────────────────────────────────────────────────
        seg_expanded = seg_probs.unsqueeze(2)                              # (B, N, 1, H, W)
        composed     = (independent_recs * seg_expanded).sum(dim=1)       # (B, C, H, W)

        # ── 4. Heal patch-boundary seams ──────────────────────────────────────
        # SeamSmoother: two learnable 3x3 convs + residual.
        # Zero-initialised at construction → identity at epoch 0.
        # Gradient from the discriminator trains it to smooth seams specifically.
        final_rec = self.seam_smoother(composed)

        return final_rec, mask

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
