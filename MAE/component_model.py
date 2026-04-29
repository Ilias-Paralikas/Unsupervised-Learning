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
        return x
        residual = x          # ← save the composed image BEFORE processing
        h = x
        for layer in self.layers:
            h = layer(h)
        delta = self.final_conv(h)   # near-zero at init due to zero-init weights
        # residual + delta ≈ residual at epoch 0  (identity behaviour)
        return torch.clamp(residual + delta, -1.0, 1.0)


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

   
   # ── Mask utility ──────────────────────────────────────────────────────────

    def expand_mask_to_pixels(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Upsample patch-level binary mask to pixel space.

        Args:
            mask : (B, N)  1 = masked, 0 = visible
        Returns:
            (B, 1, H, W)  nearest-neighbor fill — each patch block is uniform
        """
        B    = mask.shape[0]
        grid = self.img_size // self.patch_size
        mask_2d = mask.reshape(B, 1, grid, grid).float()
        return F.interpolate(mask_2d, scale_factor=self.patch_size, mode='nearest')

# ── Forward ───────────────────────────────────────────────────────────────

    def forward(self,
            x:                 torch.Tensor,
            seg_probs:         torch.Tensor = None,
            return_components: bool         = False):

        # ── 1. Encode once ────────────────────────────────────────────────────
        latent, mask, ids_restore = self.encoder(x)

        # ── 2. N decoders, once ───────────────────────────────────────────────
        component_imgs = []
        for decoder in self.decoders:
            pred = decoder(latent, ids_restore)
            img  = self.unpatchify(pred)
            component_imgs.append(torch.tanh(img))

        independent_recs = torch.stack(component_imgs, dim=1)  # (B, N, C, H, W)

        # ── 3. Independent-only mode ──────────────────────────────────────────
        if seg_probs is None:
            return [independent_recs], mask

        # ── 4. Compose ────────────────────────────────────────────────────────
        seg_expanded = seg_probs.unsqueeze(2)                          # (B, N, 1, H, W)
        composed     = (independent_recs * seg_expanded).sum(dim=1)    # (B, C, H, W)

        # ── 5. Stitch original visible patches back in ────────────────────────
        # mask_pixels: 1 = was masked (decoder filled this), 0 = was visible (use original)
        mask_pixels = self.expand_mask_to_pixels(mask)                 # (B, 1, H, W)
        composited  = mask_pixels * composed + (1.0 - mask_pixels) * x # (B, C, H, W)

        # ── 6. Heal seams between real and generated regions ──────────────────
        final_rec = self.seam_smoother(composited)

        # ── 7. Optionally expose components ───────────────────────────────────
        if return_components:
            return final_rec, independent_recs, mask

        return final_rec, mask
        # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
