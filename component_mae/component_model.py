import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MAEEncoder
from .decoder import MAEDecoder


class ComponentMAE(nn.Module):
    """
    One shared MAEEncoder  +  N independent shallow MAEDecoders.

    Each MAEDecoder owns its own ``self.mask_token`` nn.Parameter —
    the component-specific prior that replaces the SpatialVectorizer
    tile codebook from VectorizedUNet.

    Interface is intentionally kept compatible with VectorizedUNet:
        forward(x, seg_probs)  → (output_list, mask)
        output_list[-1]        → same shape as before

    When seg_probs is None  : output_list[0] shape (B, N, C, H, W)
    When seg_probs provided : output_list[0] shape (B, C, H, W)
    mask                    : (B, num_patches)  1 = masked  0 = visible
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
                 number_of_components: int   = 4):
        super().__init__()

        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

        self.number_of_components = number_of_components
        self.patch_size           = patch_size
        self.in_channels          = in_channels
        self.img_size             = img_size
        self.num_patches          = (img_size // patch_size) ** 2

        # ── Shared encoder (patch embed + pos embed + masking + ViT blocks) ──
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
        # Instantiating N separate MAEDecoder objects gives each its OWN
        # self.mask_token parameter automatically — no extra bookkeeping needed.
        # Those mask tokens are the per-component prior (what the model fills
        # in for masked patches of component i).
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

    # ── Patch utilities (mirrored from MAEReconstructionLoss for convenience) ─

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
            seg_probs : (B, N, H, W)  softmax probabilities from UNet.
                        Pass None for independent (unblended) mode.

        Returns:
            output_list : list with one element — mirrors the multi-scale list
                          interface of VectorizedUNet so ``[-1]`` still works.
                          elem shape: (B, C, H, W)     when seg_probs given
                                      (B, N, C, H, W)  when seg_probs is None
            mask        : (B, num_patches)  1 = masked position
        """

        # ── 1. Shared encode ──────────────────────────────────────────────────
        latent, mask, ids_restore = self.encoder(x)
        # latent      : (B, N_visible + 1, encoder_embed_dim)  — +1 for CLS
        # mask        : (B, num_patches)   binary
        # ids_restore : (B, num_patches)   inverse shuffle permutation

        # ── 2. N independent decodes ──────────────────────────────────────────
        # Every decoder sees the SAME latent but uses its OWN mask tokens,
        # so each specialises in reconstructing a different component.
        component_imgs = []
        for decoder in self.decoders:
            pred = decoder(latent, ids_restore)   # (B, num_patches, patch_dim)
            img  = self.unpatchify(pred)           # (B, C, H, W)
            img  = torch.tanh(img)                # output in [-1, 1]
            component_imgs.append(img)

        # (B, N, C, H, W)
        independent_recs = torch.stack(component_imgs, dim=1)

        if seg_probs is None:
            return [independent_recs], mask

        # ── 3. Compose by segmentation probabilities ──────────────────────────
        # Each pixel is the weighted sum of component reconstructions.
        # seg_probs: (B, N, H, W) → (B, N, 1, H, W) to broadcast over C
        seg_expanded = seg_probs.unsqueeze(2)
        final_rec    = (independent_recs * seg_expanded).sum(dim=1)  # (B, C, H, W)

        return [final_rec], mask

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
