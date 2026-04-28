import torch
import torch.nn as nn

from .encoder import MAEEncoder
from .decoder import MAEDecoder


class MAE(nn.Module):
    """
    Masked Autoencoder — He et al., 2021.

    "Masked Autoencoders Are Scalable Vision Learners"
    arXiv: https://arxiv.org/abs/2111.06377

    The model pairs an asymmetric encoder–decoder:
      - Encoder: full ViT that sees only the visible (~25%) patches.
      - Decoder: lighter ViT that receives encoded visible tokens + learnable
        [MASK] tokens and reconstructs the pixel values of all patches.

    Pre-training usage:
        model = MAE(...)
        pred, mask = model(imgs)
        loss = criterion(pred, imgs, mask)

    Feature extraction / fine-tuning:
        cls_feat, patch_feats = model.encode(imgs)   # no masking

    Args:
        img_size          (int):   Input image height = width.
        patch_size        (int):   Patch height = width.
        in_channels       (int):   Image channels.
        encoder_embed_dim (int):   Encoder token dimension.
        encoder_depth     (int):   Encoder transformer depth.
        encoder_num_heads (int):   Encoder attention heads.
        decoder_embed_dim (int):   Decoder token dimension.
        decoder_depth     (int):   Decoder transformer depth.
        decoder_num_heads (int):   Decoder attention heads.
        mlp_ratio         (float): FFN expansion factor for both encoder/decoder.
        mask_ratio        (float): Fraction of patches masked during pre-training.
        dropout           (float): Dropout probability.
    """

    def __init__(
        self,
        img_size:          int   = 224,
        patch_size:        int   = 16,
        in_channels:       int   = 3,
        # Encoder
        encoder_embed_dim: int   = 768,
        encoder_depth:     int   = 12,
        encoder_num_heads: int   = 12,
        # Decoder
        decoder_embed_dim: int   = 512,
        decoder_depth:     int   = 8,
        decoder_num_heads: int   = 16,
        # Shared
        mlp_ratio:         float = 4.0,
        mask_ratio:        float = 0.75,
        dropout:           float = 0.0,
    ):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.mask_ratio  = mask_ratio

        num_patches = (img_size // patch_size) ** 2

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            mask_ratio=mask_ratio,
            dropout=dropout,
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = MAEDecoder(
            num_patches=num_patches,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            in_channels=in_channels,
            dropout=dropout,
        )

    # ── pre-training forward ──────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Full MAE forward pass for pre-training.

        Args:
            x: (B, C, H, W) — input images

        Returns:
            pred: (B, N, patch_dim) — predicted pixel values for *all* patches
            mask: (B, N)            — binary mask (1 = masked, used by the loss)
        """
        latent, mask, ids_restore = self.encoder(x)
        pred = self.decoder(latent, ids_restore)
        return pred, mask

    # ── feature extraction ────────────────────────────────────────────────────

    def encode(
        self,
        x:          torch.Tensor,
        mask_ratio: float = 0.0,
    ):
        """
        Encodes images for downstream tasks.

        By default (mask_ratio=0.0) all patches are visible so the encoder
        produces a complete, unmasked sequence — suitable for classification
        or dense prediction fine-tuning.

        Args:
            x:          (B, C, H, W)
            mask_ratio: float — masking fraction override.  0.0 = no masking.

        Returns:
            cls_token:    (B, D_enc) — global [CLS] representation
            patch_tokens: (B, N, D_enc) — per-patch representations
        """
        original_ratio = self.encoder.masking.mask_ratio
        self.encoder.masking.mask_ratio = mask_ratio

        latent, _, _ = self.encoder(x)

        self.encoder.masking.mask_ratio = original_ratio

        cls_token    = latent[:, 0,  :]   # (B, D)
        patch_tokens = latent[:, 1:, :]   # (B, N, D)
        return cls_token, patch_tokens

    # ── reconstruction utility ────────────────────────────────────────────────

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Converts decoder predictions back to image space.

        Args:
            patches: (B, N, patch_dim)

        Returns:
            imgs: (B, C, H, W)
        """
        P = self.patch_size
        C = self.in_channels
        H = W = self.img_size
        h = w = H // P

        # (B, N, C*P*P) → (B, h, w, P, P, C)
        x = patches.reshape(patches.shape[0], h, w, P, P, C)
        # (B, h, w, P, P, C) → (B, C, h, P, w, P)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        # (B, C, h, P, w, P) → (B, C, H, W)
        imgs = x.reshape(patches.shape[0], C, H, W)
        return imgs

    # ── parameter counting ────────────────────────────────────────────────────

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Returns the total number of (trainable) parameters."""
        params = self.parameters() if not only_trainable else filter(
            lambda p: p.requires_grad, self.parameters()
        )
        return sum(p.numel() for p in params)
