"""
Pre-defined MAE model configurations matching Table 1 of He et al., 2021.

"Masked Autoencoders Are Scalable Vision Learners"
arXiv: https://arxiv.org/abs/2111.06377

Encoder specifications (Depth / Dim / Heads):
  ViT-Base  (B):  12 / 768  / 12
  ViT-Large (L):  24 / 1024 / 16
  ViT-Huge  (H):  32 / 1280 / 16

Decoder is shared across all variants: 8 / 512 / 16
"""

from .mae import MAE


def mae_vit_base_patch16(img_size: int = 224, **kwargs) -> MAE:
    """MAE-Base — ViT-B/16 encoder.  Standard pre-training configuration."""
    return MAE(
        img_size=img_size,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )


def mae_vit_large_patch16(img_size: int = 224, **kwargs) -> MAE:
    """MAE-Large — ViT-L/16 encoder."""
    return MAE(
        img_size=img_size,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )


def mae_vit_huge_patch14(img_size: int = 224, **kwargs) -> MAE:
    """MAE-Huge — ViT-H/14 encoder."""
    return MAE(
        img_size=img_size,
        patch_size=14,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )
