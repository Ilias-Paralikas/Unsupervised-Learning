import torch
import torch.nn as nn

from .modules import (
    PatchEmbed,
    RandomMasking,
    TransformerBlock,
    build_2d_sincos_pos_embed,
)


class MAEEncoder(nn.Module):
    """
    MAE Encoder: a ViT that processes *only* the visible (unmasked) patches.

    The encoder is intentionally unaware of the masked patches — they are
    never passed through the attention layers.  This asymmetry is the key
    efficiency gain of the MAE architecture.

    Pipeline:
        1. PatchEmbed: image → N patch tokens
        2. Add fixed 2-D sin-cos positional embeddings
        3. RandomMasking: keep N_vis = N × (1 − mask_ratio) patches
        4. Prepend CLS token (with its own positional embedding)
        5. L × TransformerBlock
        6. LayerNorm

    Args:
        img_size         (int):   Input image height = width.
        patch_size       (int):   Patch height = width.
        in_channels      (int):   Input image channels.
        embed_dim        (int):   Encoder token embedding dimension  D.
        depth            (int):   Number of transformer blocks  L.
        num_heads        (int):   Attention heads per block.
        mlp_ratio        (float): FFN hidden-dim expansion factor.
        mask_ratio       (float): Fraction of patches to mask.  Default 0.75.
        dropout          (float): Dropout in attention / FFN.

    Input:  (B, C, H, W)
    Output:
        latent:      (B, N_vis+1, D)  — encoded visible tokens + CLS token
        mask:        (B, N)           — binary mask (1 = masked, 0 = visible)
        ids_restore: (B, N)           — inverse permutation to restore patch order
    """

    def __init__(
        self,
        img_size:    int   = 224,
        patch_size:  int   = 16,
        in_channels: int   = 3,
        embed_dim:   int   = 768,
        depth:       int   = 12,
        num_heads:   int   = 12,
        mlp_ratio:   float = 4.0,
        mask_ratio:  float = 0.75,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        # ── Patch embedding ───────────────────────────────────────────────────
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # ── CLS token ─────────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Positional embedding (fixed, not learned) ─────────────────────────
        # Shape: (1, N+1, D).  Stored as a buffer so it moves with the model.
        pos_embed = build_2d_sincos_pos_embed(
            embed_dim=embed_dim,
            grid_size=self.grid_size,
            cls_token=True,
        )
        self.register_buffer("pos_embed", pos_embed)

        # ── Random masking ────────────────────────────────────────────────────
        self.masking = RandomMasking(mask_ratio=mask_ratio)

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self._init_weights()

    # ── weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.view(m.weight.shape[0], -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) — input images

        Returns:
            latent:      (B, N_vis+1, D)
            mask:        (B, N)
            ids_restore: (B, N)
        """
        # 1. Patch embedding
        tokens = self.patch_embed(x)                              # (B, N, D)

        # 2. Add patch positional embeddings (position 0 is CLS; skip it here)
        tokens = tokens + self.pos_embed[:, 1:, :]                # (B, N, D)

        # 3. Random masking — returns only the visible subset
        tokens_visible, mask, ids_restore = self.masking(tokens)  # (B, N_vis, D)

        # 4. Prepend CLS token with its positional embedding
        cls = self.cls_token.expand(x.shape[0], -1, -1)           # (B, 1, D)
        cls = cls + self.pos_embed[:, :1, :]
        tokens_visible = torch.cat([cls, tokens_visible], dim=1)   # (B, N_vis+1, D)

        # 5. Transformer blocks
        for block in self.blocks:
            tokens_visible = block(tokens_visible)

        # 6. LayerNorm
        latent = self.norm(tokens_visible)                         # (B, N_vis+1, D)

        return latent, mask, ids_restore
