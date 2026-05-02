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
    MAE Encoder — no CLS token.

    Pipeline
    --------
    1. PatchEmbed        (B, C, H, W)  →  (B, N, D)
    2. Add fixed 2-D sin-cos pos embeddings
    3. RandomMasking     keep N_vis = N × (1 - mask_ratio) patches
    4. L × TransformerBlock on visible tokens only

    Returns
    -------
    latent      : (B, N_vis, D)
    mask        : (B, N)          1 = masked, 0 = visible
    ids_restore : (B, N)          inverse permutation
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

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

        # No CLS token — pos embed covers patches only, shape (1, N, D)
        pos_embed = build_2d_sincos_pos_embed(
            embed_dim=embed_dim,
            grid_size=self.grid_size,
            cls_token=False,
        )
        self.register_buffer("pos_embed", pos_embed)

        self.masking = RandomMasking(mask_ratio=mask_ratio)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self._init_weights()

    def _init_weights(self):
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

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        latent      : (B, N_vis, D)
        mask        : (B, N)
        ids_restore : (B, N)
        """
        tokens = self.patch_embed(x)                              # (B, N, D)
        tokens = tokens + self.pos_embed                          # fixed pos enc

        tokens_visible, mask, ids_restore = self.masking(tokens) # (B, N_vis, D)

        for block in self.blocks:
            tokens_visible = block(tokens_visible)

        latent = self.norm(tokens_visible)                        # (B, N_vis, D)
        return latent, mask, ids_restore
