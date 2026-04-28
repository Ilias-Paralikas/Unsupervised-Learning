import torch
import torch.nn as nn

from .modules import (
    TransformerBlock,
    build_2d_sincos_pos_embed,
)


class MAEDecoder(nn.Module):
    """
    MAE Decoder: lightweight transformer that reconstructs *all* N patches.

    The decoder is much shallower than the encoder (8 vs. 12 blocks in ViT-B)
    and uses a narrower embedding dimension (512 vs. 768), making pre-training
    computationally efficient.

    Pipeline:
        1. Project encoder embedding (D_enc) → decoder embedding (D_dec)
        2. Insert learnable [MASK] tokens at masked positions
        3. Un-shuffle the sequence back to the original patch order
        4. Add fixed 2-D sin-cos positional embeddings to all tokens
        5. L × TransformerBlock
        6. LayerNorm
        7. Linear prediction head → C × P × P pixel values per patch

    Args:
        num_patches          (int):   Total number of patches N = (H/P)².
        encoder_embed_dim    (int):   Encoder output dimension (for the projection).
        decoder_embed_dim    (int):   Decoder internal dimension.
        depth                (int):   Number of transformer blocks.
        num_heads            (int):   Attention heads per block.
        mlp_ratio            (float): FFN hidden-dim expansion factor.
        patch_size           (int):   Patch height = width P.
        in_channels          (int):   Input image channels C.
        dropout              (float): Dropout in attention / FFN.
    """

    def __init__(
        self,
        num_patches:          int,
        encoder_embed_dim:    int   = 768,
        decoder_embed_dim:    int   = 512,
        depth:                int   = 8,
        num_heads:            int   = 16,
        mlp_ratio:            float = 4.0,
        patch_size:           int   = 16,
        in_channels:          int   = 3,
        dropout:              float = 0.0,
    ):
        super().__init__()
        self.decoder_embed_dim = decoder_embed_dim
        self.num_patches       = num_patches
        self.grid_size         = int(num_patches ** 0.5)

        # Number of pixel values the decoder predicts per patch
        self.patch_dim = in_channels * patch_size * patch_size

        # ── Encoder-to-decoder dimension projection ───────────────────────────
        self.proj = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # ── Learnable [MASK] token ────────────────────────────────────────────
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # ── Positional embedding (fixed, not learned) ─────────────────────────
        # Shape: (1, N+1, D_dec).  CLS position is index 0.
        pos_embed = build_2d_sincos_pos_embed(
            embed_dim=decoder_embed_dim,
            grid_size=self.grid_size,
            cls_token=True,
        )
        self.register_buffer("pos_embed", pos_embed)

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=decoder_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)

        # ── Pixel prediction head ─────────────────────────────────────────────
        self.pred_head = nn.Linear(decoder_embed_dim, self.patch_dim, bias=True)

        self._init_weights()

    # ── weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _restore_full_sequence(
        self,
        latent:      torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inserts [MASK] tokens at the masked positions and restores the original
        patch ordering produced by the encoder's RandomMasking.

        Args:
            latent:      (B, N_vis+1, D_dec) — projected visible tokens + CLS
            ids_restore: (B, N)              — inverse permutation from the encoder

        Returns:
            tokens: (B, N+1, D_dec) — full sequence in original patch order,
                                       with CLS token at position 0
        """
        B       = latent.shape[0]
        N       = ids_restore.shape[1]          # total patches
        N_vis   = latent.shape[1] - 1           # subtract CLS
        N_mask  = N - N_vis
        D       = self.decoder_embed_dim

        # Expand the single mask token across all masked positions
        mask_tokens = self.mask_token.expand(B, N_mask, -1)            # (B, N_mask, D)

        # Concatenate in the shuffled order: [visible (no CLS) | mask tokens]
        tokens_no_cls = latent[:, 1:, :]                               # (B, N_vis, D)
        tokens_full   = torch.cat([tokens_no_cls, mask_tokens], dim=1) # (B, N, D)

        # Un-shuffle back to original patch order
        tokens_full = torch.gather(
            tokens_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
        )  # (B, N, D)

        # Re-prepend the CLS token
        cls         = latent[:, :1, :]                                  # (B, 1, D)
        tokens_full = torch.cat([cls, tokens_full], dim=1)              # (B, N+1, D)

        return tokens_full

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        latent:      torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latent:      (B, N_vis+1, D_enc) — encoder output
            ids_restore: (B, N)              — from MAEEncoder.forward()

        Returns:
            pred: (B, N, patch_dim) — predicted pixel values for *all* patches
        """
        # 1. Project encoder dim → decoder dim
        latent = self.proj(latent)                                      # (B, N_vis+1, D_dec)

        # 2. Insert [MASK] tokens and restore original ordering
        tokens = self._restore_full_sequence(latent, ids_restore)       # (B, N+1, D_dec)

        # 3. Add decoder positional embeddings to all tokens (including CLS)
        tokens = tokens + self.pos_embed                                # (B, N+1, D_dec)

        # 4. Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # 5. LayerNorm
        tokens = self.norm(tokens)                                      # (B, N+1, D_dec)

        # 6. Pixel prediction (CLS token is discarded)
        pred = self.pred_head(tokens[:, 1:, :])                        # (B, N, patch_dim)

        return pred
