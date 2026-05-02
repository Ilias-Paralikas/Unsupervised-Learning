import torch
import torch.nn as nn

from .modules import TransformerBlock, build_2d_sincos_pos_embed


class MAEDecoder(nn.Module):
    """
    MAE Decoder — no CLS token.

    Normal forward
    --------------
    pred = decoder(latent, ids_restore)
        → (B, N, patch_dim)

    Attention-tracking forward
    --------------------------
    pred, attn_weights = decoder(latent, ids_restore, return_attn=True)
        attn_weights : (B, N_full, N_full)  — averaged over heads,
                       from the LAST transformer block only.

    This is used by ComponentMAE to build the cross-mask consistency
    loss weight: rows corresponding to originally-masked patches tell
    us which visible patches each component "looked at" to reconstruct
    those positions.
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
        self.patch_dim         = in_channels * patch_size * patch_size

        self.proj       = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # No CLS → pos embed shape (1, N, D)
        pos_embed = build_2d_sincos_pos_embed(
            embed_dim=decoder_embed_dim,
            grid_size=self.grid_size,
            cls_token=False,
        )
        self.register_buffer("pos_embed", pos_embed)

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.pred_head = nn.Linear(decoder_embed_dim, self.patch_dim, bias=True)

        self._init_weights()

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

    def _restore_full_sequence(
        self,
        latent:      torch.Tensor,   # (B, N_vis, D_dec)  — already projected
        ids_restore: torch.Tensor,   # (B, N)
    ) -> torch.Tensor:
        """
        Fills masked positions with self.mask_token and un-shuffles to
        the original patch order.  No CLS token.

        Returns (B, N, D_dec).
        """
        B, N_vis, D = latent.shape
        N            = ids_restore.shape[1]
        N_mask       = N - N_vis

        mask_tokens  = self.mask_token.expand(B, N_mask, -1)          # (B, N_mask, D)
        tokens_full  = torch.cat([latent, mask_tokens], dim=1)         # (B, N, D)

        # Un-shuffle
        tokens_full  = torch.gather(
            tokens_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
        )  # (B, N, D)

        return tokens_full

    def forward(
        self,
        latent:      torch.Tensor,
        ids_restore: torch.Tensor,
        return_attn: bool = False,
    ):
        """
        Parameters
        ----------
        latent      : (B, N_vis, D_enc)
        ids_restore : (B, N)
        return_attn : if True, also return head-averaged attention from
                      the last transformer block, shape (B, N, N).

        Returns
        -------
        pred         : (B, N, patch_dim)
        attn_weights : (B, N, N)   — only when return_attn=True
        """
        latent = self.proj(latent)                                  # (B, N_vis, D_dec)
        tokens = self._restore_full_sequence(latent, ids_restore)   # (B, N, D_dec)
        tokens = tokens + self.pos_embed                            # add pos enc

        last_attn = None
        for i, block in enumerate(self.blocks):
            is_last = (i == len(self.blocks) - 1)
            if return_attn and is_last:
                tokens, last_attn = block(tokens, return_attn=True) # (B, H, N, N)
            else:
                tokens = block(tokens)

        tokens = self.norm(tokens)                                  # (B, N, D_dec)
        pred   = self.pred_head(tokens)                             # (B, N, patch_dim)

        if return_attn:
            # Average over heads → (B, N, N)
            attn_weights = last_attn.mean(dim=1)
            return pred, attn_weights

        return pred
