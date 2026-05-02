import torch
import torch.nn as nn

from .encoder import MAEEncoder
from .decoder import MAEDecoder


class ComponentMAE(nn.Module):
    def __init__(
        self,
        img_size:             int   = 256,
        patch_size:           int   = 16,
        in_channels:          int   = 3,
        encoder_embed_dim:    int   = 768,
        encoder_depth:        int   = 12,
        encoder_num_heads:    int   = 12,
        decoder_embed_dim:    int   = 512,
        decoder_depth:        int   = 4,
        decoder_num_heads:    int   = 16,
        mlp_ratio:            float = 4.0,
        mask_ratio:           float = 0.5,
        dropout:              float = 0.0,
        number_of_components: int   = 4,
    ):
        super().__init__()

        assert img_size % patch_size == 0
        assert 0.0 < mask_ratio < 1.0

        self.number_of_components = number_of_components
        self.patch_size           = patch_size
        self.in_channels          = in_channels
        self.img_size             = img_size
        self.num_patches          = (img_size // patch_size) ** 2
        self.mask_ratio           = mask_ratio

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

    # ── patch utilities ───────────────────────────────────────────────────────

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * p * p)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        B, N, _ = x.shape
        h = w = int(N ** 0.5)
        x = x.reshape(B, h, w, self.in_channels, p, p)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(
            B, self.in_channels, h * p, w * p
        )

    # ── masking helpers ───────────────────────────────────────────────────────

    def _apply_mask(
        self,
        tokens: torch.Tensor,  # (B, N, D)  post pos-embed
        mask:   torch.Tensor,  # (B, N)  1 = masked, 0 = visible
    ):
        """
        Extract visible tokens and build ids_restore from a given binary mask.
        Assumes uniform number of visible patches across the batch.
        """
        B, N, D = tokens.shape
        num_keep = int((mask == 0).sum(dim=1)[0].item())

        # argsort puts 0s (visible) before 1s (masked)
        ids_shuffle = torch.argsort(mask.float(), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        tokens_visible = torch.gather(
            tokens, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )
        return tokens_visible, ids_restore

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:           torch.Tensor,
        mask:        torch.Tensor = None,
        return_attn: bool         = False,
    ):
        """
        Parameters
        ----------
        x           : (B, C, H, W)
        mask        : (B, N)  optional — 1 = masked, 0 = visible.
                      If None, a random mask is generated using self.mask_ratio.
        return_attn : if True, also return per-decoder head-averaged attention
                      from the last decoder block (used by CrossMaskConsistencyLoss).

        Returns
        -------
        component_preds : (B, N_comp, N, patch_dim)
        mask            : (B, N)
        attn_weights    : list[N_comp] of (B, N, N)  — only when return_attn=True
        """
        tokens = self.encoder.patch_embed(x)
        tokens = tokens + self.encoder.pos_embed

        if mask is None:
            tokens_vis, mask, ids_restore = self.encoder.masking(tokens)
        else:
            tokens_vis, ids_restore = self._apply_mask(tokens, mask)

        for block in self.encoder.blocks:
            tokens_vis = block(tokens_vis)
        latent = self.encoder.norm(tokens_vis)

        component_preds = []
        attn_weights    = []
        for decoder in self.decoders:
            if return_attn:
                pred, attn = decoder(latent, ids_restore, return_attn=True)
                attn_weights.append(attn.detach())   # (B, N, N), head-averaged
            else:
                pred = decoder(latent, ids_restore)
            component_preds.append(pred)

        component_preds = torch.stack(component_preds, dim=1)  # (B, N_comp, N, patch_dim)

        if return_attn:
            return component_preds, mask, attn_weights
        return component_preds, mask

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
