import torch
import torch.nn as nn


class RandomMasking(nn.Module):
    """
    Performs random per-sample patch masking by shuffling and dropping tokens.

    Uses an efficient "random noise → argsort" trick so that no explicit
    scatter/gather of indices is needed to select the kept patches.

    The `restore` helper re-inserts learnable mask tokens at masked positions
    and un-shuffles the sequence back to the original patch order, as required
    by the MAE decoder.

    Args:
        mask_ratio (float): Fraction of patches to mask.  Default 0.75.
    """

    def __init__(self, mask_ratio: float = 0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Randomly masks (mask_ratio × N) patches per sample.

        Args:
            x: (B, N, D) — patch embeddings with positional encoding added.

        Returns:
            x_visible    (B, N_vis, D):  Embeddings of the kept patches.
            mask         (B, N):         Binary mask — 1 = masked, 0 = visible.
            ids_restore  (B, N):         Indices that invert the shuffle permutation.
        """
        B, N, D = x.shape
        num_keep = int(N * (1.0 - self.mask_ratio))

        # Per-sample uniform noise → argsort gives an independent shuffle per row
        noise = torch.rand(B, N, device=x.device)                      # (B, N)
        ids_shuffle = torch.argsort(noise, dim=1)                      # (B, N)  ascending
        ids_restore = torch.argsort(ids_shuffle, dim=1)                # (B, N)  inverse

        # Keep the first num_keep entries of the shuffled sequence
        ids_keep = ids_shuffle[:, :num_keep]                           # (B, N_vis)
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, N_vis, D)

        # Binary mask (0 = visible, 1 = masked), aligned to original patch order
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0.0
        mask = torch.gather(mask, dim=1, index=ids_restore)            # (B, N)

        return x_visible, mask, ids_restore

    # ------------------------------------------------------------------
    def restore(
        self,
        x_visible: torch.Tensor,
        mask_token: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inserts mask tokens at masked positions and restores the original order.

        Args:
            x_visible    (B, N_vis, D): Encoder output for visible patches.
            mask_token   (1, 1, D):     Learnable mask-token embedding.
            ids_restore  (B, N):        From the matching `forward()` call.

        Returns:
            x_full: (B, N, D) — full sequence in the original patch order.
        """
        B, N_vis, D = x_visible.shape
        N = ids_restore.shape[1]
        N_mask = N - N_vis

        # Tile the mask token to fill all masked positions
        mask_tokens = mask_token.expand(B, N_mask, -1)                 # (B, N_mask, D)

        # Concatenate in shuffled order: [visible | mask tokens]
        x_full = torch.cat([x_visible, mask_tokens], dim=1)            # (B, N, D)

        # Un-shuffle back to original patch order
        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, N, D)

        return x_full
