import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .eq_lr_layers import EQLRConv2d


class SpatialVectorizer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,          # C': output channels per component
                 number_of_components,  # N: independent vectorizers
                 degrees_of_freedom,    # D: codebook size (tiles per component)
                 grid_size=1):          # gs: spatial size of each tile
        super().__init__()

        assert out_channels <= degrees_of_freedom, \
            f"out_channels ({out_channels}) must be <= degrees_of_freedom ({degrees_of_freedom})"

        self.N  = number_of_components
        self.D  = degrees_of_freedom
        self.C  = out_channels          # active tiles = out_channels (same thing)
        self.gs = grid_size
        self.out_channels = out_channels

        # ── 1. Routing network ────────────────────────────────────────────────
        # One router for all N components.
        # Output: (B, N*D, H/gs, W/gs), reshaped to (B, N, D, Hc, Wc).
        self.router = EQLRConv2d(
            in_channels=in_channels,
            out_channels=self.N * self.D,
            kernel_size=grid_size,
            stride=grid_size,
            padding=0
        )

        # ── 2. Shared tile codebook ───────────────────────────────────────────
        # Each component has its own D tiles, each a single-channel gs×gs patch.
        # Shape: (N, D, 1, gs, gs)
        elements_per_tile = grid_size ** 2
        self.shared_tiles = nn.Parameter(
            torch.randn(self.N, self.D, 1, grid_size, grid_size) / math.sqrt(elements_per_tile)
        )

        # ── 3. Channel projection ─────────────────────────────────────────────
        # Projects D sparse channels → out_channels, independently per component.
        # Grouped conv (groups=N) ensures component n only sees its own D channels.
        # Input:  (B, N*D, H, W)
        # Output: (B, N*out_channels, H, W)
        self.proj = nn.Conv2d(
            in_channels=self.N * self.D,
            out_channels=self.N * self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.N
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _normalize_tiles(self):
        """
        L2-normalise each tile across its gs*gs elements, then rescale
        so each element has unit variance.
        Returns: (N, D, 1, gs, gs)
        """
        flat   = self.shared_tiles.view(self.N, self.D, -1)        # (N, D, gs*gs)
        normed = F.normalize(flat, p=2, dim=-1, eps=1e-8)
        normed = normed * math.sqrt(self.gs ** 2)
        return normed.view(self.N, self.D, 1, self.gs, self.gs)

    def _topk_ste(self, routing):
        """
        Top-C hard selection with Straight-Through Estimator.

        Forward  : binary mask — exactly C ones per (n, spatial location).
        Backward : gradients flow through sigmoid(routing) so every tile trains.

        Args:
            routing  : (B, N, D, Hc, Wc)
        Returns:
            mask_ste : (B, N, D, Hc, Wc)
        """
        _, topk_idx = torch.topk(routing, self.C, dim=2)            # (B, N, C, Hc, Wc)
        mask_hard   = torch.zeros_like(routing)
        mask_hard.scatter_(2, topk_idx, 1.0)

        mask_soft = torch.sigmoid(routing)

        return mask_soft + (mask_hard - mask_soft).detach()

    # ── similarity loss ───────────────────────────────────────────────────────

    def similarity_loss(self):
        """
        Computes two diversity losses over the tile codebook.

        Intra-component loss: penalises similarity between tiles that belong to
        the SAME component. High loss → tiles within a codebook are redundant.

        Inter-component loss: penalises similarity between tiles that belong to
        DIFFERENT components. High loss → components are learning the same things.

        Both losses are the mean absolute cosine similarity over the relevant
        pairs (diagonal / self-similarity terms are excluded).

        Returns:
            intra_loss : scalar tensor  (same-component tile similarity)
            inter_loss : scalar tensor  (cross-component tile similarity)
        """
        # Unit-normalise tiles for cosine similarity
        # (N, D, gs*gs)
        flat   = self.shared_tiles.view(self.N, self.D, -1)
        normed = F.normalize(flat, p=2, dim=-1, eps=1e-8)          # (N, D, gs*gs)

        # ── Intra-component ───────────────────────────────────────────────────
        # For each component n, compute the D×D cosine similarity matrix.
        # bmm: (N, D, gs*gs) × (N, gs*gs, D) → (N, D, D)
        sim_intra = torch.bmm(normed, normed.transpose(1, 2))       # (N, D, D)

        # Exclude diagonal (each tile is trivially similar to itself)
        diag_mask  = torch.eye(self.D, dtype=torch.bool, device=normed.device)  # (D, D)
        off_diag   = ~diag_mask                                      # True for pairs we care about
        intra_loss = sim_intra[:, off_diag].abs().mean()

        # ── Inter-component ───────────────────────────────────────────────────
        # Flatten all N*D tiles into one matrix and compute the full similarity.
        flat_all = normed.view(self.N * self.D, -1)                 # (N*D, gs*gs)
        sim_all  = flat_all @ flat_all.T                            # (N*D, N*D)

        # Build a mask that is True only for CROSS-component pairs
        # comp_idx[i] = which component tile i belongs to
        comp_idx   = torch.arange(self.N, device=normed.device).repeat_interleave(self.D)
        cross_mask = comp_idx.unsqueeze(0) != comp_idx.unsqueeze(1) # (N*D, N*D)

        inter_loss = sim_all[cross_mask].abs().mean()

        return intra_loss, inter_loss

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x):
        """
        Args:
            x  : (B, in_channels, H, W)
        Returns:
            out: (B, N, out_channels, H, W)   ← same interface as original
        """
        b, c, h, w = x.shape
        gs = self.gs

        assert h % gs == 0 and w % gs == 0, \
            f"Spatial dims {(h, w)} must be divisible by grid_size {gs}"

        hc, wc = h // gs, w // gs

        # ── Step 1: routing ───────────────────────────────────────────────────
        # (B, N*D, Hc, Wc) → (B, N, D, Hc, Wc)
        routing = self.router(x).view(b, self.N, self.D, hc, wc)

        # ── Step 2: top-C selection with STE ─────────────────────────────────
        # (B, N, D, Hc, Wc)
        mask_ste = self._topk_ste(routing)

        # ── Step 3: normalise tile codebook ──────────────────────────────────
        # (N, D, gs, gs)
        tiles = self._normalize_tiles().squeeze(2)

        # ── Step 4: stamp tiles at every coarse location ─────────────────────
        # mask_ste : (B, N, D, Hc, Wc)
        # tiles    : (N, D, gs, gs)
        # output   : (B, N, D, Hc, Wc, gs, gs)
        out = torch.einsum('bndxy,ndIJ->bndxyIJ', mask_ste, tiles)

        # ── Step 5: fold back to full resolution ─────────────────────────────
        # (B, N, D, Hc, Wc, gs, gs)
        # → (B, N, D, Hc, gs, Wc, gs)  via permute
        # → (B, N, D, H, W)             via view
        out = out.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
        out = out.view(b, self.N, self.D, h, w)

        # ── Step 6: project D → out_channels per component ───────────────────
        # (B, N, D, H, W) → (B, N*D, H, W)
        out = out.view(b, self.N * self.D, h, w)
        # grouped conv: component n sees only channels [n*D : (n+1)*D]
        # (B, N*D, H, W) → (B, N*out_channels, H, W)
        out = self.proj(out)
        # (B, N*out_channels, H, W) → (B, N, out_channels, H, W)
        out = out.view(b, self.N, self.out_channels, h, w)

        return out