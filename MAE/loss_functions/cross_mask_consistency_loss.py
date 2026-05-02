import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossMaskConsistencyLoss(nn.Module):
    """
    Two-pass cross-mask consistency loss.

    Intuition
    ---------
    In pass A, component n uses visible patch i (e.g. car door) to reconstruct
    masked patch j (e.g. car window).  If patch j is heavily weighted in the
    final blended reconstruction (high seg_prob), then patch i must carry
    information about this component's semantic region.  Pass B inverts the
    mask so that i is now masked; we then penalise component n for failing to
    reconstruct i, weighted by how important i was in pass A.

    Weight for visible-in-A patch i, component n:

        w_n[i] = Σ_{j ∈ masked_A}  (seg_prob_n[j] / mean_n)  ·  attn_n[j → i]

    where mean_n = seg_prob_n.mean() over all patches, and attn_n[j → i] is
    re-normalised to sum over visible keys only (masked positions are just
    mask_token embeddings and carry no signal).

    Dividing by mean_n makes the loss area-independent: a component covering
    50% of the image gets the same total gradient as one covering 5%, so
    large-area classes (pavement) don't dominate over small-area ones (humans).

    The MSE is computed in patch space directly against the raw decoder output,
    avoiding the unpatchify → interpolate round-trip.

    When a discriminator is supplied, a feature-matching term is added alongside
    the MSE, using the same importance weights w_n.  Real features are computed
    once (outside the component loop) since target_patches is shared.  This adds
    N_comp discriminator forward passes per batch — the real pass is under
    torch.no_grad() so it is cheap.

    Parameters
    ----------
    weight      : float  — overall loss scale
    fm_weight   : float  — feature-matching weight relative to MSE within this
                           loss (0 disables feature matching entirely)
    patch_size  : int
    in_channels : int
    """

    def __init__(
        self,
        weight:      float = 0.1,
        fm_weight:   float = 0.0,
        patch_size:  int   = 16,
        in_channels: int   = 3,
    ):
        super().__init__()
        self.weight      = weight
        self.fm_weight   = fm_weight
        self.patch_size  = patch_size
        self.in_channels = in_channels

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * p * p)

    def _patch_weights(
        self,
        seg_probs: torch.Tensor,   # (B, N_comp, H, W)
        attn:      torch.Tensor,   # (B, N, N)  head-averaged, last decoder block
        mask_a:    torch.Tensor,   # (B, N)  1 = masked in pass A
        comp_idx:  int,
    ) -> torch.Tensor:             # (B, N)  — weight for each visible-in-A patch
        B, N = mask_a.shape
        h    = int(N ** 0.5)

        masked_flag  = mask_a.float()           # (B, N)
        visible_flag = 1.0 - masked_flag        # (B, N)

        seg_patch = F.adaptive_avg_pool2d(
            seg_probs[:, comp_idx : comp_idx + 1].detach(), (h, h)
        ).reshape(B, N)

        area_n     = seg_patch.mean(dim=1, keepdim=True).clamp(min=1e-8)
        seg_masked = (seg_patch / area_n) * masked_flag

        attn_vis = attn * visible_flag.unsqueeze(1)
        attn_vis = attn_vis / (attn_vis.sum(dim=-1, keepdim=True) + 1e-8)

        w = torch.bmm(seg_masked.unsqueeze(1), attn_vis).squeeze(1)  # (B, N)
        return w * visible_flag

    def forward(
        self,
        seg_probs:     torch.Tensor,        # (B, N_comp, H, W)  after softmax
        attn_weights:  list,                 # N_comp × (B, N, N)
        mask_a:        torch.Tensor,         # (B, N)  1 = masked in pass A
        pass_b_preds:  list,                 # N_comp × (B, N, patch_dim)
        target_imgs:   torch.Tensor,         # (B, C, H, W)
        discriminator = None,                # PatchDiscriminator | None
    ) -> torch.Tensor:
        P  = self.patch_size
        C  = self.in_channels
        target_patches = self.patchify(target_imgs)          # (B, N, C*P*P)
        B, N, _        = target_patches.shape

        # Compute discriminator features for real patches once — shared across
        # all components.  Kept under no_grad since they are fixed targets.
        feat_real = None
        if discriminator is not None and self.fm_weight > 0:
            real_4d = target_patches.reshape(B * N, C, P, P)
            with torch.no_grad():
                feat_real = discriminator.get_features(real_4d)  # list of (B*N, ...)

        total = torch.tensor(0.0, device=target_imgs.device)

        for n, (attn, pred_b) in enumerate(zip(attn_weights, pass_b_preds)):
            w_n = self._patch_weights(seg_probs, attn, mask_a, n)   # (B, N)

            # ── MSE in patch space ────────────────────────────────────────────
            per_patch_mse = (pred_b - target_patches).pow(2).mean(dim=-1)  # (B, N)
            mse_loss_n    = (per_patch_mse * w_n).sum() / (w_n.sum() + 1e-8)

            loss_n = mse_loss_n

            # ── Feature-matching in discriminator space ───────────────────────
            if feat_real is not None:
                fake_4d    = torch.tanh(pred_b).reshape(B * N, C, P, P)
                feat_fake  = discriminator.get_features(fake_4d)  # list of (B*N, ...)

                per_patch_fm = sum(
                    (ff - fr).pow(2).mean(dim=(1, 2, 3))   # (B*N,)
                    for ff, fr in zip(feat_fake, feat_real)
                ) / len(feat_real)
                per_patch_fm = per_patch_fm.reshape(B, N)

                fm_loss_n = (per_patch_fm * w_n).sum() / (w_n.sum() + 1e-8)
                loss_n    = loss_n + self.fm_weight * fm_loss_n

            total = total + loss_n

        return self.weight * total / len(attn_weights)
