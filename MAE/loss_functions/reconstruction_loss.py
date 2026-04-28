import torch
import torch.nn as nn


class MAEReconstructionLoss(nn.Module):
    """
    MAE pre-training loss: mean squared error computed *only* on masked patches.

    Two modes are supported:
      - normalize_target=True  (default, recommended by the paper): each patch's
        pixel values are normalised to zero mean / unit variance before computing
        the MSE.  This prevents the model from exploiting low-frequency colour
        statistics and forces it to learn structure.
      - normalize_target=False: raw pixel MSE, useful for ablation studies.

    Reference: He et al., 2021 — "We compute the mean squared error (MSE)
    between the reconstructed and original images in the pixel space … We
    compute the loss only on masked patches."

    Args:
        patch_size        (int):   Patch height = width P.
        in_channels       (int):   Input image channels C.
        normalize_target  (bool):  Per-patch pixel normalisation.  Default True.
        weight            (float): Scalar multiplier applied to the final loss.
    """

    def __init__(
        self,
        patch_size:       int   = 16,
        in_channels:      int   = 3,
        normalize_target: bool  = True,
        weight:           float = 1.0,
    ):
        super().__init__()
        self.patch_size       = patch_size
        self.in_channels      = in_channels
        self.normalize_target = normalize_target
        self.weight           = weight

    # ── helpers ───────────────────────────────────────────────────────────────

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Re-arranges a batch of images into flat patch vectors.

        Args:
            imgs: (B, C, H, W)

        Returns:
            patches: (B, N, patch_dim)  where  patch_dim = C × P × P
                                               N = (H/P) × (W/P)
        """
        B, C, H, W = imgs.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, (
            f"Image size ({H}×{W}) must be divisible by patch_size {P}."
        )
        h, w = H // P, W // P

        # (B, C, H, W) → (B, C, h, P, w, P)
        patches = imgs.reshape(B, C, h, P, w, P)
        # (B, C, h, P, w, P) → (B, h, w, P, P, C)
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        # (B, h, w, P, P, C) → (B, N, C×P×P)
        patches = patches.reshape(B, h * w, P * P * C)
        return patches

    def unpatchify(self, patches: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Reconstructs images from patch predictions.

        Args:
            patches:  (B, N, patch_dim)
            img_size: (int) height = width of the original image

        Returns:
            imgs: (B, C, H, W)
        """
        P = self.patch_size
        C = self.in_channels
        h = w = img_size // P

        # (B, N, C*P*P) → (B, h, w, P, P, C)
        x = patches.reshape(patches.shape[0], h, w, P, P, C)
        # (B, h, w, P, P, C) → (B, C, h, P, w, P)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        # (B, C, h, P, w, P) → (B, C, H, W)
        imgs = x.reshape(patches.shape[0], C, img_size, img_size)
        return imgs

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        pred:        torch.Tensor,
        target_imgs: torch.Tensor,
        mask:        torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:        (B, N, patch_dim) — decoder predictions for all patches
            target_imgs: (B, C, H, W)      — original input images
            mask:        (B, N)            — binary mask (1 = masked, 0 = visible)

        Returns:
            loss: scalar — weighted MSE averaged over masked patches
        """
        # 1. Convert target images to patch vectors
        target = self.patchify(target_imgs)           # (B, N, patch_dim)

        # 2. Per-patch normalisation (zero-mean, unit-variance per patch)
        if self.normalize_target:
            mean   = target.mean(dim=-1, keepdim=True)                          # (B, N, 1)
            std    = target.var(dim=-1, keepdim=True, unbiased=False).sqrt() + 1e-6  # (B, N, 1)
            target = (target - mean) / std

        # 3. MSE between prediction and (optionally normalised) target
        loss_per_patch = (pred - target).pow(2).mean(dim=-1)  # (B, N)

        # 4. Average only over the masked positions
        loss = (loss_per_patch * mask).sum() / (mask.sum() + 1e-8)

        return self.weight * loss
