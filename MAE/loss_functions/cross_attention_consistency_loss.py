import torch
import torch.nn as nn   


class CrossMaskConsistencyLoss(nn.Module):
    """
    Cross-Mask Consistency Loss — single forward pass.

    One call to cross_mask_forward gives everything needed:
        - mask_a, mask_b        : which patches are masked / visible
        - attn_weights          : attention maps from Pass A (detached)
        - pass_b_preds          : Pass B predictions (gradients kept)

    Weight formula for component n, target patch j (old-visible):

        w[n, j] = sum_{i in masked}  seg_n[i]  *  A_n[i, j]

        seg_n[i]  = mean seg prob of component n over pixels of masked patch i
        A_n[i, j] = attention from masked patch i to visible context patch j

    Only pass_b_preds receives gradients. Everything else is detached.
    seg_probs is always detached — this loss ONLY trains rec_model.

    Parameters
    ----------
    pass_b_weight : float
    """

    def __init__(self, pass_b_weight: float = 1.0):
        super().__init__()
        self.pass_b_weight = pass_b_weight

    @staticmethod
    def _patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
        B, C, H, W = imgs.shape
        P = patch_size
        h = w = H // P
        x = imgs.reshape(B, C, h, P, w, P)
        x = x.permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, h * w, P * P * C)

    @staticmethod
    def _patchify_seg(seg_probs: torch.Tensor, patch_size: int) -> torch.Tensor:
        """(B, N_comp, H, W) -> (B, N_comp, N)  mean seg prob per patch"""
        B, N_comp, H, W = seg_probs.shape
        P = patch_size
        h = w = H // P
        x = seg_probs.reshape(B, N_comp, h, P, w, P)
        return x.mean(dim=(3, 5)).reshape(B, N_comp, h * w)

    @staticmethod
    def _normalize(patches: torch.Tensor) -> torch.Tensor:
        mean = patches.mean(dim=-1, keepdim=True)
        std  = patches.var(dim=-1, keepdim=True, unbiased=False).sqrt() + 1e-6
        return (patches - mean) / std

    def forward(
        self,
        target_imgs:  torch.Tensor,       # (B, C, H, W)
        seg_probs:    torch.Tensor,       # (B, N_comp, H, W)
        mask_a:       torch.Tensor,       # (B, N)  1 = masked
        attn_weights: list,               # N_comp x (B, N, N)
        pass_b_preds: list,               # N_comp x (B, N, patch_dim)
        mask_b:       torch.Tensor,       # (B, N)  1 = old-visible = Pass B targets
        patch_size:   int,
    ) -> torch.Tensor:
        seg_det     = seg_probs.detach()
        seg_patches = self._patchify_seg(seg_det, patch_size)    # (B, N_comp, N)
        target_patches = self._normalize(
            self._patchify(target_imgs, patch_size)
        )                                                         # (B, N, patch_dim)

        N_comp     = len(pass_b_preds)
        loss_total = torch.zeros(1, device=target_imgs.device)
        n_active   = 0

        for n in range(N_comp):
            seg_n    = seg_patches[:, n, :]                      # (B, N)
            row_gate = (seg_n * mask_a).unsqueeze(-1)            # (B, N, 1)
            w_n      = (attn_weights[n] * row_gate).sum(dim=1)   # (B, N)
            w_n      = w_n * mask_b                              # zero out non-targets

            w_sum = w_n.sum()
            if w_sum.item() < 1e-4:
                continue

            # plain weighted MSE — nothing fancy
            mse_n  = (pass_b_preds[n] - target_patches).pow(2).mean(dim=-1)  # (B, N)
            loss_n = (w_n * mse_n).sum() / w_sum                 # weighted average

            loss_total = loss_total + loss_n
            n_active  += 1

        if n_active == 0:
            return torch.zeros(1, device=target_imgs.device).squeeze()

        return self.pass_b_weight * (loss_total / n_active).squeeze()