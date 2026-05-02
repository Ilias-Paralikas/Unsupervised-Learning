import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_cm_attention(
    img,           # (C, H, W)  tanh-normalised, single image
    mask_a,        # (N,) or (1, N)  1 = masked in pass A
    attn_weights,  # list of N_comp tensors, each (1, N, N)  head-averaged
    seg_probs,     # (1, N_comp, H, W) or (N_comp, H, W)  after softmax
    query_patch,   # int — patch index to inspect (must be masked in pass A)
    patch_size=16,
):
    """
    For a single query patch (masked in pass A) show, for every component:
      - Top row : attention heatmap from the query to visible patches
      - Bottom row: segmentation probability map for that component

    The attention is re-normalised to visible keys only (masked positions
    hold mask-token embeddings and carry no content signal), matching what
    CrossMaskConsistencyLoss does internally.

    Usage
    -----
    with torch.no_grad():
        preds, mask_a, attn_weights = rec_model(img.unsqueeze(0), return_attn=True)
        seg_logits = seg_model(img.unsqueeze(0))
        seg_probs  = torch.softmax(seg_logits, dim=1)

    visualize_cm_attention(img, mask_a[0], attn_weights, seg_probs, query_patch=42)
    """
    C, H, W  = img.shape
    mask_a   = mask_a.reshape(-1).float().cpu()   # (N,)
    N        = mask_a.shape[0]
    h        = H // patch_size
    N_comp   = len(attn_weights)

    assert mask_a[query_patch] == 1, (
        f"query_patch {query_patch} is visible in pass A (mask_a == 0). "
        f"Pick a patch where mask_a == 1."
    )

    visible_flag = (1.0 - mask_a)                 # (N,)  1 = visible
    qi, qj       = query_patch // h, query_patch % h
    img_disp     = img.mul(0.5).add(0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(
        2, N_comp + 1,
        figsize=(3.5 * (N_comp + 1), 7),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )

    # ── Column 0: original image + mask overview ──────────────────────────
    for row in range(2):
        ax = axes[row, 0]
        ax.imshow(img_disp)

        # blue tint on masked patches
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        mask_grid = mask_a.reshape(h, h).numpy()
        for pi in range(h):
            for pj in range(h):
                if mask_grid[pi, pj] == 1:
                    r0, r1 = pi * patch_size, (pi + 1) * patch_size
                    c0, c1 = pj * patch_size, (pj + 1) * patch_size
                    overlay[r0:r1, c0:c1] = [0.2, 0.4, 1.0, 0.35]
        ax.imshow(overlay)

        # red highlight on query patch
        rect = mpatches.Rectangle(
            (qj * patch_size, qi * patch_size), patch_size, patch_size,
            linewidth=2, edgecolor="red", facecolor="red", alpha=0.55,
        )
        ax.add_patch(rect)
        ax.axis("off")

    axes[0, 0].set_title("mask A\n(blue=masked, red=query)", fontsize=8)
    axes[1, 0].set_title("", fontsize=8)

    # ── Columns 1…N_comp: per-component attention + seg ──────────────────
    for n, attn in enumerate(attn_weights):
        attn_n = attn[0].cpu()                    # (N, N)  remove batch dim

        # re-normalise to visible keys only
        attn_vis = attn_n * visible_flag.unsqueeze(0)        # (N, N)
        attn_vis = attn_vis / (attn_vis.sum(dim=-1, keepdim=True) + 1e-8)

        # attention vector FROM the query patch  →  (N,)
        query_attn = attn_vis[query_patch].numpy()           # only visible slots non-zero

        # upsample to pixel space via nearest-neighbour (same as mask upsampling)
        heatmap = query_attn.reshape(h, h)
        heatmap_px = np.kron(heatmap, np.ones((patch_size, patch_size)))  # (H, W)

        # ── top: attention heatmap ────────────────────────────────────────
        ax_top = axes[0, n + 1]
        ax_top.imshow(img_disp)
        im_attn = ax_top.imshow(heatmap_px, alpha=0.65, cmap="hot", vmin=0)
        # re-draw query highlight
        rect = mpatches.Rectangle(
            (qj * patch_size, qi * patch_size), patch_size, patch_size,
            linewidth=2, edgecolor="cyan", facecolor="cyan", alpha=0.45,
        )
        ax_top.add_patch(rect)
        ax_top.set_title(f"comp {n}  |  attn from query", fontsize=8)
        ax_top.axis("off")
        plt.colorbar(im_attn, ax=ax_top, fraction=0.035, pad=0.02)

        # ── bottom: segmentation probability ─────────────────────────────
        sp = seg_probs
        if sp.dim() == 4:
            sp = sp[0]                             # (N_comp, H, W)
        seg_n = sp[n].cpu().numpy()               # (H, W)

        ax_bot = axes[1, n + 1]
        ax_bot.imshow(img_disp)
        im_seg = ax_bot.imshow(seg_n, alpha=0.65, cmap="viridis", vmin=0, vmax=1)
        ax_bot.set_title(f"comp {n}  |  seg prob", fontsize=8)
        ax_bot.axis("off")
        plt.colorbar(im_seg, ax=ax_bot, fraction=0.035, pad=0.02)

    fig.suptitle(
        f"CM attention  |  query patch {query_patch}  "
        f"(row {qi}, col {qj})",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    return fig


# ── Example usage (paste into a notebook cell) ────────────────────────────────
#
# from visualize_cm_attention import visualize_cm_attention
#
# data, _ = next(iter(train_loader))
# img = data[0].to(device)                # pick first image in batch
#
# with torch.no_grad():
#     preds, mask_a, attn_weights = rec_model(img.unsqueeze(0), return_attn=True)
#     seg_logits = seg_model(img.unsqueeze(0))
#     seg_probs  = torch.softmax(seg_logits, dim=1)
#
# # pick any patch index where mask_a[0, idx] == 1
# masked_indices = mask_a[0].nonzero(as_tuple=True)[0].tolist()
# query = masked_indices[len(masked_indices) // 2]   # middle masked patch
#
# fig = visualize_cm_attention(img, mask_a[0], attn_weights, seg_probs, query_patch=query)
