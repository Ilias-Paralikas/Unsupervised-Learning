import math
import torch


def build_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
) -> torch.Tensor:
    """
    Builds a 2-D fixed sinusoidal positional embedding as used in MAE.

    The embedding independently encodes the (h, w) grid position with
    sin/cos at different frequencies, concatenated across four quarters
    of the embedding dimension.

    Reference: He et al., 2021, Appendix A — "fixed 2D sin-cos position embedding."

    Args:
        embed_dim (int):  Total embedding dimension.  Must be divisible by 4.
        grid_size (int):  Number of patches along each spatial dimension.
        cls_token (bool): If True, prepends a zero-filled CLS position embedding.

    Returns:
        pos_embed: (1, N, embed_dim)   or   (1, N+1, embed_dim) if cls_token=True
                   where N = grid_size * grid_size
    """
    assert embed_dim % 4 == 0, (
        f"embed_dim ({embed_dim}) must be divisible by 4 for 2D sin-cos embeddings."
    )

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    # grid_w is the column index (x-axis), grid_h is the row index (y-axis)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")  # (G, G)

    # Frequency bands for one quarter of the embedding
    quarter = embed_dim // 4
    omega = torch.arange(quarter, dtype=torch.float32) / quarter       # [0, 1)
    omega = 1.0 / (10000 ** omega)                                      # (D/4,)

    # Outer product: each position × each frequency
    pos_h = grid_h.reshape(-1, 1) * omega.reshape(1, -1)  # (N, D/4)
    pos_w = grid_w.reshape(-1, 1) * omega.reshape(1, -1)  # (N, D/4)

    pos_embed = torch.cat(
        [torch.sin(pos_h), torch.cos(pos_h), torch.sin(pos_w), torch.cos(pos_w)],
        dim=1,
    )  # (N, embed_dim)

    pos_embed = pos_embed.unsqueeze(0)  # (1, N, embed_dim)

    if cls_token:
        cls_pos = torch.zeros(1, 1, embed_dim)
        pos_embed = torch.cat([cls_pos, pos_embed], dim=1)  # (1, N+1, embed_dim)

    return pos_embed
