import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Vanilla multi-head self-attention with scaled dot-product.

    The QKV projection is fused into a single Linear layer for efficiency.
    Attention weights are computed as:

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_head)) * V

    Args:
        embed_dim  (int):   Total embedding dimension  D.
        num_heads  (int):   Number of attention heads  H.
                            embed_dim must be divisible by num_heads.
        dropout    (float): Dropout probability on attention weights.
        bias       (bool):  Whether to add bias to QKV and output projections.

    Input:  (B, N, D)
    Output: (B, N, D)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Fused QKV: output is 3 × embed_dim
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim,     bias=bias)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            out: (B, N, D)
        """
        B, N, D = x.shape

        # Compute fused QKV and split heads
        # (B, N, 3D) → (B, N, 3, H, head_dim) → (3, B, H, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)             # each: (B, H, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of value vectors
        out = attn @ v                       # (B, H, N, head_dim)
        out = out.transpose(1, 2)            # (B, N, H, head_dim)
        out = out.reshape(B, N, D)           # (B, N, D)
        out = self.proj(out)
        return out
