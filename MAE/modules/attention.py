import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Vanilla multi-head self-attention.

    Extra flag
    ----------
    return_attn : bool
        When True, forward() returns (out, attn_weights) where
        attn_weights has shape (B, H, N, N).  Used by the decoder to
        extract which visible patches each masked-patch position
        attends to.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout:   float = 0.0,
        bias:      bool  = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv       = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj      = nn.Linear(embed_dim, embed_dim,     bias=bias)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x:           torch.Tensor,
        return_attn: bool = False,
    ):
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)                 # each (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                           # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        if return_attn:
            return out, attn                     # attn: (B, H, N, N)
        return out
