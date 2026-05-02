import torch
import torch.nn as nn

from .attention    import MultiHeadSelfAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    Pre-LN transformer block.

    Extra flag
    ----------
    return_attn : bool
        Propagated to the inner MHSA.  When True, forward() returns
        (out, attn_weights).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.0,
        bias:      bool  = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout, bias)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn   = FeedForward(embed_dim, mlp_ratio, dropout, bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, attn_weights          # (B, H, N, N)
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x
