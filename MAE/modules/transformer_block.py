import torch
import torch.nn as nn

from .attention    import MultiHeadSelfAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    Standard ViT transformer block with Pre-LayerNorm (Pre-LN) residual connections.

    Architecture:
        x ← x + MHSA(LN(x))
        x ← x + FFN(LN(x))

    Pre-LN is used throughout both the encoder and decoder as in the MAE paper.

    Args:
        embed_dim  (int):   Token embedding dimension.
        num_heads  (int):   Number of self-attention heads.
        mlp_ratio  (float): FFN hidden-dim expansion factor.  Default 4.0.
        dropout    (float): Dropout for both attention weights and FFN activations.
        bias       (bool):  Bias in attention / FFN projections and LayerNorm.

    Input:  (B, N, D)
    Output: (B, N, D)
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
        self.attn  = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn   = FeedForward(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            out: (B, N, D)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
