import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (two-layer MLP with GELU activation).

    Follows the standard ViT / Transformer architecture:
        Linear → GELU → Dropout → Linear → Dropout

    The hidden dimension is expanded by `mlp_ratio` relative to the input.

    Args:
        embed_dim  (int):   Input and output dimensionality.
        mlp_ratio  (float): Hidden-layer expansion factor.  Default 4.0.
        dropout    (float): Dropout probability applied after each linear layer.
        bias       (bool):  Whether to add bias to both linear layers.

    Input:  (B, N, D)
    Output: (B, N, D)
    """

    def __init__(
        self,
        embed_dim:  int,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.0,
        bias:       bool  = True,
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1   = nn.Linear(embed_dim,  hidden_dim, bias=bias)
        self.act   = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(hidden_dim, embed_dim,  bias=bias)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            out: (B, N, D)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
