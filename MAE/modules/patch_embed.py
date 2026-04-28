import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Splits an image into non-overlapping patches and linearly projects each patch.

    Implemented as a single strided convolution, which is equivalent to
    extracting patches and applying a shared linear layer.

    Args:
        img_size    (int): Height and width of the (square) input image.
        patch_size  (int): Height and width of each (square) patch.
        in_channels (int): Number of input image channels.
        embed_dim   (int): Dimensionality of the output patch embeddings.

    Input:  (B, C, H, W)
    Output: (B, N, embed_dim)  where N = (H // P) * (W // P)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        assert img_size % patch_size == 0, (
            f"Image size {img_size} must be divisible by patch size {patch_size}."
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Strided conv: each output spatial cell = one patch embedding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            tokens: (B, N, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, (
            f"Input size ({H}x{W}) must match model img_size ({self.img_size}x{self.img_size})."
        )

        x = self.proj(x)          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)          # (B, embed_dim, N)
        x = x.transpose(1, 2)     # (B, N, embed_dim)
        return x
