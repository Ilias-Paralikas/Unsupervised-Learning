import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    Wasserstein critic for 16×16 image patches.

    No BatchNorm — required by WGAN-GP so the gradient penalty can be
    computed on interpolated inputs without the normalisation statistics
    interfering with the gradient norm.

    Architecture: 3 strided convolutions (16→8→4→2) + linear head.
    Intermediate features from each block are exposed via get_features()
    for use in feature-matching loss.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        bc = base_channels
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, bc,     4, 2, 1),   # 16→8
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(bc,     bc * 2, 4, 2, 1),        # 8→4
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(bc * 2, bc * 4, 4, 2, 1),        # 4→2
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bc * 4 * 2 * 2, 1),
        )

    def get_features(self, x: torch.Tensor) -> list:
        """Returns [f1, f2, f3] — intermediate activations at each conv block."""
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        return [f1, f2, f3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        return self.head(f3)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
