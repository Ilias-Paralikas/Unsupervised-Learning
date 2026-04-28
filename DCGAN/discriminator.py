import torch
import torch.nn as nn

from DCGAN.modules.blocks.eq_lr_layers import EQLRConv2d
from DCGAN.modules.blocks             import ConvBlock
from DCGAN.modules                    import ResidualBlock


class Discriminator(nn.Module):
    """
    Single-scale CNN discriminator for the ComponentMAE framework.

    Takes a single (B, C, H, W) image — either the real image or the
    composed + seam-smoothed fake — and outputs a scalar score per sample.

    Architecture: from_rgb  →  [ResidualBlock + AvgPool] x depth  →  final head
    Uses EQLRConv2d (equalized learning rate) and LeakyReLU throughout,
    consistent with the rest of the codebase.
    """

    def __init__(self,
                 channels:     list,
                 img_channels: int  = 3,
                 block_depth:  int  = 2,
                 residual:     bool = True,
                 use_norm:     bool = False):
        """
        Args:
            channels     : feature-map widths, e.g. [32, 64, 128, 256, 512].
                           Progression goes channels[0] -> channels[-1].
            img_channels : input image channels (3 for RGB).
            block_depth  : conv layers inside each ResidualBlock.
            residual     : use residual connections inside each block.
            use_norm     : apply PixelNorm (usually False for discriminators).
        """
        super().__init__()

        # Input projection
        self.from_rgb = EQLRConv2d(
            img_channels, channels[0],
            kernel_size=1, stride=1, padding=0,
        )

        # Downsampling blocks
        self.blocks     = nn.ModuleList()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        for i in range(len(channels) - 1):
            self.blocks.append(
                ResidualBlock(
                    in_channels  = channels[i],
                    out_channels = channels[i + 1],
                    depth        = block_depth,
                    residual     = residual,
                    use_norm     = use_norm,
                )
            )

        # Scalar head: two convs to collapse spatial dims, then 1x1 to scalar
        self.head = nn.Sequential(
            ConvBlock(channels[-1], channels[-1],
                      kernel_size=3, stride=1, padding=1, use_norm=use_norm),
            ConvBlock(channels[-1], channels[-1],
                      kernel_size=4, stride=1, padding=0, use_norm=use_norm),
            EQLRConv2d(channels[-1], 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x      : (B, C, H, W)
        Returns:
            scores : (B, 1)  raw un-activated logits
        """
        x = self.from_rgb(x)

        for block in self.blocks:
            x = block(x)
            x = self.downsample(x)

        x = self.head(x)
    
        return x.view(x.size(0), 1)
