# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# from .eq_lr_layers import EQLRConv2d

# class SpatialVectorizer(nn.Module):
#     def __init__(self, 
#                  in_channels, 
#                  out_channels, 
#                  number_of_components,
#                  degrees_of_freedom,
#                  grid_size=4): # 'd'
#         super().__init__()
#         self.number_of_components = number_of_components
#         self.degrees_of_freedom = degrees_of_freedom
#         self.out_channels = out_channels
#         self.grid_size = grid_size
        
#         # 1. The Coarse Router
#         # Output shape: (b, n*m, h//d, w//d)
#         self.router = EQLRConv2d(
#             in_channels=in_channels, 
#             out_channels=self.number_of_components * self.degrees_of_freedom, 
#             kernel_size=self.grid_size, 
#             stride=self.grid_size, 
#             padding=0
#         )
        
#         # 2. Trainable DxD Tiles!
#         # Shape: (N, M, C, D, D) 
#         # Each 'vector' is now a fully expressible DxD spatial patch.
#         # We initialize it based on the number of elements in the patch (C * D * D)
#         elements_per_tile = self.out_channels * (self.grid_size ** 2)
#         self.shared_tiles = nn.Parameter(
#             torch.randn(self.number_of_components, self.degrees_of_freedom, self.out_channels, self.grid_size, self.grid_size) / math.sqrt(elements_per_tile)
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape
#         d = self.grid_size
        
#         # Ensure the image dimensions are cleanly divisible by the grid size
#         assert h % d == 0 and w % d == 0, f"Image dims {(h,w)} must be divisible by grid_size {d}"
        
#         h_coarse, w_coarse = h // d, w // d
        
#         # Step 1: Generate routing map for the patches
#         # routing_map: (b, n*m, h_coarse, w_coarse)
#         routing_map = self.router(x)
        
#         # Step 2: Reshape to separate components and degrees of freedom
#         # routing_map: (b, n, m, h_coarse, w_coarse)
#         routing_map = routing_map.view(b, self.number_of_components, self.degrees_of_freedom, h_coarse, w_coarse)
        
#         # Step 3: Softmax across 'm' to choose the tile combination
#         routing_weights = F.softmax(routing_map, dim=2)

#         # Step 4: Normalize the tiles (flattening the spatial dims for normalization)
#         # We normalize across (C, D, D) so the whole patch has a variance of 1.0
#         normalized_tiles = F.normalize(
#             self.shared_tiles.view(self.number_of_components, self.degrees_of_freedom, -1), 
#             p=2, dim=-1, eps=1e-8
#         ) * math.sqrt(self.out_channels * d * d)
        
#         # Reshape back to tile format
#         normalized_tiles = normalized_tiles.view(self.number_of_components, self.degrees_of_freedom, self.out_channels, d, d)

#         # Step 5: Compute the linear combination of tiles for each grid location
#         # routing_weights shape: (b, n, m, h_coarse, w_coarse)
#         # normalized_tiles shape: (n, m, c', d, d)
#         # Output shape: (b, n, c', h_coarse, w_coarse, d, d)
#         coarse_out = torch.einsum('bnmxy,nmcIJ->bncxyIJ', routing_weights, normalized_tiles)
        
#         # Step 6: Fold the patches back into a full (h, w) image!
#         # We have (h_coarse, w_coarse) blocks of (d, d) size.
#         # We use permute to interleave them correctly into (h_coarse * d, w_coarse * d)
#         out = coarse_out.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
#         out = out.view(b, self.number_of_components, self.out_channels, h, w)
        
#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .eq_lr_layers import EQLRConv2d

class SpatialVectorizer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 number_of_components,
                 degrees_of_freedom,
                 grid_size=4):
        super().__init__()
        self.number_of_components = number_of_components
        self.degrees_of_freedom = degrees_of_freedom
        self.out_channels = out_channels
        self.grid_size = grid_size
        
        # 1. The Coarse Router now predicts BOTH weights AND Affine params
        # For each component and DoF, we need 1 weight + 6 affine params = 7
        self.router = EQLRConv2d(
            in_channels=in_channels, 
            out_channels=self.number_of_components * self.degrees_of_freedom * 7, 
            kernel_size=self.grid_size, 
            stride=self.grid_size, 
            padding=0
        )
        
        # 2. Trainable Oversized 2Dx2D Tiles (Canonical templates)
        # We make the tiles twice the size of the grid cell so the affine 
        # transform has "canvas" to translate and scale without hitting the edges.
        self.tile_size = self.grid_size * 2 
        elements_per_tile = self.out_channels * (self.tile_size ** 2)

        self.shared_tiles = nn.Parameter(
            torch.randn(
                self.number_of_components, 
                self.degrees_of_freedom, 
                self.out_channels, 
                self.tile_size, 
                self.tile_size
            ) / math.sqrt(elements_per_tile)
        )

        # 3. Identity Affine Matrix for stable initialization
        # Shape: (1, 1, 1, 1, 1, 6) -> represents [ [1, 0, 0], [0, 1, 0] ]
        self.register_buffer(
            'identity_affine', 
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).view(1, 1, 1, 1, 1, 6)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        d = self.grid_size
        
        assert h % d == 0 and w % d == 0, f"Image dims {(h,w)} must be divisible by grid_size {d}"
        h_coarse, w_coarse = h // d, w // d
        
        # Step 1: Generate routing map (Weights + Affine params)
        # Shape: (b, n*m*7, h_coarse, w_coarse)
        routing_map = self.router(x)
        
        # Step 2: Reshape to separate components, DoF, and the 7 parameters
        routing_map = routing_map.view(b, self.number_of_components, self.degrees_of_freedom, 7, h_coarse, w_coarse)
        
        # Step 3: Split into Selection Weights (1) and Affine Params (6)
        raw_weights = routing_map[:, :, :, 0, :, :]      # Shape: (b, n, m, h_c, w_c)
        affine_params = routing_map[:, :, :, 1:, :, :]   # Shape: (b, n, m, 6, h_c, w_c)
        
        # Softmax the weights
        routing_weights = F.softmax(raw_weights, dim=2)
        
        # Format Affine Params into 2x3 matrices, adding the identity matrix so they start un-warped
        # Shape becomes: (b, n, m, h_c, w_c, 6)
        affine_params = affine_params.permute(0, 1, 2, 4, 5, 3).contiguous() 
        
        # Dampen the affine predictions slightly so it doesn't instantly flip the image off-screen early in training
        affine_matrices = self.identity_affine + (affine_params * 0.1) 
        affine_matrices = affine_matrices.view(-1, 2, 3) # Flatten for affine_grid: (b*n*m*h_c*w_c, 2, 3)

        # Step 4: Normalize the 2Dx2D oversized tiles
        normalized_tiles = F.normalize(
            self.shared_tiles.view(self.number_of_components, self.degrees_of_freedom, -1), 
            p=2, dim=-1, eps=1e-8
        ) * math.sqrt(self.out_channels * (self.tile_size ** 2))
        
        normalized_tiles = normalized_tiles.view(
            self.number_of_components, self.degrees_of_freedom, self.out_channels, self.tile_size, self.tile_size
        )
        
        # Expand the tiles to match the batch and spatial grid size so we can warp them
        # From (n, m, c', 2d, 2d) -> (b, n, m, h_c, w_c, c', 2d, 2d)
        expanded_tiles = normalized_tiles.view(1, self.number_of_components, self.degrees_of_freedom, 1, 1, self.out_channels, self.tile_size, self.tile_size)
        expanded_tiles = expanded_tiles.expand(b, -1, -1, h_coarse, w_coarse, -1, -1, -1).contiguous()
        
        # Flatten for grid_sample: (b*n*m*h_c*w_c, c', 2d, 2d)
        flat_tiles = expanded_tiles.view(-1, self.out_channels, self.tile_size, self.tile_size)

        # Step 5: WARP THE TILES
        # We explicitly tell affine_grid we want the output to be size DxD, NOT 2Dx2D.
        # This causes the affine matrix to "crop" a DxD window out of our oversized tiles!
        b_flat = flat_tiles.size(0)
        output_size = torch.Size([b_flat, self.out_channels, d, d])
        
        grid = F.affine_grid(affine_matrices, output_size, align_corners=False).to(flat_tiles.device)

        # grid_sample pulls the pixels using the affine grid. Zeros padding handles anything shifted out of bounds.
        warped_tiles = F.grid_sample(flat_tiles, grid, align_corners=False, padding_mode='zeros')
        
        # Reshape warped DxD tiles back to the separated dimensions
        # Shape: (b, n, m, h_c, w_c, c', d, d)
        warped_tiles = warped_tiles.view(b, self.number_of_components, self.degrees_of_freedom, h_coarse, w_coarse, self.out_channels, d, d)

        # Step 6: Compute the linear combination of warped tiles 
        # We use einsum to multiply the softmax probabilities by the warped tiles and sum over 'm'
        coarse_out = torch.einsum('bnmxy,bnmxycIJ->bncxyIJ', routing_weights, warped_tiles)
        
        # Step 7: Fold the patches back into the full (h, w) feature map
        # Interleave the DxD spatial blocks perfectly into the HxW grid
        out = coarse_out.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
        out = out.view(b, self.number_of_components, self.out_channels, h, w)
        
        return out