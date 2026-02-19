import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

class GroupRandomAffine(nn.Module):
    def __init__(self, degrees=30, translate=None, scale=None, shear=None, interpolation=F.InterpolationMode.BILINEAR):
        super().__init__()
        self.affine = T.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation
        )

    def forward(self, batch):
        # Detect if we have a group dimension N
        is_4d = False
        if batch.ndim == 4:
            # Input is [B, C, H, W], we'll treat it as [B, 1, C, H, W]
            batch = batch.unsqueeze(1)
            is_4d = True
            
        B, N, C, H, W = batch.shape
        out = torch.zeros_like(batch)

        for i in range(B):
            # Sample params once for this batch element
            params = self.affine.get_params(
                self.affine.degrees,
                self.affine.translate,
                self.affine.scale,
                self.affine.shear,
                (H, W),
            )

            # Fold N and C to transform everything in one batch element at once
            group = batch[i].view(1, -1, H, W)
            transformed = F.affine(
                group, 
                *params, 
                interpolation=self.affine.interpolation
            )
            out[i] = transformed.view(N, C, H, W)

        # If we started with 4D, return 4D
        if is_4d:
            out = out.squeeze(1)
            
        return out


class Augmentations():
    def __init__(self,
                 degrees=30,
                 translate=(0.1, 0.2),
                 scale=(0.8, 1.2),
                 shear=15):
        
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

        self.affine_augmentations = GroupRandomAffine(
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear
        )

    def __call__(self, batch,train=True):
        if train:
            batch= self.affine_augmentations(batch)

        return batch 

            