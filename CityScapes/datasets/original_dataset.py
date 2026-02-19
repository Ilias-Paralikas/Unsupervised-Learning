import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
class CityscapesDataset(Dataset):
    def __init__(self, root='./CityScapes/data/original_data',size=(512,512),get_labels=False):
        self.root = root
        self.size=  size
        self.get_labels = get_labels
        self.transform  = v2.Compose([
                v2.Resize(size=self.size, interpolation=InterpolationMode.BILINEAR),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True), 
            ])
        self.target_transform = v2.Compose([
            v2.Resize(size=self.size, interpolation=InterpolationMode.NEAREST),
            v2.PILToTensor(),
            v2.ToDtype(torch.int64, scale=False),
            v2.Lambda(lambda x: x.squeeze(0)) 
        ])

        self.images_dir = os.path.join(
            root, "leftImg8bit_trainextra","leftImg8bit", "train_extra"
        )
        self.targets_dir = os.path.join(
            root, "gtCoarse", "train"
        )

        self.images = []

        for city in os.listdir(self.images_dir):
            city_img_dir = os.path.join(self.images_dir, city)
            for file in os.listdir(city_img_dir):
                if file.endswith("_leftImg8bit.png"):
                    self.images.append(os.path.join(city_img_dir, file))

        self.images.sort()


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.get_labels:
            target_path = (
                    img_path
                    .replace("leftImg8bit_trainextra/leftImg8bit", "gtCoarse")
                    .replace("_leftImg8bit.png", "_gtCoarse_labelIds.png")
                )

            target = Image.open(target_path)
            if self.target_transform:
                target = self.target_transform(target)

            return image, target
        else:
            return image
