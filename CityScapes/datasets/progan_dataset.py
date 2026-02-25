import os
import torch
from torch.utils.data import Dataset


class ProGanCityscapesDataset(Dataset):
    """
    Dataset class for loading preprocessed Cityscapes data.

    Folder structure expected:
    root/
        activation/
            sizeHxW/
                leftImg8bit_trainextra/leftImg8bit/train_extra
                gtCoarse/train
    """

    def __init__(self,
                 root='./CityScapes/data/progan',
                 activation='tanh',
                 size=256,
                 get_labels=False):

        self.root = root
        self.activation = activation
        self.get_labels = get_labels

        # Convert int size to (H, W)
        if isinstance(size, int):
            size = (size, size)

        size_folder = f"{size[0]}x{size[1]}"

        # Build paths according to new preprocessing logic
        base_dir = os.path.join(root, activation, size_folder)

        self.images_dir = os.path.join(
            base_dir,
            "leftImg8bit_trainextra",
            "leftImg8bit",
            "train_extra"
        )

        self.targets_dir = os.path.join(
            base_dir,
            "gtCoarse",
            "train"
        )

        if not os.path.exists(self.images_dir):
            raise ValueError(
                f"Directory not found: {self.images_dir}\n"
                f"Check activation='{activation}' and size={size}"
            )

        self.images = []

        for city in os.listdir(self.images_dir):
            city_img_dir = os.path.join(self.images_dir, city)
            if not os.path.isdir(city_img_dir):
                continue
            for file in os.listdir(city_img_dir):
                if file.endswith("_leftImg8bit.pt"):
                    self.images.append(os.path.join(city_img_dir, file))

        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Load preprocessed tensor directly
        image = torch.load(img_path, weights_only=False)

        if self.get_labels:
            target_path = (
                img_path
                .replace("leftImg8bit_trainextra/leftImg8bit", "gtCoarse")
                .replace("_leftImg8bit.pt", "_gtCoarse_labelIds.pt")
            )

            target = torch.load(target_path, weights_only=False)
            return image, target
        else:
            return image