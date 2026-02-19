import os
import torch
from torch.utils.data import Dataset

class PreprocessedCityscapesDataset(Dataset):
    """
    Dataset class for loading preprocessed Cityscapes data.
    Much faster than loading and transforming raw images on-the-fly.
    """
    def __init__(self, root='./CityScapes/data/preprocessed_data', get_labels=False):
        self.root = root
        self.get_labels = get_labels

        self.images_dir = os.path.join(
            root, "leftImg8bit_trainextra", "leftImg8bit", "train_extra"
        )
        self.targets_dir = os.path.join(
            root, "gtCoarse", "train"
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

        # Load preprocessed tensor directly (much faster!)
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
