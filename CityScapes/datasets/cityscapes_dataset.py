import os
import torch
from torch.utils.data import Dataset

class CityscapesDataset(Dataset):
    def __init__(self, data_dir = 'CityScapes/data/stylegan/tanh/256x256', get_labels=True):
        """
        Args:
            data_dir (str): Path to the specific size/activation directory 
                            (e.g., './preprocessed_data/tanh/512x512')
            get_labels (bool): Whether to load and return the target labels
        """
        self.data_dir = data_dir
        self.get_labels = get_labels
        
        # Base directory for the preprocessed images
        self.images_dir = os.path.join(
            self.data_dir, 
            "leftImg8bit_trainextra", 
            "leftImg8bit", 
            "train_extra"
        )
        
        # Collect all image tensor paths
        self.image_paths = []
        if os.path.exists(self.images_dir):
            for city in os.listdir(self.images_dir):
                city_path = os.path.join(self.images_dir, city)
                if os.path.isdir(city_path):
                    for file in os.listdir(city_path):
                        if file.endswith("_leftImg8bit.pt"):
                            self.image_paths.append(os.path.join(city_path, file))
                            
        self.image_paths.sort()
        
        if len(self.image_paths) == 0:
            print(f"Warning: No image tensors found in {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load the preprocessed image tensor directly (No longer in a list)
        image = torch.load(img_path, weights_only=False)
        
        if self.get_labels:
            # Construct target path by swapping directory and suffix
            target_path = (
                img_path
                .replace("leftImg8bit_trainextra/leftImg8bit", "gtCoarse")
                .replace("_leftImg8bit.pt", "_gtCoarse_labelIds.pt")
            )
            
            # Load the target tensor if it exists
            if os.path.exists(target_path):
                target = torch.load(target_path, weights_only=False)
            else:
                # Return a dummy tensor if missing so the DataLoader doesn't crash during batching
                target = torch.empty(0) 
        else:
            target = torch.empty(0)

        return image, target