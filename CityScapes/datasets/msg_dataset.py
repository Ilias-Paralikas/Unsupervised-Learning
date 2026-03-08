import os
import torch
from torch.utils.data import Dataset

class CityScapesMultiScaleDataset(Dataset):
    def __init__(self, 
                 preprocessed_root='./CityScapes/data/msg', 
                 output_activation="tanh", 
                 get_labels=False):
        """
        Args:
            preprocessed_root: Path to the './preprocessed_data' folder.
            output_activation: "tanh" or "sigmoid" (matches the folder structure).
            get_labels: If False, labels are not loaded from disk (saves time).
        """
        self.get_labels = get_labels
        
        # Build the base path to the images
        self.base_img_dir = os.path.join(
            preprocessed_root, 
            output_activation, 
            "all_sizes", 
            "leftImg8bit_trainextra", 
            "leftImg8bit", 
            "train_extra"
        )
        
        # Collect all image .pt files
        self.image_paths = []
        for city in os.listdir(self.base_img_dir):
            city_dir = os.path.join(self.base_img_dir, city)
            if not os.path.isdir(city_dir):
                continue
            for file in os.listdir(city_dir):
                if file.endswith("_leftImg8bit.pt"):
                    self.image_paths.append(os.path.join(city_dir, file))
        
        self.image_paths.sort()
        print(f"Dataset initialized with {len(self.image_paths)} samples. Labels enabled: {self.get_labels}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        
        # Load the list of image tensors [size1, size2, ...]
        # weights_only=True is recommended for security if you're only loading tensors/lists
        image_list = torch.load(img_path, weights_only=False)
        
        target = []
        if self.get_labels:
            # Construct target path by swapping directory and suffix
            target_path = (
                img_path
                .replace("leftImg8bit_trainextra/leftImg8bit", "gtCoarse")
                .replace("_leftImg8bit.pt", "_gtCoarse_labelIds.pt")
            )
            
            # Load the list of label tensors [size1, size2, ...]
            target = torch.load(target_path, weights_only=False)

        return image_list, target
    
