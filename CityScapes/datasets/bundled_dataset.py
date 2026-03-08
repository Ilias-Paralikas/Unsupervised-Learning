import os 
import torch
from torch.utils.data import Dataset

class MSGBundledDataset(Dataset):
    def __init__(self, root='./CityScapes/data/progan', activation='tanh'):
        self.bundled_dir = os.path.join(root, activation, "bundled")
        self.files = []
        
        # Collect all bundled file paths
        for city in os.listdir(self.bundled_dir):
            city_path = os.path.join(self.bundled_dir, city)
            if not os.path.isdir(city_path): continue
            for f in os.listdir(city_path):
                if f.endswith(".pt"):
                    self.files.append(os.path.join(city_path, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Single Disk Load (Efficiency King)
        bundle = torch.load(self.files[idx], weights_only=False)
        
        return bundle
    



