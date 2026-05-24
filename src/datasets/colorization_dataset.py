import os
import torch
import numpy as np
from torch.utils.data import Dataset

class ColorizationDataset(Dataset):

    def __init__(self, x_dir, y_dir,file_path):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.files = []
        with open(file_path, 'r') as f:
            self.files = [line.strip() for line in f]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_name = self.files[idx]

        x = np.load(os.path.join(self.x_dir, file_name))
        y = np.load(os.path.join(self.y_dir, file_name))

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        x = x.unsqueeze(0)

        return x, y