# src/dataset.py
# PyTorch Dataset for the Pokémon image classification task.

import os
from PIL import Image
from torch.utils.data import Dataset


class PokemonDataset(Dataset):
    """
    Loads images from a directory structure:
        data_dir/
            class_name_1/
                img1.png
                img2.png
            class_name_2/
                ...
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []      # list of (image_path, label_index)
        self.class_names = []  # sorted list of class names
        self.class_to_idx = {}

        self._scan_directory()

    def _scan_directory(self):
        """Walk the data directory and build the sample list."""
        self.class_names = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_path = os.path.join(self.data_dir, class_name)
            for filename in os.listdir(class_path):
                filepath = os.path.join(class_path, filename)
                if self._is_image(filepath):
                    self.samples.append((filepath, self.class_to_idx[class_name]))

    @staticmethod
    def _is_image(path):
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label