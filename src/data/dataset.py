from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os

class OPCDataset(Dataset):
    def __init__(self, image_dir, target_dir, transform=None):
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        target_path = os.path.join(self.target_dir, self.images[index])

        image = Image.open(img_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_path

class BinarizeTransform:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, tensor):
        # Assuming the input tensor is of shape (1, 1, 1024, 1024)
        # Convert the tensor to a NumPy array
        image_array = tensor.squeeze().numpy()  # Shape will be (1024, 1024)

        # Binarize the image
        binarized_image = (image_array > self.threshold).astype(np.float32)  # Binary (0 or 1)

        # Convert back to tensor and maintain shape (1, 1, 1024, 1024)
        return torch.from_numpy(binarized_image).unsqueeze(0)  # Shape will be (1, 1, 1024, 1024)

