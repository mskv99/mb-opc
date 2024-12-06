from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import DATASET_PATH
from tqdm import tqdm
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

def calculate_mean_std(data_loader):
    total_pixels = 0.0
    total_sum = 0.0
    total_sum_squared = 0.0

    for images, _ in tqdm(data_loader):
        # Assuming the shape of images: (batch_size, 1, H, W)
        total_pixels += images.numel()
        total_sum += images.sum().item()
        total_sum_squared += (images ** 2).sum().item()
    # Compute mean and standard deviation
    mean = total_sum / total_pixels
    std = (total_sum_squared / total_pixels - mean ** 2) ** 0.5

    return round(mean, 5), round(std, 5)

def apply_transform(mean=0, std=0):

    if mean == 0 or std == 0:
        TRANSFORM = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Grayscale(),
            BinarizeTransform(threshold=0.5)])
    else:
        TRANSFORM = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Grayscale(),
            BinarizeTransform(threshold=0.5),
            transforms.Normalize(mean=[mean], std=[std])])

    return TRANSFORM


if __name__ == '__main__':

    TRAIN_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/train_origin'),
                               os.path.join(DATASET_PATH, 'correction/train_correction'), transform = apply_transform())
    VALID_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/valid_origin'),
                               os.path.join(DATASET_PATH, 'correction/valid_correction'), transform = apply_transform())
    # Define dataloader
    TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size = 1, shuffle = True, num_workers = 2)
    VALID_LOADER = DataLoader(VALID_DATASET, batch_size = 1, shuffle = True, num_workers = 2)

    print(f'Number of images in train subset:{len(TRAIN_DATASET)}\n')
    print(f'Number of images in valid subset:{len(VALID_DATASET)}\n')

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_mean, train_std = calculate_mean_std(TRAIN_LOADER)
    valid_mean, valid_std = calculate_mean_std(VALID_LOADER)

    print(f'Train Dataset Mean:{train_mean}, Std:{train_std}')
    print(f'Valid Dataset Mean:{valid_mean}, Std:{valid_std}')

    image, target = next(iter(TRAIN_LOADER))
    image, target = image.to(DEVICE), target.to(DEVICE)
    print(f'Image shape: {image.shape}')
    print(f'Target shape: {target.shape}')
    print(f'Image shape after removing batch dimension: {image[0,0].shape}')



    NORM_TRAIN_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/train_origin'),
                               os.path.join(DATASET_PATH, 'correction/train_correction'), transform = apply_transform(mean = train_mean, std = train_std))

    NORM_TRAIN_LOADER = DataLoader(NORM_TRAIN_DATASET, batch_size = 1, shuffle = True, num_workers = 2)

    image, target = next(iter(NORM_TRAIN_LOADER))

    print(f'Image mean: {image.mean()}, image std: {image.std()}')

    new_train_mean, new_train_std = calculate_mean_std(NORM_TRAIN_LOADER)

    print(f'New train dataset mean:{new_train_mean}, new Std: {new_train_std}')





