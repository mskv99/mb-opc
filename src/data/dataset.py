from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
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

        return image

# Define transformations (resize to 1024x1024)
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

