from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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


def apply_transform(binarize_flag=False, normalize_flag=False, mean=0, std=0):
  '''
  case 1: working with non-binary and non-normalized grayscale images, pixel values lie within the range [0,1]

  case 2: working with binary non-normalized grayscale images, pixel values lie withinh the range [0,1]
  the most preferable variant for dataset preparation

  case 3: working with non-binary normalized grayscale images, pixel values MAY NOT lie within the range [0,1]
  this variant is not desirable when using a sigmoid as an activation function in the last layer of the Generator
  we must ensure that the prediction values will have the same range as the preprocessed data
  this configs might be Ok, when out input data not necessarily lies within the range [0,1]

  After applying on a single image we we can get something like:
  Img_mean ~ 0, img_std ~ 1, img_min < 0, img_max > 1

  Note: it does not make sense to apply both Normalization and Binarization:
  (a) normalize -> binarize - after applying binarization in the end we get mean and standard deviation different
  from (0) and (1) for our dataset
  (b) binarize -> normalize - after applying normalization in the end we get mean and standard devation equal to
   (0) and (1 )but the data is non-binary. not desirable if we use sigmoid in the last layer
  '''

  if ((not binarize_flag) and (not normalize_flag)):
    TRANSFORM = transforms.Compose([
      transforms.Resize((1024, 1024)),
      transforms.ToTensor(),
      transforms.Grayscale()])

  elif ((binarize_flag) and (not normalize_flag)):
    TRANSFORM = transforms.Compose([
      transforms.Resize((1024, 1024)),
      transforms.ToTensor(),
      transforms.Grayscale(),
      BinarizeTransform(threshold=0.5)])

  elif ((not binarize_flag) and (normalize_flag)):
    TRANSFORM = transforms.Compose([
      transforms.Resize((1024, 1024)),
      transforms.ToTensor(),
      transforms.Grayscale(),
      transforms.Normalize(mean=[mean], std=[std])])

  return TRANSFORM


if __name__ == '__main__':
  from config import DATASET_PATH

  TRAIN_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/train_origin'),
                             os.path.join(DATASET_PATH, 'correction/train_correction'), transform=apply_transform())
  VALID_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/valid_origin'),
                             os.path.join(DATASET_PATH, 'correction/valid_correction'), transform=apply_transform())
  # Define dataloader
  TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=1, shuffle=False, num_workers=2)
  VALID_LOADER = DataLoader(VALID_DATASET, batch_size=1, shuffle=False, num_workers=2)

  print(f'Number of images in train subset:{len(TRAIN_DATASET)}')
  print(f'Number of images in valid subset:{len(VALID_DATASET)}\n')

  DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  SAVE_IMAGE = False

  # train_mean, train_std = calculate_mean_std(TRAIN_LOADER)
  valid_mean, valid_std = calculate_mean_std(VALID_LOADER)

  # print(f'Train Dataset Mean:{train_mean}, Std:{train_std}')
  print(f'Valid Dataset Mean:{valid_mean}, Std:{valid_std}')

  image, target = next(iter(TRAIN_LOADER))
  image, target = image.to(DEVICE), target.to(DEVICE)
  print(f'Image shape: {image.shape}')
  print(f'Target shape: {target.shape}')
  print(f'Image shape after removing batch dimension: {image[0, 0].shape}')
  print(f'Image mean value: {image.mean()}, image std value: {image.std()}')
  print(f"Image min value: {image.min()}, Image max value: {image.max()}")

  NORM_VALID_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/valid_origin'),
                                  os.path.join(DATASET_PATH, 'correction/valid_correction'),
                                  transform=apply_transform(normalize_flag=True, mean=valid_mean, std=valid_std))
  NORM_VALID_LOADER = DataLoader(NORM_VALID_DATASET, batch_size=1, shuffle=False, num_workers=2)

  norm_image, norm_target = next(iter(NORM_VALID_LOADER))

  print(f'Image mean value: {norm_image.mean()}, image std value: {norm_image.std()}')
  print(f"Image min value: {norm_image.min()}, Image max value: {norm_image.max()}")

  new_valid_mean, new_valid_std = calculate_mean_std(NORM_VALID_LOADER)

  print(f'New valid dataset mean:{new_valid_mean}, new Std: {new_valid_std}')

  if SAVE_IMAGE:
    plt.imshow(image[0, 0].cpu().numpy(), cmap='gray')
    plt.savefig('data/external/testing_norm.jpg')
