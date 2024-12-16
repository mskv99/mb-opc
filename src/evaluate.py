import torch
import random
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os

from dataset import OPCDataset, TestDataset, BinarizeTransform
from torch.utils.data import DataLoader, Dataset

# fixing seeds during evaluation
def set_random_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

set_random_seed(42)

class EvalDataset(Dataset):
  def __init__(self, target_dir, mask_dir, transform=None):
    self.target_dir = target_dir
    self.mask_dir = mask_dir
    self.transform = transform
    self.images = os.listdir(target_dir)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    target_path = os.path.join(self.target_dir, self.images[index])
    mask_path = os.path.join(self.mask_dir, self.images[index])

    target = Image.open(target_path)
    mask = Image.open(mask_path)

    if self.transform:
      target = self.transform(target)
      mask = self.transform(mask)

    return target, mask

def calculate_iou(eval_loader):
  iou_list = []

  for target_batch, mask_batch in eval_loader:
    intersection = (mask_batch * target_batch).sum() #.sum(dim=(2, 3))
    union = (mask_batch + target_batch).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    iou_list.append(iou)
  print(f'Average IoU: {sum(iou_list) / len(eval_loader)}')

def calculate_pixel_accuracy(eval_loader):
  pixel_acc_list = []
  for target_batch, mask_batch in eval_loader:
    correct = (target_batch == mask_batch).float().sum()
    total = torch.numel(target_batch)
    pixel_acc = correct / total
    pixel_acc_list.append(pixel_acc)
  print(f'Average Pixel Accuracy: {sum(pixel_acc_list) / len(eval_loader)}')

TRANSFORM = transforms.Compose([
  transforms.Resize((1024, 1024)),
  transforms.ToTensor(),
  transforms.Grayscale(),
  BinarizeTransform(threshold=0.5)
])

TARGET_PATH = 'data/processed/gds_dataset/correction/test_correction'
MASK_PATH = 'inference/output_img/exp_6'
BATCH_SIZE = 2

# Dataset paths
EVAL_DATASET = EvalDataset(TARGET_PATH, MASK_PATH, transform = TRANSFORM)

# DataLoader
EVAL_LOADER = DataLoader(EVAL_DATASET, batch_size = BATCH_SIZE, shuffle = False)

calculate_iou(EVAL_LOADER)
calculate_pixel_accuracy(EVAL_LOADER)