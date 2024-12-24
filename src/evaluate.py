import torch
import random
from PIL import Image
import torchvision.transforms as transforms
import logging
from tqdm import tqdm
import numpy as np
import os

from dataset import OPCDataset, TestDataset, BinarizeTransform, apply_transform
from torch.utils.data import DataLoader, Dataset
from config import DATASET_PATH, CHECKPOINT_PATH
from models.unet import Generator
from utils import IoU, PixelAccuracy

# fixing seeds during evaluation
def set_random_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

set_random_seed(42)

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

def evaluate_model(model, loader, device='cuda', log=False):
  model.eval()

  pixel_acc_epoch = 0
  iou_epoch = 0

  with torch.no_grad():
    for idx, (image, target) in tqdm(enumerate(loader)):
      image, target = image.to(device), target.to(device)
      params = model(image)
      mask = torch.sigmoid(params)

      # calculating metrics for evaluation
      pixel_acc_iter = pixel_accuracy(mask, target)
      iou_iter = iou(mask, target)

      pixel_acc_epoch += pixel_acc_iter.item()
      iou_epoch += iou_iter.item()

    log_info = {
      'pixel_acc': pixel_acc_epoch,
      'iou': iou_epoch,
      'len_loader': len(loader)
    }

    # Print and log the message
  print(f"Pixel Accuracy: {log_info['pixel_acc'] / log_info['len_loader'] :.4f}, ")
  print(f"IoU: {log_info['iou'] / log_info['len_loader'] :.4f}")

  if log:
    logging.info(f"Pixel Accuracy: {log_info['pixel_acc'] / log_info['len_loader'] :.4f}, ")
    logging.info(f"IoU: {log_info['iou'] / log_info['len_loader'] :.4f}")



  return log_info['pixel_acc'] / log_info['len_loader'], log_info['iou'] / log_info['len_loader']

if __name__ == '__main__':
  MODEL_PATH = os.path.join(CHECKPOINT_PATH, 'exp_9/last_checkpoint.pth')
  BATCH_SIZE = 2
  DEVICE = 'cuda'

  TRAIN_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/train_origin'), os.path.join(DATASET_PATH,'correction/train_correction'), transform = apply_transform(binarize_flag = True))
  VALID_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/valid_origin'), os.path.join(DATASET_PATH, 'correction/valid_correction'), transform = apply_transform(binarize_flag = True))
  TEST_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/test_origin'), os.path.join(DATASET_PATH, 'correction/test_correction'), transform = apply_transform(binarize_flag = True))

  # Define dataloader
  TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
  VALID_LOADER = DataLoader(VALID_DATASET, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
  TEST_LOADER = DataLoader(TEST_DATASET, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)

  generator_model = Generator(in_ch = 1, out_ch = 1)
  generator_model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE)['model_state_dict'])
  generator_model = generator_model.to(DEVICE)
  pixel_accuracy = PixelAccuracy()
  iou = IoU()
  # just in case we want to store the output in txt file and
  # test the function implementation
  logging.basicConfig(filename = 'log.txt',datefmt = '%d/%m/%Y %H:%M')
  evaluate_model(model = generator_model, loader=TEST_LOADER, device=DEVICE, log=False)