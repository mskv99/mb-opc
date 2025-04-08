import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

from models.unet import Generator
from dataset import OPCDataset, TestDataset, BinarizeTransform
from config import DATASET_PATH, CHECKPOINT_PATH
from utils import BoundaryLoss, TVLoss, ContourLoss, IouLoss, PixelAccuracy

matplotlib.use('Agg')

MODEL_PATH = os.path.join(CHECKPOINT_PATH, 'exp_22/last_checkpoint.pth')
OUTPUT_DIR = 'data/external'
DEVICE = torch.device('cuda:0')
CHECK_DIMENSIONS = True
SAVE_PREDICTION = True
BATCH_SIZE = 1

generator_model = Generator(in_ch=1, out_ch=1)
# generator_model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
generator_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict'])
generator_model = generator_model.to(DEVICE)
generator_model.eval()
print('Model initialized:', generator_model)


def set_random_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)


set_random_seed(42)


def save_generated_image(output, epoch, step, checkpoint_dir="checkpoints", image_type='true_correction'):
  '''
  Example with applying blurring as a postprocessing step:
  single_channel_tensor = output[0].mean(dim=0)
  image = single_channel_tensor.numpy()
  blurred_image = cv2.GaussianBlur(image, (7,7),0)
  _, binarized_image = cv2.threshold(blurred_image, 0.5, 1.0, cv2.THRESH_BINARY)
  '''
  single_image = output[0].squeeze(dim=0)
  print(f'Single image shape:{single_image.shape}')
  single_image[single_image > 0.5] = 1.0
  single_image[single_image <= 0.5] = 0.0

  img_save_path = os.path.join(checkpoint_dir, f"{image_type}_epoch{epoch}_step{step}.jpg")
  cv2.imwrite(f"{img_save_path}", (single_image * 255).detach().cpu().numpy())
  print(f"Saved generated image at {img_save_path}")


TRANSFORM = transforms.Compose([
  transforms.Resize((1024, 1024)),
  transforms.ToTensor(),
  transforms.Grayscale(),
  BinarizeTransform(threshold=0.5)
])

TEST_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/test_origin'),
                          os.path.join(DATASET_PATH, 'correction/test_correction'),
                          transform=TRANSFORM)

TEST_LOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False)

# define loss functions
tv_loss = TVLoss(weight=1.0)
contour_loss = ContourLoss(weight=1.0, device=DEVICE)
mae_loss = torch.nn.L1Loss()
iou_loss = IouLoss(weight=1.0)
pixel_acc = PixelAccuracy()
bce = torch.nn.BCELoss()
boundary_loss = BoundaryLoss(device=DEVICE)

image, target = next(iter(TEST_LOADER))
image, target = image.to(DEVICE), target.to(DEVICE)
print(f'Image shape: {image.shape}')
print(f'Slice from image: {image[:1].shape}')
print(f'Target shape: {target.shape}')
print(f'Image shape after removing batch dimension: {image[0, 0].shape}')

with torch.no_grad():
  pred = generator_model(image).sigmoid()

cv2.imwrite(f"data/external/conf1.jpg", (image.squeeze() * 255).detach().cpu().numpy())
contour_loss_iter = contour_loss(pred, target)
mae_loss_iter = mae_loss(pred, target)
iou_loss_iter = iou_loss(pred, target)
tv_loss_pred = tv_loss(pred.sigmoid())
bce_loss = bce(pred, target)
bd_loss = boundary_loss(pred, target)
total_loss = mae_loss_iter + contour_loss_iter + iou_loss_iter

print(f'contour loss:{contour_loss_iter}')
print(f'mse loss:{mae_loss_iter}')
print(f'iou_loss:{iou_loss_iter}')
print(f'tv loss pred:{tv_loss_pred}')
print(f'bce loss:{bce_loss}')
print(f'Boundary loss:{bd_loss}')

pixel_acc = PixelAccuracy()
accuracy = pixel_acc(pred, target)
print(f'Pixel Accuracy: {accuracy}')

if CHECK_DIMENSIONS:
  # 1. Find the maximum values across the last two dimensions
  max_across_width, _ = image.max(dim=(-1))
  max_values, _ = max_across_width.max(dim=-1)
  print("Maximum values across last two dimensions:")
  print(max_values)
  print("Shape of max values:", max_values.shape)  # Expected shape: [1, 3]

  # 2. Find unique values across the last two dimensions
  reshaped_tensor = image.view(image.size(0), image.size(1), -1)  # Shape: [1, 3, 1024*1024]
  unique_values = torch.unique(reshaped_tensor)

  print("Unique values across last two dimensions:")
  print(unique_values)
  print("Number of unique values:", unique_values.numel())

if SAVE_PREDICTION:
  save_generated_image(pred, epoch=0, step=0, checkpoint_dir=OUTPUT_DIR, image_type='generated_test')
