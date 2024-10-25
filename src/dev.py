import cv2
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_sched
import matplotlib.pyplot as plt
import logging

from model import *
from dataset import *
import numpy as np
MODEL_PATH = 'checkpoints/exp_2/last_checkpoint.pth' #'/home/amoskovtsev/MBOPC/custom_unet/checkpoints/checkpoint_14.10.24.pth'
OUTPUT_DIR = 'output_img'
device = torch.device('cpu') # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

generator_model = Generator(in_ch = 3, out_ch = 3)
# generator_model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
generator_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))['model_state_dict'])
generator_model = generator_model.to(device)
generator_model.eval()
print('Model initialized:', generator_model)


def save_generated_image(output, epoch, step, checkpoint_dir="checkpoints", image_type='true_correction'):
  # # Convert from [-1, 1] to [0, 255] for saving

  #image = single_channel_tensor.numpy()
  #blurred_image = cv2.GaussianBlur(image, (5,5),0)
  #binarized_image = cv2.adaptiveThreshold(blurred_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

  # binarized_image = (binarized_image * 255).astype(np.uint8)
  #img_save_path = os.path.join(checkpoint_dir, f"{image_type}_epoch{epoch}_step{step}.png")
  #cv2.imwrite(f"{img_save_path}", binarized_image)
  #print(f"Saved generated image at {img_save_path}")



  # single_channel_tensor = output[0].mean(dim=0)
  # image = single_channel_tensor.numpy()
  # blurred_image = cv2.GaussianBlur(image, (7,7),0)
  # _, binarized_image = cv2.threshold(blurred_image, 0.5, 1.0, cv2.THRESH_BINARY)
  #
  # binarized_image = (binarized_image * 255).astype(np.uint8)
  # img_save_path = os.path.join(checkpoint_dir, f"{image_type}_epoch{epoch}_step{step}.png")
  # cv2.imwrite(f"{img_save_path}", binarized_image)
  # print(f"Saved generated image at {img_save_path}")


  single_image = torch.mean(output[0], axis=0)
  print(f'Single image shape:{single_image.shape}')
  single_image[single_image > 0.5] = 1.0
  single_image[single_image <= 0.5] = 0.0

  img_save_path = os.path.join(checkpoint_dir, f"{image_type}_epoch{epoch}_step{step}.png")
  cv2.imwrite(f"{img_save_path}", (single_image * 255).detach().cpu().numpy())
  # saved_img = Image.open(img_save_path)
  # saved_img.show()
  print(f"Saved generated image at {img_save_path}")


BATCH_SIZE = 1

# Dataset paths
TRAIN_DATASET = OPCDataset("/home/amoskovtsev/Загрузки/datasets/gds_dataset_new/origin/train_origin", "/home/amoskovtsev/Загрузки/datasets/gds_dataset_new/correction/train_correction", transform = transform)
VALID_DATASET = OPCDataset("/home/amoskovtsev/Загрузки/datasets/gds_dataset_new/origin/valid_origin", "/home/amoskovtsev/Загрузки/datasets/gds_dataset_new/correction/valid_correction", transform = transform)
TEST_DATASET = OPCDataset("/home/amoskovtsev/Загрузки/datasets/gds_dataset_new/origin/test_origin", "/home/amoskovtsev/Загрузки/datasets/gds_dataset_new/correction/test_correction", transform = transform)

# DataLoader
TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size = BATCH_SIZE, shuffle = True)
VALID_LOADER = DataLoader(VALID_DATASET, batch_size = BATCH_SIZE, shuffle = False)
TEST_LOADER = DataLoader(TEST_DATASET, batch_size = BATCH_SIZE, shuffle = True)

def iou_loss(pred, target, eps=1e-6):
  intersection = (pred * target).sum(dim=(2, 3))
  union = (pred + target).sum(dim=(2, 3)) - intersection
  iou = (intersection + eps) / (union + eps)
  return 1 - iou.mean()

image, target = next(iter(TEST_LOADER))
print(f'Image shape: {image.shape}')
print(f'Target shape: {target.shape}')
print(f'Image shape after removing batch dimension: {image[0,0].shape}')

print(image)



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

# Выполним проверку подсчёта функций потерь
# with torch.no_grad():
#   output = generator_model(image)

# print(f'Output shape: {output.shape}')
# iou_loss_value = iou_loss(target, output.sigmoid())
# mse_loss_value = torch.nn.functional.mse_loss(output.sigmoid(), target)

# print(output[0].shape, output[1].shape)
# print(f'IoU loss:{iou_loss_value}')
# print(f'L1-loss:{mse_loss_value}')
#
# save_generated_image(output.sigmoid(), epoch=0, step=0, checkpoint_dir=OUTPUT_DIR, image_type='generated_mask')

# for idx, (image, target) in enumerate(TEST_LOADER):
#   with torch.no_grad():
#     output = generator_model(image)
#     save_generated_image(output.sigmoid(), epoch=1, step=0, checkpoint_dir=OUTPUT_DIR, image_type=f'generated_mask_{idx}')

