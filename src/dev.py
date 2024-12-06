import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import matplotlib
import matplotlib.pyplot as plt

from models.model import Generator
from dataset import OPCDataset, TestDataset, BinarizeTransform
from config import DATASET_PATH, CHECKPOINT_PATH, BATCH_SIZE
from utils import BoundaryLoss, TVLoss, ContourLoss, IouLoss

matplotlib.use('Agg')

MODEL_PATH = os.path.join(CHECKPOINT_PATH, 'exp_3/last_checkpoint.pth')
OUTPUT_DIR = 'inference/output_img'
DEVICE = torch.device('cuda:0')

generator_model = Generator(in_ch = 1, out_ch = 1)
# generator_model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
generator_model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE)['model_state_dict'])
generator_model = generator_model.to(DEVICE)
generator_model.eval()
print('Model initialized:', generator_model)


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

TRANSFORM = transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Grayscale(),
    BinarizeTransform(threshold=0.5)
])

TEST_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/test_origin'),
                          os.path.join(DATASET_PATH, 'correction/test_correction'),
                          transform = transform)

TEST_LOADER = DataLoader(TEST_DATASET, batch_size = BATCH_SIZE, shuffle = True)

# define loss functions
tv_loss = TVLoss(weight=1.0)
contour_loss = ContourLoss(weight=1.0, device=DEVICE)
mse_loss = torch.nn.MSELoss()
iou_loss = IouLoss(weight=1.0)

image, target = next(iter(TEST_LOADER))
image, target = image.to(DEVICE), target.to(DEVICE)
print(f'Image shape: {image.shape}')
print(f'Slice from image: {image[:1].shape}')
print(f'Target shape: {target.shape}')
print(f'Image shape after removing batch dimension: {image[0,0].shape}')

with torch.no_grad():
  pred = generator_model(image).sigmoid()

contour_loss_iter = contour_loss(pred, target)
mse_loss_iter = mse_loss(pred, target)
iou_loss_iter = iou_loss(pred, target)
tv_loss_pred = tv_loss(pred.sigmoid())
total_loss = mse_loss_iter + contour_loss_iter + iou_loss_iter
print(f'contour loss:{contour_loss_iter}')

print(f'mse loss:{mse_loss_iter}')
print(f'iou_loss:{iou_loss_iter}')
print(f'tv loss pred:{tv_loss_pred}')
print(total_loss)

# # 1. Find the maximum values across the last two dimensions
# max_across_width, _ = image.max(dim=(-1))
# max_values, _ = max_across_width.max(dim=-1)
# print("Maximum values across last two dimensions:")
# print(max_values)
# print("Shape of max values:", max_values.shape)  # Expected shape: [1, 3]
#
# # 2. Find unique values across the last two dimensions
# reshaped_tensor = image.view(image.size(0), image.size(1), -1)  # Shape: [1, 3, 1024*1024]
# unique_values = torch.unique(reshaped_tensor)
#
# print("Unique values across last two dimensions:")
# print(unique_values)
# print("Number of unique values:", unique_values.numel())
#
# # Выполним проверку подсчёта функций потерь
# with torch.no_grad():
#   output = generator_model(image)
#
# print(f'Output shape: {output.shape}')
#
# boundary_loss = BoundaryLoss(weight=1.0, device = DEVICE)
# iou_loss_value = iou_loss(target, output.sigmoid())
# mse_loss_value = torch.nn.functional.mse_loss(output.sigmoid(), target)
# boundary_loss_value = boundary_loss(output.sigmoid(), target)
#
# print(output.shape)
# print(f'IoU loss:{iou_loss_value}')
# print(f'L2-loss:{mse_loss_value}')
# print(f'Boundary loss:{boundary_loss_value}')
#
# print(f'L2-loss shape:{mse_loss_value.shape}')
# print(f'IoU-loss shape:{iou_loss_value.shape}')
# #print(f'Boundary loss shape:{boundary_loss.shape}')
#
# total_loss = mse_loss_value + iou_loss_value + boundary_loss_value
# total_loss.backward()

#save_generated_image(output.sigmoid(), epoch=0, step=0, checkpoint_dir=OUTPUT_DIR, image_type='generated_test')

# for idx, (image, target) in enumerate(TEST_LOADER):
#   with torch.no_grad():
#     output = generator_model(image)
#     save_generated_image(output.sigmoid(), epoch=1, step=0, checkpoint_dir=OUTPUT_DIR, image_type=f'generated_mask_{idx}')

