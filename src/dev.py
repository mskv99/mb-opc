import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import matplotlib
import matplotlib.pyplot as plt

from models.model import Generator
from data.dataset import OPCDataset, TestDataset, BinarizeTransform
from config import DATASET_PATH, CHECKPOINT_PATH, BATCH_SIZE
from utils import BoundaryLoss, TVLoss

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
  # saved_img = Image.open(img_save_path)
  # saved_img.show()
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

# DataLoader
TEST_LOADER = DataLoader(TEST_DATASET, batch_size = BATCH_SIZE, shuffle = True)

def iou_loss(pred, target, eps=1e-6):
  intersection = (pred * target).sum(dim=(2, 3))
  union = (pred + target).sum(dim=(2, 3)) - intersection
  iou = (intersection + eps) / (union + eps)
  print(f'IoU shape:{iou.shape}')
  print(f'IoU value:{iou}')
  return 1 - iou.mean()

tv_loss = TVLoss(weight=1.0)
image, target = next(iter(TEST_LOADER))
image, target = image.to(DEVICE), target.to(DEVICE)
print(f'Image shape: {image.shape}')
print(f'Slice from image: {image[:1].shape}')
print(f'Target shape: {target.shape}')
print(f'Image shape after removing batch dimension: {image[0,0].shape}')

sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1,0,-1]], dtype = torch.float32, device = DEVICE).view(1,1,3,3)
sobel_y = torch.tensor([[1,2,1], [0,0,0], [-1, -2, -1]], dtype = torch.float32, device = DEVICE).view(1,1,3,3)
print(sobel_x)
print(sobel_x.shape)

# Calculating sobel edges for target
target_upsampled = torch.nn.functional.interpolate(target, scale_factor=2, mode='bilinear', align_corners=True)

target_edge_x = torch.nn.functional.conv2d(target_upsampled, sobel_x, padding = 1)
target_edge_y = torch.nn.functional.conv2d(target_upsampled, sobel_y, padding = 1)
target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)

target_edge_image = target_edge.squeeze().cpu().numpy()
plt.imshow(target_edge_image, cmap='gray')
#plt.colorbar()
plt.title('Tensor image')
plt.axis('off')
plt.savefig('data/external/target_sobel_edges.png')
# plt.show()

with torch.no_grad():
  pred = generator_model(image)
pred_upsampled = torch.nn.functional.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=True)
pred_edge_x = torch.nn.functional.conv2d(pred_upsampled.sigmoid(), sobel_x, padding = 1)
pred_edge_y = torch.nn.functional.conv2d(pred_upsampled.sigmoid(), sobel_y, padding = 1)
pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)

pred_edge_image = pred_edge.squeeze().cpu().numpy()
plt.imshow(pred_edge_image, cmap='gray')
#plt.colorbar()
plt.title('Tensor image')
plt.axis('off')
plt.savefig('data/external/pred_sobel_edges.png')
# plt.show()

boundary_diff = torch.abs(pred_edge - target_edge)
print(boundary_diff.mean())

tv_loss_pred = tv_loss(pred.sigmoid())
print(f'tv loss pred:{tv_loss_pred}')
tv_loss_target = tv_loss(target.sigmoid())
print(f'tv loss target:{tv_loss_target}')

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

