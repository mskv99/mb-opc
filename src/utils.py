import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torchvision import models
import os

def compute_distance_map(mask):
  '''
  Compute dostance map for the binary mask.
  For each pixel, the distance map gives a distance to the nearest boundary
  :param mask:
  :return:
  '''
  mask = mask.cpu().numpy()
  distance_map = distance_transform_edt(mask) + distance_transform_edt(1 - mask)

  return torch.tensor(distance_map).float()

class BoundaryLoss(nn.Module):
  def __init__(self, weight=1.0, device='cpu'):
    super(BoundaryLoss, self).__init__()
    self.weight = weight
    self.device = device
  def forward(self, pred, target):
    # Convert target to ddistance maps
    dist_maps = torch.stack([compute_distance_map(t) for t in target]).to(self.device)
    # Compute boundary loss
    boundary_loss = torch.mean(pred * dist_maps)

    return torch.from_numpy(self.weight * boundary_loss)

class ContourLoss(nn.Module):
  def __init__(self, weight=1.0, device='cpu'):
    super(ContourLoss, self).__init__()
    self.weight = weight
    self.device = device
    self.sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1,0,-1]], dtype = torch.float32, device = device).view(1,1,3,3)
    self.sobel_kernel_y = torch.tensor([[1,2,1], [0,0,0], [-1, -2, -1]], dtype = torch.float32, device = device).view(1,1,3,3)
  def forward(self, pred, target):
    target_edge_x = torch.nn.functional.conv2d(target, self.sobel_kernel_x, padding = 1)
    target_edge_y = torch.nn.functional.conv2d(target, self.sobel_kernel_y, padding = 1)
    target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)

    pred_edge_x = torch.nn.functional.conv2d(pred, self.sobel_kernel_x, padding = 1)
    pred_edge_y = torch.nn.functional.conv2d(pred, self.sobel_kernel_y, padding = 1)
    pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)

    # calculate difference for every predicted and target correction in a batch
    loss_per_image = torch.abs(pred_edge - target_edge).sum(dim=(1,2,3)) # (N, ), sum over height, width and channels
    # calculate the contour area for every target correction in a batch
    target_edge_sum = target_edge.sum(dim=(1,2,3)) + 1e-6 # (N, ) sum over height, width, and prevent division by zero
    # normalise the difference by the contour area of target correction
    mean_loss = (loss_per_image / target_edge_sum) * self.weight

    # return mean value over a batch
    return mean_loss.mean()  # (),

class TVLoss(nn.Module):
  def __init__(self, weight=1.0):
    super(TVLoss, self).__init__()
    self.weight = weight
  def forward(self, x):
    tv_loss = torch.sum(torch.abs(x[:,:,1:,:] - x[:, :, :-1, :])) + torch.sum(torch.abs(x[:,:,:,1:] - x[:, :, :, :-1]))

    return self.weight * tv_loss

class PerceptualLoss(nn.Module):
  def __init__(self, weight=1.0, layers=None):
    super(PerceptualLoss, self).__init__()
    self.weight = weight
    # Load pretrained VGG model and select specific layers
    vgg = models.vgg19(weights='VGG19_Weights.DEFAULT').features
    self.layers = layers if layers else [0,5,10,19, 28] # use diferent layers for multiscale features
    self.features = nn.ModuleList([vgg[i] for i in self.layers]).eval()

    # Freeze vgg parameters
    for param in self.features.parameters():
      param.requires_grad = False

  def forward(self, pred, target):
    pred, target = self.normalize_input(pred), self.normalize_input(target)
    perceptual_loss = 0.0
    for layer in self.features:
      pred, target = layer(pred), layer(target)
      perceptual_loss += F.l1_loss(pred, target)

    return self.weight * perceptual_loss

  @staticmethod
  def normalize_input(x):
    # Normalize according to VGG preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(x.device)
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]

    return x

class IouLoss(nn.Module):
  def __init__(self, weight=1.0, eps=1e-6):
    super(IouLoss, self).__init__()
    self.weight = weight
    self.eps = eps
  def forward(self, pred, target):
    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - intersection
    iou = (intersection + self.eps) / (union + self.eps)
    return (1 - iou.mean()) * self.weight

class IoU(nn.Module):
  def __init__(self, eps=1e-6):
    super(IoU, self).__init__()
    self.eps = eps

  def forward(self, pred, target):
    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - intersection
    iou = (intersection + self.eps) / (union + self.eps)

    return iou.mean()

class PixelAccuracy(nn.Module):
  def __init__(self, eps=1e-6):
    super(PixelAccuracy, self).__init__()
    self.eps = eps

  def forward(self, pred, target):
    pred[pred > 0.5] = 1.0
    pred[pred <= 0.5] = 0.0
    correct = (pred == target).float().sum()
    print(f'Correct shape:{correct.shape}')
    total = torch.numel(target)
    pixel_acc = (correct + self.eps) / (total + self.eps)

    return pixel_acc


def get_next_experiment_folder(checkpoints_dir):
  # Ensure the checkpoints directory exists
  if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

  # Find the next available experiment number
  exp_number = 1
  while True:
    exp_folder = os.path.join(checkpoints_dir, f'exp_{exp_number}')
    if not os.path.exists(exp_folder):
      os.makedirs(exp_folder)
      return exp_folder
    exp_number += 1

def next_exp_folder(checkpoints_dir):
  if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
  dir_list = os.listdir(checkpoints_dir)
  give_numb = lambda x: int(x.split('_')[-1])
  dir_numbers = [give_numb(name) for name in dir_list if not name.endswith('.gitkeep')]
  max_number = max(dir_numbers)
  new_exp_folder = os.path.join(checkpoints_dir, f'exp_{max_number + 1}')
  os.makedirs(new_exp_folder)
  return new_exp_folder


'''
for images, targets in dataloader:
  preds = model(images)
  loss_tv = tv_loss(preds)
  loss_boundary = boundary_loss(preds, targets)
  loss_perceptual = perceptual_loss(preds, targets)
  
  # Total loss
  total_loss = loss_tv + loss_boundary + loss_perceptual
'''
