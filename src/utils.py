import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torchvision import models

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

boundary_loss = BoundaryLoss(weight=1.0)
perceptual_loss = PerceptualLoss(weight=0.5)

'''
for images, targets in dataloader:
  preds = model(images)
  loss_tv = tv_loss(preds)
  loss_boundary = boundary_loss(preds, targets)
  loss_perceptual = perceptual_loss(preds, targets)
  
  # Total loss
  total_loss = loss_tv + loss_boundary + loss_perceptual
'''