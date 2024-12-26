import os
import sys

sys.path.append(".")
import math
import time
import json
import random
import pickle

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from torchinfo import summary

def conv2d(chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False):
  layers = []
  layers.append(nn.Conv2d(chIn, chOut, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
  if norm:
    layers.append(nn.BatchNorm2d(chOut, affine=bias))
  if relu:
    layers.append(nn.ReLU())
  return nn.Sequential(*layers)

def deconv2d(chIn, chOut, kernel_size, stride, padding, output_padding, bias=True, norm=True, relu=False):
  layers = []
  layers.append(nn.ConvTranspose2d(chIn, chOut, kernel_size=kernel_size, stride=stride, padding=padding,
                                   output_padding=output_padding, bias=bias))
  if norm:
    layers.append(nn.BatchNorm2d(chOut, affine=bias))
  if relu:
    layers.append(nn.ReLU())
  return nn.Sequential(*layers)

def sepconv2d(chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False):
  layers = []
  layers.append(nn.Conv2d(chIn, chOut, groups=chIn, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
  if norm:
    layers.append(nn.BatchNorm2d(chOut, affine=bias))
  if relu:
    layers.append(nn.ReLU())
  return nn.Sequential(*layers)

def repeat2d(n, chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False):
  layers = []
  for idx in range(n):
    layers.append(
      nn.Conv2d(chIn if idx == 0 else chOut, chOut, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    if norm:
      layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu:
      layers.append(nn.ReLU())
  return nn.Sequential(*layers)

def spsr(r, chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False):
  layers = []
  layers.append(nn.Conv2d(chIn, chOut * (r ** 2), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
  layers.append(nn.PixelShuffle(r))
  if norm:
    layers.append(nn.BatchNorm2d(chOut, affine=bias))
  if relu:
    layers.append(nn.ReLU())
  return nn.Sequential(*layers)

def linear(chIn, chOut, bias=True, norm=True, relu=False):
  layers = []
  layers.append(nn.Linear(chIn, chOut, bias=bias))
  if norm:
    layers.append(nn.BatchNorm1d(chOut, affine=bias))
  if relu:
    layers.append(nn.ReLU())
  return nn.Sequential(*layers)

def split(x, size=16):
  return rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=size, s2=size)

def apply_complex(fr, fi, input):
  return torch.complex(fr(input.real) - fi(input.imag), fr(input.imag) + fi(input.real))

class ComplexLinear(nn.Module):

  def __init__(self, in_features, out_features):
    super(ComplexLinear, self).__init__()
    self.fc_r = nn.Linear(in_features, out_features)
    self.fc_i = nn.Linear(in_features, out_features)

  def forward(self, input):
    return apply_complex(self.fc_r, self.fc_i, input)

class CFNO(nn.Module):
  def __init__(self, c=1, d=16, k=16, s=1, size=(128, 128)):
    super().__init__()
    self.c = c
    self.d = d
    self.k = k
    self.s = s
    self.size = size
    self.fc = ComplexLinear(self.c * (self.k ** 2), self.d)
    self.conv = sepconv2d(self.d, self.d, kernel_size=2 * self.s + 1, stride=1, padding="same", relu=False)

  def forward(self, x):
    batchsize = x.shape[0]
    c = x.shape[1]
    h = x.shape[2] // self.k
    w = x.shape[3] // self.k
    patches = split(x, self.k)
    patches = patches.view(-1, self.c * (self.k ** 2))
    fft = torch.fft.fft(patches, dim=-1)
    fc = self.fc(fft)
    ifft = torch.fft.ifft(fc).real
    ifft = rearrange(ifft, '(b h w) d -> b d h w', h=h, w=w)
    conved = self.conv(ifft)
    return F.interpolate(conved, size=self.size)

class CFNONet(nn.Module):
  def __init__(self):
    super().__init__()

    self.cfno0 = CFNO(c=1, d=16, k=16, s=1)
    self.cfno1 = CFNO(c=1, d=32, k=32, s=1)
    self.cfno2 = CFNO(c=1, d=64, k=64, s=1)

    self.conv0a = conv2d(1, 32, kernel_size=3, stride=2, padding=1, relu=True)
    self.conv0b = repeat2d(2, 32, 32, kernel_size=3, stride=1, padding=1, relu=True)
    self.conv1a = conv2d(32, 64, kernel_size=3, stride=2, padding=1, relu=True)
    self.conv1b = repeat2d(2, 64, 64, kernel_size=3, stride=1, padding=1, relu=True)
    self.conv2a = conv2d(64, 128, kernel_size=3, stride=2, padding=1, relu=True)
    self.conv2b = repeat2d(2, 128, 128, kernel_size=3, stride=1, padding=1, relu=True)
    self.branch = nn.Sequential(self.conv0a, self.conv0b, self.conv1a, self.conv1b, self.conv2a, self.conv2b)

    self.deconv0a = deconv2d(16 + 32 + 64 + 128, 128, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
    self.deconv0b = repeat2d(2, 128, 128, kernel_size=3, stride=1, padding=1, relu=True)
    self.deconv1a = deconv2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
    self.deconv1b = repeat2d(2, 64, 64, kernel_size=3, stride=1, padding=1, relu=True)
    self.deconv2a = deconv2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
    self.deconv2b = repeat2d(2, 32, 32, kernel_size=3, stride=1, padding=1, relu=True)

    self.conv3 = conv2d(32, 32, kernel_size=3, stride=1, padding=1, relu=True)
    self.conv4 = conv2d(32, 32, kernel_size=3, stride=1, padding=1, relu=True)
    self.conv5 = conv2d(32, 32, kernel_size=3, stride=1, padding=1, relu=True)
    self.conv6 = conv2d(32, 1, kernel_size=3, stride=1, padding=1, norm=False, relu=False)

    self.tail = nn.Sequential(self.deconv0a, self.deconv0b, self.deconv1a, self.deconv1b, self.deconv2a, self.deconv2b,
                              self.conv3, self.conv4, self.conv5, self.conv6)
  def forward(self, x):
    br0 = self.cfno0(x)
    br1 = self.cfno1(x)
    br2 = self.cfno2(x)
    br3 = self.branch(x)

    feat = torch.cat([br0, br1, br2, br3], dim=1)
    result = self.tail(feat)

    return result

# class CFNOILT(ModelILT):
#   def __init__(self, size=(1024, 1024)):
#     super().__init__(size=size, name="CFNOILT")
#     self.simLitho = litho.LithoSim("./config/lithosimple.txt")
#     self.net = CFNONet()
#     if torch.cuda.is_available():
#       self.net = self.net.cuda()
#
#   @property
#   def size(self):
#     return self._size
#
#   @property
#   def name(self):
#     return self._name
#
#   def pretrain(self, train_loader, val_loader, epochs=1):
#     opt = optim.Adam(self.net.parameters(), lr=1e-3)
#     sched = lr_sched.StepLR(opt, 1, gamma=0.1)
#     for epoch in range(epochs):
#       print(f"[Pre-Epoch {epoch}] Training")
#       self.net.train()
#       progress = tqdm(train_loader)
#       for target, label in progress:
#         if torch.cuda.is_available():
#           target = target.cuda()
#           label = label.cuda()
#
#         mask = self.net(target)
#         loss = F.mse_loss(mask, label)
#
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#
#         progress.set_postfix(loss=loss.item())
#
#       print(f"[Pre-Epoch {epoch}] Testing")
#       self.net.eval()
#       losses = []
#       progress = tqdm(val_loader)
#       for target, label in progress:
#         with torch.no_grad():
#           if torch.cuda.is_available():
#             target = target.cuda()
#             label = label.cuda()
#
#           mask = self.net(target)
#           loss = F.mse_loss(mask, label)
#           losses.append(loss.item())
#
#           progress.set_postfix(loss=loss.item())
#
#       print(f"[Pre-Epoch {epoch}] loss = {np.mean(losses)}")
#
#       if epoch == epochs // 2:
#         sched.step()
#
#   def train(self, train_loader, val_loader, epochs=1):
#     opt = optim.Adam(self.net.parameters(), lr=1e-3)
#     sched = lr_sched.StepLR(opt, 1, gamma=0.1)
#     for epoch in range(epochs):
#       print(f"[Epoch {epoch}] Training")
#       self.net.train()
#       progress = tqdm(train_loader)
#       for target, label in progress:
#         if torch.cuda.is_available():
#           target = target.cuda()
#           label = label.cuda()
#
#         mask = self.net(target)
#         printedNom, printedMax, printedMin = self.simLitho(mask.squeeze(1))
#         l2loss = F.mse_loss(printedNom.unsqueeze(1), target)
#         printedNom, printedMax, printedMin = self.simLitho(label.squeeze(1))
#         l2lossRef = F.mse_loss(printedNom.unsqueeze(1), target)
#         loss = F.mse_loss(mask, mask) if l2loss.item() < l2lossRef.item() else F.mse_loss(mask, label)
#
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#
#         progress.set_postfix(loss=loss.item())
#
#       print(f"[Epoch {epoch}] Testing")
#       self.net.eval()
#       l2losses = []
#       progress = tqdm(val_loader)
#       for target, label in progress:
#         with torch.no_grad():
#           if torch.cuda.is_available():
#             target = target.cuda()
#             label = label.cuda()
#
#           mask = self.net(target)
#           mask = mask.squeeze(1)
#           printedNom, printedMax, printedMin = self.simLitho(mask)
#           l2loss = F.mse_loss(printedNom.unsqueeze(1), target)
#           l2losses.append(l2loss.item())
#
#           progress.set_postfix(l2loss=l2loss.item())
#
#       print(f"[Epoch {epoch}] L2 loss = {np.mean(l2losses)}")
#
#       if epoch == epochs // 2:
#         sched.step()
#
#   def save(self, filenames):
#     filename = filenames[0] if isinstance(filenames, list) else filenames
#     torch.save(self.net.state_dict(), filename)
#
#   def load(self, filenames):
#     filename = filenames[0] if isinstance(filenames, list) else filenames
#     self.net.load_state_dict(torch.load(filename))
#
#   def run(self, target):
#     self.net.eval()
#     return self.net(target)[0, 0].detach()

if __name__ == '__main__':
  # Define input tensor
  input_tensor = torch.randn(1, 1, 1024, 1024)  # Batch size = 1, single channel
  # Initialize the model
  model = CFNONet()
  # Forward pass
  output = model(input_tensor)
  # Check output shape
  print(f"Output shape: {output.shape}")  # Should be torch.Size([1, 1, 1024, 1024])
  print(f'Model: {model}')
  print(summary(model, (1,1,1024,1024)))


