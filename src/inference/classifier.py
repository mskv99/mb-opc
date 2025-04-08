import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_sched
from torchvision import transforms, datasets
from torchinfo import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging
import time
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import next_exp_folder, draw_plot
from src.dataset import BinarizeTransform
from src.config import CLASSIFACTION_DATA_PATH, CHECKPOINT_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE


def set_random_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)


# set_random_seed(42)

data_transforms = {
  'train': transforms.Compose([
    transforms.Resize(1024),
    transforms.ToTensor(),
    transforms.Grayscale(),
    BinarizeTransform(threshold=0.5)
  ]),
  'val': transforms.Compose([
    transforms.Resize(1024),
    transforms.ToTensor(),
    transforms.Grayscale(),
    BinarizeTransform(threshold=0.5)
  ])
}


class BinaryClassificationCNN(nn.Module):
  def __init__(self):
    super(BinaryClassificationCNN, self).__init__()

    # Convolutional layers
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Output: 16 x 512 x 512
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Output: 32 x 256 x 256
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: 64 x 128 x 128
    self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output: 128 x 64 x 64
    self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Output: 256 x 32 x 32

    # Fully connected layers
    self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Flattened input to 512 nodes
    self.fc2 = nn.Linear(512, 1)  # Output layer for binary classification

    # Normalization layers
    self.bn1 = nn.BatchNorm2d(16)
    self.bn2 = nn.BatchNorm2d(32)
    self.bn3 = nn.BatchNorm2d(64)
    self.bn4 = nn.BatchNorm2d(128)
    # Activation and pooling
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.global_avgpool = nn.AdaptiveAvgPool2d((4, 4))

  def forward(self, x):
    x = self.avgpool(x)
    x = self.bn1(self.relu(self.conv1(x)))
    x = self.bn2(self.relu(self.conv2(x)))
    x = self.bn3(self.relu(self.conv3(x)))
    x = self.bn4(self.relu(self.conv4(x)))
    x = self.relu(self.conv5(x))

    # Global average pooling
    x = self.global_avgpool(x)

    # Flatten the tensor
    x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers

    # Fully connected layers
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    # Apply sigmoid for binary classification
    # x = self.sigmoid(x)
    return x


class SaveActivations:
  def __init__(self, module):
    self.hook = module.register_forward_hook(self.hook_fn)
    self.features = None

  def hook_fn(self, module, input, output):
    self.features = output

  def remove(self):
    self.hook.remove()


def calculate_accuracy(outputs, labels):
  predictions = (outputs > 0.5).float()  # Convert probabilities to binary predictions (0 or 1)
  correct = (predictions == labels).sum().item()
  accuracy = correct / labels.size(0)
  return accuracy


def evaluate(model, loader, loader_type='test'):
  accuracy = 0
  model.eval()
  with torch.no_grad():
    for images, labels in loader:
      images, labels = images.to(device), labels.to(device)
      labels = labels.view(-1, 1).float()

      outputs = model(images).sigmoid()
      accuracy += calculate_accuracy(outputs, labels)

  # Average loss and accuracy for the validation set
  accuracy /= len(loader)

  print(f'Evaluation on {loader_type} loader:')
  print(f'Accuracy: {accuracy}')


def predict(model, loader, loader_type='test', num_samples=10):
  model.eval()
  probs = []
  true_labels = []
  pred_labels = []
  images_list = []

  running_correct = 0
  for images, labels in loader:
    images, labels = images.to(device), labels.to(device)

    outputs = model(images).sigmoid()
    preds = (outputs > 0.5).sum()

    images_list.append(np.squeeze(images.detach().cpu().numpy(), axis=(0)))
    true_labels.append(labels.detach().cpu().data.numpy())
    probs.append(outputs.detach().cpu().data.numpy())
    pred_labels.append(preds.detach().cpu().data.numpy())

  rows = 2
  cols = 5
  fig = plt.figure(figsize=(4 * cols - 1, 4 * rows - 1))
  for i in range(cols):
    for j in range(rows):
      random_index = np.random.randint(0, len(true_labels))
      ax = fig.add_subplot(rows, cols, i * rows + j + 1)
      ax.grid('off')
      ax.axis('off')
      ax.imshow(np.transpose(images_list[random_index], (1, 2, 0)), cmap='gray')
      ax.set_title(f'real_class: {class_names[int(true_labels[random_index])]} \n'
                   f'predicted class: {class_names[int(pred_labels[random_index])]} \n'
                   f'probability: {probs[random_index]}')
  plt.show()

  # print(f'True labels: {true_labels}')
  # print(f'Probabilities: {probs}')
  # print(f'Predicted labels: {pred_labels}')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'checkpoints/exp_22/last_checkpoint.pth'
VAL_DATASET = datasets.ImageFolder(os.path.join(CLASSIFACTION_DATA_PATH, 'val'), data_transforms['val'])
TEST_DATASET = datasets.ImageFolder(os.path.join(CLASSIFACTION_DATA_PATH, 'test'), data_transforms['val'])

VAL_LOADER = torch.utils.data.DataLoader(VAL_DATASET, batch_size=1, shuffle=False)
TEST_LOADER = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=True)

class_names = VAL_DATASET.classes
print(f'Dataset classes:{class_names}')

classifier_model = BinaryClassificationCNN()
classifier_model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
classifier_model = classifier_model.to(device)
classifier_model.eval()
print(classifier_model)
print(summary(classifier_model, (1, 1, 1024, 1024)))

# conv2_hook = SaveActivations(classifier_model.conv2)
# conv3_hook = SaveActivations(classifier_model.conv3)
# conv4_hook = SaveActivations(classifier_model.conv4)
# conv5_hook = SaveActivations(classifier_model.conv5)
#
# image, label = next(iter(TEST_LOADER))
# image, label = image.to(device), label.to(device)
#
# with torch.no_grad():
#   prediction = classifier_model(image)
# conv2_features = conv2_hook.features
# conv3_features = conv3_hook.features
# conv4_features = conv4_hook.features
# conv5_features = conv5_hook.features
#
# conv2_hook.remove()
# conv3_hook.remove()
# conv4_hook.remove()
# conv5_hook.remove()
#
# print(conv2_features.shape)  # Should be [batch_size, 32, 256, 256]
# print(conv3_features.shape)
# print(conv4_features.shape)
# print(conv5_features.shape)  # Should be [batch_size, 256, 32, 32]
#
# # feature_map = conv5_features[0, 0].cpu().detach().numpy()  # Select the first feature map of the first image in the batch
# #
# # num_channels = conv4_features.shape[1]
# # for i in range(min(num_channels, 8)):  # Visualize 8 channels only for simplicity
# #   plt.imshow(conv4_features[0, i].cpu().detach().numpy(), cmap='gray')
# #   plt.title(f'Feature Map from conv5 - Channel {i}')
# #   plt.colorbar()
# #   plt.show()
#
#
#
# # Number of channels in the feature map
# num_channels = conv5_features.shape[1]
#
# # Choose how many channels you want to display (or set to num_channels to show all)
# # For large model, it might be impractical to show all feature maps at once.
# num_to_display = min(num_channels, 16 + 1)  # For example, display the first 16 channels
#
# # Calculate the grid size (e.g., 4x4 for 16 channels)
# grid_size = math.ceil(math.sqrt(num_to_display))
#
# # Create subplots
# fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
#
# # Flatten the axes array for easy iteration
# axes = axes.flatten()
#
# # Loop over the selected channels and display them
# for i in range(num_to_display + 1):
#     if i == num_to_display:
#       axes[i].imshow(image[0,0].cpu().detach().numpy(), cmap='gray')
#       axes[i].set_title(f'Raw image')
#       axes[i].axis('off')
#       continue
#
#     feature_map = conv5_features[0, i].cpu().detach().numpy()  # Get the i-th feature map
#     axes[i].imshow(feature_map, cmap='gray')  # Show the feature map in grayscale
#     axes[i].set_title(f'Channel {i}')
#     axes[i].axis('off')  # Hide axis for better visualization
#
# # Turn off any remaining empty subplots
# for i in range(num_to_display, len(axes)):
#     axes[i].axis('off')
#
# # Adjust the layout
# plt.tight_layout()
# plt.show()

# evaluate(model = classifier_model, loader = VAL_LOADER, loader_type = 'val')
# evaluate(model = classifier_model, loader = TEST_LOADER, loader_type = 'test')
predict(model=classifier_model, loader=TEST_LOADER, loader_type='test')
