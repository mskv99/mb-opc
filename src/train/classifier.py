import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_sched
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import next_exp_folder, draw_plot
from src.dataset import OPCDataset, BinarizeTransform, calculate_mean_std, apply_transform
from src.config import DATASET_PATH, CHECKPOINT_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE

def set_random_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

set_random_seed(42)

def setup_logging(exp_folder):
  # Set up logging configuration
  log_file_path = os.path.join(exp_folder, 'training_log.txt')
  logging.basicConfig(filename = log_file_path,
                      level = logging.INFO,
                      datefmt = '%d/%m/%Y %H:%M')


data_transforms = {
  'train': transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Grayscale(),
    BinarizeTransform(threshold=0.5)
  ]),
  'val': transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Grayscale(),
    BinarizeTransform(threshold=0.5)
  ])
}

DATA_DIR = 'data/processed/classifier/'

TRAIN_DATASET = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train'])
VAL_DATASET = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val'])
TRAIN_LOADER = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size = 3, shuffle = True, num_workers = 2)
VAL_LOADER = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size = 3, shuffle = False, num_workers = 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = TRAIN_DATASET.classes
dataset_sizes = {'train': len(TRAIN_DATASET), 'val': len(VAL_DATASET)}
print(f'Dataset size:{dataset_sizes}')
print(f'Class names:{class_names}')


class BinaryClassificationCNN(nn.Module):
  def __init__(self):
    super(BinaryClassificationCNN, self).__init__()

    # Convolutional layers
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Output: 16 x 512 x 512
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Output: 32 x 256 x 256
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: 64 x 128 x 128
    self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output: 128 x 64 x 64

    # Fully connected layers
    self.fc1 = nn.Linear(128 * 64 * 64, 512)  # Flattened input to 512 nodes
    self.fc2 = nn.Linear(512, 1)  # Output layer for binary classification

    # Activation and pooling
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))
    x = self.relu(self.conv4(x))

    # Flatten the tensor
    x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers

    # Fully connected layers
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    # Apply sigmoid for binary classification
    x = self.sigmoid(x)
    return x

def calculate_accuracy(outputs, labels):
  predictions = (outputs > 0.5).float()  # Convert probabilities to binary predictions (0 or 1)
  correct = (predictions == labels).sum().item()
  accuracy = correct / labels.size(0)
  return accuracy

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, checkpoint_dir):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')

  # Track loss and accuracy
  train_losses, val_losses = [], []
  train_accuracies, val_accuracies = [], []

  for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      labels = labels.view(-1, 1).float()  # Reshape labels for BCELoss

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)

      # Backward pass and optimization
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      train_accuracy += calculate_accuracy(outputs, labels)

    # Average loss and accuracy for the training set
    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
      for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.view(-1, 1).float()

        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_accuracy += calculate_accuracy(outputs, labels)

    # Average loss and accuracy for the validation set
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)

    # Log results to console and file
    log_line = (f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    print(log_line)
    logging.info(log_line)

    # Append losses and accuracies for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Saved checkpoint at {checkpoint_path}")
    logging.info(f"Saved checkpoint at {checkpoint_path}")

  # draw epoch losses for training and validation phases
  draw_plot(first_variable=train_losses, second_variable=val_losses,
            title='Loss plot', xlabel='epoch',
            ylabel='loss', first_label='train_loss', second_label='valid_loss',
            save_name='epoch_loss.jpg', checkpoint_dir=checkpoint_dir)

  # draw epoch Pixel Accuracy for training and validation phases
  draw_plot(first_variable=train_accuracies, second_variable=val_accuracies,
            title='Pixel Accuracy plot', xlabel='epoch',
            ylabel='accuracy', first_label='train_acc', second_label='valid_acc',
            save_name='epoch_acc.jpg', checkpoint_dir=checkpoint_dir)

if __name__ == '__main__':

  DATA_DIR = 'data/processed/classifier/'
  CHECKPOINT_DIR = next_exp_folder(CHECKPOINT_PATH)
  print(f'Experiment logs will be saved in: {CHECKPOINT_DIR}')
  setup_logging(CHECKPOINT_DIR)
  logging.info(f'Experiment logs will be saved in: {CHECKPOINT_DIR}')

  TRAIN_DATASET = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train'])
  VAL_DATASET = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val'])
  TRAIN_LOADER = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=5, shuffle=True, num_workers=2)
  VAL_LOADER = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=5, shuffle=False, num_workers=2)

  class_names = TRAIN_DATASET.classes
  dataset_sizes = {'train': len(TRAIN_DATASET), 'val': len(VAL_DATASET)}
  print(f'Dataset size:{dataset_sizes}')
  print(f'Class names:{class_names}')
  logging.info(f'Dataset size:{dataset_sizes}')
  logging.info(f'Class names:{class_names}')

  model = BinaryClassificationCNN()

  start_train = time.time()
  train_model(model = model,
              train_loader = TRAIN_LOADER,
              val_loader = VAL_LOADER,
              num_epochs = 10,
              learning_rate = 0.001,
              checkpoint_dir = CHECKPOINT_DIR)
  end_train = time.time()
  total_time = end_train - start_train
  hours, rem = divmod(total_time, 3600)
  minutes, seconds = divmod(rem, 60)
  print(f'Training took: {int(hours):02}:{int(minutes):02}:{int(seconds):02}')
  logging.info(f'Training took: {int(hours):02}:{int(minutes):02}:{int(seconds):02}')
