import cv2
import torch
import random
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
import logging
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.attention_unet import AttentionUNetGenerator
from src.utils import ContourLoss, IouLoss, next_exp_folder, IoU, PixelAccuracy, draw_plot
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

def save_generated_image(output, epoch, step, checkpoint_dir="checkpoints", image_type='true_correction'):
  single_image = output[0].squeeze(dim = 0)
  single_image[single_image > 0.5] = 1.0
  single_image[single_image <= 0.5] = 0.0

  img_save_path = os.path.join(checkpoint_dir, f"{image_type}_epoch{epoch}_step{step}.png")
  cv2.imwrite(f"{img_save_path}", (single_image * 255).detach().cpu().numpy())
  print(f"Saved generated image at {img_save_path}")
  logging.info(f"Saved generated image at {img_save_path}")

def setup_logging(exp_folder):
  # Set up logging configuration
  log_file_path = os.path.join(exp_folder, 'training_log.txt')
  logging.basicConfig(filename = log_file_path,
                      level = logging.INFO,
                      datefmt = '%d/%m/%Y %H:%M')


def validate_model(model, val_loader, current_epoch, num_epochs ,checkpoint_dir,device='cuda'):
  model.eval()
  l1_loss_epoch = 0
  iou_loss_epoch = 0
  total_loss_epoch = 0
  pixel_acc_epoch = 0
  iou_epoch = 0

  print(f"[Epoch {current_epoch}] Validating")
  logging.info(f"[Epoch {current_epoch}] Validating")

  with torch.no_grad():
    for idx, (image, target) in enumerate(val_loader):
      image, target = image.to(device), target.to(device)

      params = model(image)
      mask = torch.sigmoid(params)
      # calculating loss during validation phase
      l1_loss_iter = l1_loss(mask, target)
      # l2_loss_iter = criterion_mse(mask, target) # nn.MSELoss(mask, target)
      iou_loss_iter = iou_loss(mask, target)
      total_loss_iter = l1_loss_iter + iou_loss_iter

      # calculating metrics during validation phase
      pixel_acc_iter = pixel_accuracy(mask, target)
      iou_iter = iou(mask, target)

      if idx % 10 == 0:
        log_info_iter = {
          'epoch': current_epoch,
          'num_epochs': num_epochs,
          'step': idx,
          'len_valid_loader': len(val_loader),
          'l1_loss': l1_loss_iter.item(),
          'iou_loss': iou_loss_iter.item(),
          'total_loss': total_loss_iter.item(),
          'pixel_acc': pixel_acc_iter.item(),
          'iou': iou_iter.item()
        }

        log_message_iter = (f"Epoch [{log_info_iter['epoch']}/{log_info_iter['num_epochs']}], "
                       f"Step [{log_info_iter['step']}/{log_info_iter['len_valid_loader']}], "
                       f"L1 Loss: {log_info_iter['l1_loss']:.4f}, "
                       f"IoU Loss: {log_info_iter['iou_loss']:.4f}, "
                       f"Total Loss: {log_info_iter['total_loss']:.4f}, "
                       f"Pixel Accuracy: {log_info_iter['pixel_acc']:.4f}, "
                       f"IoU: {log_info_iter['iou']:.4f}")

        # Print and log the message
        print(log_message_iter)
        logging.info(log_message_iter)

      if idx % 200 == 0:
        # save_generated_image(target, epoch, idx, checkpoint_dir=checkpoint_dir, image_type='true_correction')
        save_generated_image(mask, current_epoch, idx, checkpoint_dir=checkpoint_dir, image_type='generated_correction_val')

      l1_loss_epoch += l1_loss_iter.item()
      iou_loss_epoch += iou_loss_iter.item()
      total_loss_epoch += total_loss_iter.item()
      pixel_acc_epoch += pixel_acc_iter.item()
      iou_epoch += iou_iter.item()

    log_info_epoch = {
      'epoch': current_epoch,
      'num_epochs': num_epochs,
      'len_valid_loader': len(val_loader),
      'l1_loss': l1_loss_epoch,
      'iou_loss': iou_loss_epoch,
      'total_loss': total_loss_epoch,
      'pixel_acc': pixel_acc_epoch,
      'iou': iou_epoch
    }

    log_message_epoch = (f"Losses per epoch [{log_info_epoch['epoch']}/{log_info_epoch['num_epochs']}], "
                        f"L1 Loss: {log_info_epoch['l1_loss'] / log_info_epoch['len_valid_loader'] :.4f}, "
                        f"IoU Loss: {log_info_epoch['iou_loss'] / log_info_epoch['len_valid_loader'] :.4f}, "
                        f"Total Loss: {log_info_epoch['total_loss'] / log_info_epoch['len_valid_loader'] :.4f}, "
                        f"Pixel Accuracy: {log_info_epoch['pixel_acc'] / log_info_epoch['len_valid_loader'] :.4f}, "
                        f"IoU: {log_info_epoch['iou'] / log_info_epoch['len_valid_loader'] :.4f}")

    # Print and log the message
    print(log_message_epoch)
    logging.info(log_message_epoch)

  # return average total loss, average pixel accuracy and average iou per epoch
  return log_info_epoch['total_loss'] / log_info_epoch['len_valid_loader'], log_info_epoch['pixel_acc'] / log_info_epoch['len_valid_loader'], log_info_epoch['iou'] / log_info_epoch['len_valid_loader']

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

def train_model(model, train_loader,
                   val_loader, num_epochs,
                   lr = 2e-4, device = "cuda",
                   start_epoch = 0,
                   checkpoint_dir = "checkpoints",
                   resume = False):
  # Load checkpoint if resuming
  if resume:
    checkpoint = torch.load('/mnt/data/amoskovtsev/mb_opc/checkpoints/exp_3/last_checkpoint.pth') # torch.load(os.path.join(checkpoint_dir, "checkpoint_14.10.24.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = lr_sched.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")
    logging.info(f"Resuming training from epoch {start_epoch}")
  else:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-5)
    # scheduler = lr_sched.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)
    scheduler = lr_sched.CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=1e-7)
    print('Starting experiment...')
    logging.info('Starting experiment...')
  total_loss_epoch_list_train = []
  total_loss_epoch_list_val = []
  total_loss_iter_list = []

  pixel_acc_epoch_list_train = []
  pixel_acc_epoch_list_val = []

  iou_epoch_list_train = []
  iou_epoch_list_val = []

  torch.autograd.set_detect_anomaly(True)
  for epoch in range(start_epoch, num_epochs):
    # l2_loss_epoch = 0
    l1_loss_epoch = 0
    iou_loss_epoch = 0
    total_loss_epoch = 0
    pixel_acc_epoch = 0
    iou_epoch = 0

    print(f"[Epoch {epoch}] Training")
    logging.info(f"[Epoch {epoch}] Training")
    model.train()

    progress = tqdm(train_loader)

    for idx, (image, target) in enumerate(progress):
      image, target = image.to(device), target.to(device)
      params = model(image)
      mask = torch.sigmoid(params)

      # calculate losses during train phase
      l1_loss_iter = l1_loss(mask, target)
      iou_loss_iter = iou_loss(mask, target)
      total_loss_iter = l1_loss_iter + iou_loss_iter
      total_loss_iter_list.append(total_loss_iter.item())

      # calculate metrics during train phase
      pixel_acc_iter = pixel_accuracy(mask, target)
      iou_iter = iou(mask, target)

      optimizer.zero_grad()
      total_loss_iter.backward()
      optimizer.step()

      if idx % 50 == 0:
        # Create a dictionary to hold your log information
        log_info_iter = {
          'epoch': epoch,
          'num_epochs': num_epochs,
          'step': idx,
          'len_train_loader': len(train_loader),
          'l1_loss': l1_loss_iter.item(),
          'iou_loss': iou_loss_iter.item(),
          'total_loss': total_loss_iter.item(),
          'pixel_acc': pixel_acc_iter.item(),
          'iou': iou_iter.item()
        }

        log_message_iter = (f"Epoch [{log_info_iter['epoch']}/{log_info_iter['num_epochs']}], "
                       f"Step [{log_info_iter['step']}/{log_info_iter['len_train_loader']}], "
                       f"L1 Loss: {log_info_iter['l1_loss']:.4f}, "
                       f"IoU Loss: {log_info_iter['iou_loss']:.4f}, "
                       f"Total Loss: {log_info_iter['total_loss']:.4f}, "
                       f"Pixel Accuracy: {log_info_iter['pixel_acc']:.4f}, "
                       f"IoU: {log_info_iter['iou']:.4f} ")

        print(log_message_iter)
        logging.info(log_message_iter)
        # draw loss plot per iteration
        draw_plot(first_variable = total_loss_iter_list, label = 'train_iter_loss',
                  title = 'Iteration loss plot', xlabel = 'Iteration',
                  ylabel = 'Loss', save_name = 'train_iter_loss.jpg',
                  checkpoint_dir = checkpoint_dir)

      if idx % 200 == 0:
        # save generated mask
        save_generated_image(mask, epoch, idx, checkpoint_dir = checkpoint_dir, image_type = 'generated_correction')

      l1_loss_epoch += l1_loss_iter.item()
      iou_loss_epoch += iou_loss_iter.item()
      total_loss_epoch += total_loss_iter.item()

      pixel_acc_epoch += pixel_acc_iter.item()
      iou_epoch += iou_iter.item()

      if idx % 500 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint at {checkpoint_path}")
        logging.info(f"Saved checkpoint at {checkpoint_path}")


    log_info_epoch = {
      'epoch': epoch,
      'num_epochs': num_epochs,
      'len_train_loader': len(train_loader),
      'l1_loss': l1_loss_epoch,
      'iou_loss': iou_loss_epoch,
      'total_loss': total_loss_epoch,
      'pixel_acc': pixel_acc_epoch,
      'iou': iou_epoch
    }

    log_message_epoch = (f"Losses per epoch [{log_info_epoch['epoch']}/{log_info_epoch['num_epochs']}], "
                        f"L1 Loss: {log_info_epoch['l1_loss'] / log_info_epoch['len_train_loader'] :.4f}, "
                        f"IoU Loss: {log_info_epoch['iou_loss'] / log_info_epoch['len_train_loader'] :.4f}, "
                        f"Total Loss: {log_info_epoch['total_loss'] / log_info_epoch['len_train_loader'] :.4f}"
                        f"Pixel Accuracy: {log_info_epoch['pixel_acc'] / log_info_epoch['len_train_loader'] :.4f}"
                        f"IoU: {log_info_epoch['iou'] / log_info_epoch['len_train_loader'] :.4f}")

    print(log_message_epoch)
    logging.info(log_message_epoch)

    # get average loss, pixel accuracy and iou per epoch during training phase
    total_loss_epoch_list_train.append(log_info_epoch['total_loss'] / log_info_epoch['len_train_loader'])
    pixel_acc_epoch_list_train.append(log_info_epoch['pixel_acc'] / log_info_epoch['len_train_loader'])
    iou_epoch_list_train.append(log_info_epoch['iou'] / log_info_epoch['len_train_loader'])

    # get average loss, pixel accuracy and iou per epoch during validation phase
    total_loss_epoch_val, pixel_acc_epoch_val, iou_epoch_val = validate_model(model, val_loader, current_epoch = epoch,
                                                                              num_epochs = num_epochs, checkpoint_dir = checkpoint_dir,
                                                                              device = device)
    total_loss_epoch_list_val.append(total_loss_epoch_val)
    iou_epoch_list_val.append(iou_epoch_val)
    pixel_acc_epoch_list_val.append(pixel_acc_epoch_val)

    # draw epoch losses for training and validation phases
    draw_plot(first_variable = total_loss_epoch_list_train, second_variable = total_loss_epoch_list_val,
              title='Loss plot', xlabel='epoch',
              ylabel='loss', first_label='train_loss', second_label='valid_loss',
              save_name='epoch_loss.jpg', checkpoint_dir = checkpoint_dir)

    # draw epoch IoU for training and validation phases
    draw_plot(first_variable = iou_epoch_list_train, second_variable = iou_epoch_list_val,
              title = 'IoU plot', xlabel = 'epoch',
              ylabel = 'iou', first_label = 'train_iou', second_label = 'valid_iou',
              save_name = 'epoch_iou.jpg', checkpoint_dir = checkpoint_dir)

    # draw epoch Pixel Accuracy for training and validation phases
    draw_plot(first_variable = pixel_acc_epoch_list_train, second_variable = pixel_acc_epoch_list_val,
              title = 'Pixel Accuracy plot', xlabel = 'epoch',
              ylabel = 'pixel accuracy', first_label = 'train_pixel_acc', second_label = 'valid_pixel_acc',
              save_name = 'epoch_pixel_acc.jpg', checkpoint_dir = checkpoint_dir)

    scheduler.step()

  print('Traning complete')
  logging.info('Traning complete')

CHECKPOINT_DIR = next_exp_folder(CHECKPOINT_PATH)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Experiment logs will be saved in: {CHECKPOINT_DIR}')
setup_logging(CHECKPOINT_DIR)
logging.info(f'Experiment logs will be saved in: {CHECKPOINT_DIR}')
logging.info(f'Training device:{DEVICE}')

generator_model = AttentionUNetGenerator(in_channels = 1, out_channels = 1)
print('Model initialized:', generator_model)
logging.info('Model initialized')

test_image = torch.randn((1,1,1024,1024))
output = generator_model.forward(test_image)
print(f'Output shape: {output.shape}')
print(f'Output shape: {torch.sigmoid(output).shape}')

logging.info(f'Batch size:{BATCH_SIZE}')
# Define dataset
TRAIN_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/train_origin'), os.path.join(DATASET_PATH,'correction/train_correction'), transform = apply_transform(binarize_flag = True))
VALID_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/valid_origin'), os.path.join(DATASET_PATH, 'correction/valid_correction'), transform = apply_transform(binarize_flag = True))
TEST_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/test_origin'), os.path.join(DATASET_PATH, 'correction/test_correction'), transform = apply_transform(binarize_flag = True))

# Define dataloader
TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
VALID_LOADER = DataLoader(VALID_DATASET, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
TEST_LOADER = DataLoader(TEST_DATASET, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)

print(f'Number of images in train subset:{len(TRAIN_DATASET)}')
print(f'Number of images in valid subset:{len(VALID_DATASET)}')
print(f'Number of images in test subset:{len(TEST_DATASET)}\n')

logging.info(f'Number of images in train subset:{len(TRAIN_DATASET)}')
logging.info(f'Number of images in valid subset:{len(VALID_DATASET)}')
logging.info(f'Number of images in test subset:{len(TEST_DATASET)}\n')

image, target = next(iter(TRAIN_LOADER))
image, target = image.to(DEVICE), target.to(DEVICE)
print(f'Image shape: {image.shape}')
print(f'Target shape: {target.shape}')
print(f'Image shape after removing batch dimension: {image[0,0].shape}')

# Выполним проверку подсчёта функций потерь
generator_model = generator_model.to(DEVICE)
output = generator_model(image)
print(f'Output shape: {output.shape}')

iou_loss = IouLoss(weight=1.0)
# criterion_mse = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()
pixel_accuracy = PixelAccuracy()
iou = IoU()

iou_loss_value = iou_loss(target, output.sigmoid())
l1_loss_value = l1_loss(target, output.sigmoid())

print(f'IoU loss:{iou_loss_value}')
print(f'L1-loss:{l1_loss_value}')
print(f'IoU loss shape:{iou_loss_value.shape}')
print(f'L1-loss shape:{l1_loss_value.shape}')

# plt.imshow(target[0,0], cmap='gray')
# plt.show()

start_train = time.time()
#Train the model
train_model(model = generator_model,
               train_loader = TRAIN_LOADER,
               val_loader = VALID_LOADER,
               num_epochs = EPOCHS,
               lr = LEARNING_RATE,
               device = DEVICE,
               start_epoch = 0,
               checkpoint_dir = CHECKPOINT_DIR,
               resume=False)
end_train = time.time()
total_time = end_train - start_train
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f'Training took: {int(hours):02}:{int(minutes):02}:{int(seconds):02}')
logging.info(f'Training took: {int(hours):02}:{int(minutes):02}:{int(seconds):02}')

print(f'Evaluating model on validation set..:')
logging.info(f'Evaluating model on validation set..:')
evaluate_model(model = generator_model,
               loader = VALID_LOADER,
               device = DEVICE,
               log = True)
print(f'Evaluating model on test set..:')
logging.info(f'Evaluating model on test set..:')
evaluate_model(model = generator_model,
               loader = TEST_LOADER,
               device = DEVICE,
               log = True)