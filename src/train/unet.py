import cv2
import wandb
import torch
import random
import numpy as np
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
import logging
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.unet import Generator
from src.utils import ContourLoss, IouLoss, next_exp_folder, IoU, PixelAccuracy, draw_plot, MobileNetPerceptualLoss
from src.dataset import OPCDataset, BinarizeTransform, calculate_mean_std, apply_transform
from src.config import DATASET_PATH, CHECKPOINT_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, LOG_WANDB, RESUME, DEBUG_LOSS

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

def validate_model(model,
                   val_loader,
                   current_epoch,
                   num_epochs,
                   checkpoint_dir,
                   device = 'cuda'):
  model.eval()
  lossG_l1_epoch = 0
  lossG_iou_epoch = 0
  lossG_epoch = 0
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
      lossG_l1_iter = l1_loss(mask, target)
      lossG_iou_iter = iou_loss(mask, target)
      lossG_iter = lossG_l1_iter + lossG_iou_iter

      # calculating metrics during validation phase
      pixel_acc_iter = pixel_accuracy(mask, target)
      iou_iter = iou(mask, target)

      if idx % 10 == 0:
        log_info_iter = {
          'val/lossG_l1/iter': lossG_l1_iter.item(),
          'val/lossG_iou/iter': lossG_iou_iter.item(),
          'val/lossG/iter': lossG_iter.item(),
          'val/pixel_acc/iter': pixel_acc_iter.item(),
          'val/iou/iter': iou_iter.item()
        }

        log_message_iter = (f"Epoch [{current_epoch}/{num_epochs}], "
                       f"Step [{idx}/{len(val_loader)}], "
                       f"L1 Loss: {log_info_iter['val/lossG_l1/iter']:.4f}, "
                       f"IoU Loss: {log_info_iter['val/lossG_iou/iter']:.4f}, "
                       f"Total Loss: {log_info_iter['val/lossG/iter']:.4f}, "
                       f"Pixel Accuracy: {log_info_iter['val/pixel_acc/iter']:.4f}, "
                       f"IoU: {log_info_iter['val/iou/iter']:.4f}")

        # Print and log the message
        print(log_message_iter)
        logging.info(log_message_iter)

        if LOG_WANDB:
          wandb.log(log_info_iter)

      if idx % 200 == 0:
        # save_generated_image(target, epoch, idx, checkpoint_dir=checkpoint_dir, image_type='true_correction')
        save_generated_image(mask, current_epoch, idx, checkpoint_dir = checkpoint_dir, image_type = 'generated_correction_val')

      lossG_l1_epoch += lossG_l1_iter.item()
      lossG_iou_epoch += lossG_iou_iter.item()
      lossG_epoch += lossG_iter.item()
      pixel_acc_epoch += pixel_acc_iter.item()
      iou_epoch += iou_iter.item()

    log_info_epoch = {
      'val/lossG_l1/epoch': lossG_l1_epoch / len(val_loader),
      'val/lossG_iou/epoch': lossG_iou_epoch / len(val_loader),
      'val/lossG/epoch': lossG_epoch / len(val_loader),
      'val/pixel_acc/epoch': pixel_acc_epoch / len(val_loader),
      'val/iou/epoch': iou_epoch / len(val_loader)
    }

    log_message_epoch = (f"Losses per epoch [{current_epoch}/{num_epochs}], "
                        f"L1 Loss: {log_info_epoch['val/lossG_l1/epoch']:.4f}, "
                        f"IoU Loss: {log_info_epoch['val/lossG_iou/epoch']:.4f}, "
                        f"Total Loss: {log_info_epoch['val/lossG/epoch']:.4f}, "
                        f"Pixel Accuracy: {log_info_epoch['val/pixel_acc/epoch']:.4f}, "
                        f"IoU: {log_info_epoch['val/iou/epoch']:.4f}")

    # Print and log the message
    print(log_message_epoch)
    logging.info(log_message_epoch)

    if LOG_WANDB:
      wandb.log(log_info_epoch)

  # return average total loss, average pixel accuracy and average iou per epoch
  return log_info_epoch['val/lossG/epoch'], log_info_epoch['val/pixel_acc/epoch'], log_info_epoch['val/iou/epoch']

def evaluate_model(model,
                   loader,
                   device='cuda',
                   log = False):
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


def train_model(model,
                train_loader,
                val_loader,
                num_epochs,
                lr = 2e-4,
                device = "cuda",
                start_epoch = 0,
                checkpoint_dir = "checkpoints",
                resume = False):
  # Load checkpoint if resuming
  if resume:
    checkpoint = torch.load('/mnt/data/amoskovtsev/mb_opc/checkpoints/exp_10/last_checkpoint.pth') # torch.load(os.path.join(checkpoint_dir, "checkpoint_14.10.24.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = lr_sched.CosineAnnealingLR(optimizer = optimizer, T_max = 20, eta_min = 1e-7)
    start_epoch = checkpoint['epoch'] + 1
    num_epochs += start_epoch
    print(f"Resuming training from epoch {start_epoch}")
    logging.info(f"Resuming training from epoch {start_epoch}")
    model.to(device)
  else:
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-5)
    # scheduler = lr_sched.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)
    scheduler = lr_sched.CosineAnnealingLR(optimizer = optimizer, T_max = 30, eta_min = 1e-7)
    print('Starting experiment...')
    logging.info('Starting experiment...')
  lossG_epoch_list_train = []
  lossG_epoch_list_val = []
  lossG_iter_list = []

  pixel_acc_epoch_list_train = []
  pixel_acc_epoch_list_val = []

  iou_epoch_list_train = []
  iou_epoch_list_val = []
  model.to(device)
  torch.autograd.set_detect_anomaly(True)
  for epoch in range(start_epoch, num_epochs):
    lossG_l1_epoch = 0
    lossG_iou_epoch = 0
    lossG_epoch = 0
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
      lossG_l1_iter = l1_loss(mask, target)
      lossG_iou_iter = iou_loss(mask, target)
      lossG_iter = lossG_l1_iter + lossG_iou_iter
      lossG_iter_list.append(lossG_iter.item())

      # calculate metrics during train phase
      pixel_acc_iter = pixel_accuracy(mask, target)
      iou_iter = iou(mask, target)

      optimizer.zero_grad()
      lossG_iter.backward()
      optimizer.step()

      if idx % 50 == 0:
        # Create a dictionary to hold your log information
        log_info_iter = {
          'train/lossG_l1/iter': lossG_l1_iter.item(),
          'train/lossG_iou/iter': lossG_iou_iter.item(),
          'train/lossG/iter': lossG_iter.item(),
          'train/pixel_acc/iter': pixel_acc_iter.item(),
          'train/iou/iter': iou_iter.item()
        }

        log_message_iter = (f"Epoch [{epoch}/{num_epochs}], "
                       f"Step [{idx}/{len(train_loader)}], "
                       f"L1 Loss: {log_info_iter['train/lossG_l1/iter']:.4f}, "
                       f"IoU Loss: {log_info_iter['train/lossG_iou/iter']:.4f}, "
                       f"Total Loss: {log_info_iter['train/lossG/iter']:.4f}, "
                       f"Pixel Accuracy: {log_info_iter['train/pixel_acc/iter']:.4f}, "
                       f"IoU: {log_info_iter['train/iou/iter']:.4f} ")

        print(log_message_iter)
        logging.info(log_message_iter)

        if LOG_WANDB:
          wandb.log(log_info_iter)

        # draw loss plot per iteration
        draw_plot(first_variable = lossG_iter_list, label = 'train_iter_loss',
                  title = 'Iteration loss plot', xlabel = 'Iteration',
                  ylabel = 'Loss', save_name = 'train_iter_loss.jpg',
                  checkpoint_dir = checkpoint_dir)

      if idx % 200 == 0:
        # save generated mask
        save_generated_image(mask, epoch, idx, checkpoint_dir = checkpoint_dir, image_type = 'generated_correction')

      lossG_l1_epoch += lossG_l1_iter.item()
      lossG_iou_epoch += lossG_iou_iter.item()
      lossG_epoch += lossG_iter.item()

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
      'train/lossG_l1/epoch': lossG_l1_epoch / len(train_loader),
      'train/lossG_iou/epoch': lossG_iou_epoch / len(train_loader),
      'train/lossG/epoch': lossG_epoch / len(train_loader),
      'train/pixel_acc/epoch': pixel_acc_epoch / len(train_loader),
      'train/iou/epoch': iou_epoch / len(train_loader)
    }

    log_message_epoch = (f"Losses per epoch [{epoch}/{num_epochs}], "
                        f"L1 Loss: {log_info_epoch['train/lossG_l1/epoch']:.4f}, "
                        f"IoU Loss: {log_info_epoch['train/lossG_iou/epoch']:.4f}, "
                        f"Total Loss: {log_info_epoch['train/lossG/epoch']:.4f}"
                        f"Pixel Accuracy: {log_info_epoch['train/pixel_acc/epoch']:.4f}"
                        f"IoU: {log_info_epoch['train/iou/epoch']:.4f}")

    print(log_message_epoch)
    logging.info(log_message_epoch)

    if LOG_WANDB:
      wandb.log(log_info_epoch)

    # get average loss, pixel accuracy and iou per epoch during training phase
    lossG_epoch_list_train.append(log_info_epoch['train/lossG/epoch'])
    pixel_acc_epoch_list_train.append(log_info_epoch['train/pixel_acc/epoch'])
    iou_epoch_list_train.append(log_info_epoch['train/iou/epoch'])

    # get average loss, pixel accuracy and iou per epoch during validation phase
    lossG_epoch_val, pixel_acc_epoch_val, iou_epoch_val = validate_model(model,
                                                                         val_loader,
                                                                         current_epoch = epoch,
                                                                         num_epochs = num_epochs,
                                                                         checkpoint_dir = checkpoint_dir,
                                                                         device = device)
    lossG_epoch_list_val.append(lossG_epoch_val)
    iou_epoch_list_val.append(iou_epoch_val)
    pixel_acc_epoch_list_val.append(pixel_acc_epoch_val)

    # draw epoch losses for training and validation phases
    draw_plot(first_variable = lossG_epoch_list_train, second_variable = lossG_epoch_list_val,
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


if LOG_WANDB:
  wandb.login()
  wandb.init(
    project = "MB-OPC",
    name = "DAMO generator",
    config = {
      "architecture": "DAMO Generator, UpSample + Conv2d in decoder",
      "optimizer": "Adam",
      "optimizer_parameters": 'learning_rate - 2e-4, weight_decay - 1e-5',
      "scheduler" : "CosineAnnealing",
      "scheduler_parameters": "T_max - 30, eta_min - 1e-7",
      "learning_rate": LEARNING_RATE,
      "epochs" : EPOCHS,
      "dataset": "1024x1024 grayscale images",
      "description": "DAMO generator with Upsample + Conv2d in decoder, train from scratch"
    }
  )
  
CHECKPOINT_DIR = next_exp_folder(CHECKPOINT_PATH)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Experiment logs will be saved in: {CHECKPOINT_DIR}')
setup_logging(CHECKPOINT_DIR)
logging.info(f'Experiment logs will be saved in: {CHECKPOINT_DIR}')
logging.info(f'Training device:{DEVICE}')

generator_model = Generator(in_ch = 1, out_ch = 1)
generator_model = generator_model.to(DEVICE)
print('Model initialized:', generator_model)
logging.info('Model initialized')


logging.info(f'Batch size:{BATCH_SIZE}')
# Define dataset
TRAIN_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/train_origin'), os.path.join(DATASET_PATH,'correction/train_correction'), transform = apply_transform(binarize_flag = True))
VALID_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/valid_origin'), os.path.join(DATASET_PATH, 'correction/valid_correction'), transform = apply_transform(binarize_flag = True))
TEST_DATASET = OPCDataset(os.path.join(DATASET_PATH, 'origin/test_origin'), os.path.join(DATASET_PATH, 'correction/test_correction'), transform = apply_transform(binarize_flag = True))

# Define dataloader
TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)
VALID_LOADER = DataLoader(VALID_DATASET, batch_size = BATCH_SIZE, shuffle = False, num_workers = 8)
TEST_LOADER = DataLoader(TEST_DATASET, batch_size = BATCH_SIZE, shuffle = False, num_workers = 8)

print(f'Number of images in train subset:{len(TRAIN_DATASET)}')
print(f'Number of images in valid subset:{len(VALID_DATASET)}')
print(f'Number of images in test subset:{len(TEST_DATASET)}\n')

logging.info(f'Number of images in train subset:{len(TRAIN_DATASET)}')
logging.info(f'Number of images in valid subset:{len(VALID_DATASET)}')
logging.info(f'Number of images in test subset:{len(TEST_DATASET)}\n')

# Define loss functions
iou_loss = IouLoss(weight=1.0)
l1_loss = torch.nn.L1Loss()

# Define metrics
pixel_accuracy = PixelAccuracy()
iou = IoU()

if DEBUG_LOSS:
  image, target = next(iter(TRAIN_LOADER))
  image, target = image.to(DEVICE), target.to(DEVICE)
  print(f'Image shape: {image.shape}')
  print(f'Target shape: {target.shape}')
  print(f'Image shape after removing batch dimension: {image[0,0].shape}')

  # Выполним проверку подсчёта функций потерь
  output = generator_model(image)
  print(f'Output shape: {output.shape}')

  iou_loss_value = iou_loss(target, output.sigmoid())
  l1_loss_value = l1_loss(target, output.sigmoid())

  print(f'IoU loss:{iou_loss_value}')
  print(f'L1-loss:{l1_loss_value}')
  print(f'IoU loss shape:{iou_loss_value.shape}')
  print(f'L1-loss shape:{l1_loss_value.shape}')

start_train = time.time()
# Train the model
train_model(model = generator_model,
            train_loader = TRAIN_LOADER,
            val_loader = VALID_LOADER,
            num_epochs = EPOCHS,
            lr = LEARNING_RATE,
            device = DEVICE,
            start_epoch = 0,
            checkpoint_dir = CHECKPOINT_DIR,
            resume = RESUME)
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

if LOG_WANDB:
  wandb.finish()