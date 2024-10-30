import cv2
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import logging
import os

from models.model import Generator
from data.dataset import OPCDataset, BinarizeTransform

def save_generated_image(output, epoch, step, checkpoint_dir="checkpoints", image_type='true_correction'):
  # # Convert from [-1, 1] to [0, 255] for saving
  # output = 0.5 * (output + 1)  # Brings values to [0, 1] range

  single_image = output[0].squeeze(dim = 0)
  single_image[single_image > 0.5] = 1.0
  single_image[single_image <= 0.5] = 0.0

  img_save_path = os.path.join(checkpoint_dir, f"{image_type}_epoch{epoch}_step{step}.png")
  cv2.imwrite(f"{img_save_path}", (single_image * 255).detach().cpu().numpy())
  print(f"Saved generated image at {img_save_path}")
  logging.info(f"Saved generated image at {img_save_path}")

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

def setup_logging(exp_folder):
  # Set up logging configuration
  log_file_path = os.path.join(exp_folder, 'training_log.txt')
  logging.basicConfig(filename = log_file_path,
                      level = logging.INFO,
                      datefmt = '%d/%m/%Y %H:%M')
  # logging.info("Training log started")

def iou_loss(pred, target, eps=1e-6):
  intersection = (pred * target).sum(dim=(2, 3))
  union = (pred + target).sum(dim=(2, 3)) - intersection
  iou = (intersection + eps) / (union + eps)
  return 1 - iou.mean()

criterion_mse = torch.nn.MSELoss()

def validate_model(model, val_loader, current_epoch, num_epochs ,checkpoint_dir,device='cuda'):
  model.eval()
  l2_loss_epoch = 0
  iou_loss_epoch = 0
  total_loss_epoch = 0
  # criterion_mse = nn.MSELoss()
  print(f"[Epoch {current_epoch}] Validating")
  logging.info(f"[Epoch {current_epoch}] Validating")

  with torch.no_grad():
    for idx, (image, target) in enumerate(val_loader):
      image, target = image.to(device), target.to(device)

      params = model(image)
      mask = torch.sigmoid(params)
      l2_loss_iter = criterion_mse(mask, target) # nn.MSELoss(mask, target)
      iou_loss_iter = iou_loss(mask, target)
      total_loss_iter = l2_loss_iter + iou_loss_iter
      # total_loss_iter_list.append(total_loss_iter.item())

      if idx % 10 == 0:
        print(
          f"Epoch [{current_epoch}/{num_epochs}], Step [{idx}/{len(val_loader)}], L2 Loss: {l2_loss_iter.item():.4f}, IoU Loss: {iou_loss_iter.item():.4f}, Total Loss: {total_loss_iter.item():.4f}")
        logging.info(f"Epoch [{current_epoch}/{num_epochs}], Step [{idx}/{len(val_loader)}], L2 Loss: {l2_loss_iter.item():.4f}, IoU Loss: {iou_loss_iter.item():.4f}, Total Loss: {total_loss_iter.item():.4f}")
      if idx % 200 == 0:
        # save_generated_image(target, epoch, idx, checkpoint_dir=checkpoint_dir, image_type='true_correction')
        save_generated_image(mask, current_epoch, idx, checkpoint_dir=checkpoint_dir, image_type='generated_correction_val')

      l2_loss_epoch += l2_loss_iter.item()
      iou_loss_epoch += iou_loss_iter.item()
      total_loss_epoch += total_loss_iter.item()


    print(
      f'[Losses per {current_epoch}/{num_epochs} epoch:] L2 loss={l2_loss_epoch/len(val_loader):.4f}, IoU loss={iou_loss_epoch/len(val_loader):.4f}, Total loss={total_loss_epoch/len(val_loader):.4f}')
    logging.info(f'[Losses per {current_epoch}/{num_epochs} epoch:] L2 loss={l2_loss_epoch/len(val_loader):.4f}, IoU loss={iou_loss_epoch/len(val_loader):.4f}, Total loss={total_loss_epoch/len(val_loader):.4f}')

  return total_loss_epoch / len(val_loader)

def pretrain_model(model, train_loader, val_loader, num_epochs, lr=2e-4, device="cuda", start_epoch=0,
                  checkpoint_dir="checkpoints", resume=False):
  # Loss functions
  # criterion_mse = nn.MSELoss()
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-5)
  scheduler = lr_sched.StepLR(optimizer = optimizer, step_size = 2, gamma = 0.1)
  print('Starting experiment...')
  logging.info('Starting experiment...')

  # Load checkpoint if resuming
  if resume:
    checkpoint = torch.load('/home/amoskovtsev/MBOPC/custom_unet/checkpoints/checkpoint_14.10.24.pth') # torch.load(os.path.join(checkpoint_dir, "checkpoint_14.10.24.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")
    logging.info(f"Resuming training from epoch {start_epoch}")

  total_loss_epoch_list_train = []
  total_loss_epoch_list_val = []
  total_loss_iter_list = []

  for epoch in range(start_epoch, num_epochs):
    l2_loss_epoch = 0
    iou_loss_epoch = 0
    total_loss_epoch = 0

    print(f"[Epoch {epoch}] Training")
    logging.info(f"[Epoch {epoch}] Training")
    model.to(device)
    model.train()

    progress = tqdm(train_loader)

    for idx, (image, target) in enumerate(progress):
      image, target = image.to(device), target.to(device)
      params = model(image)
      mask = torch.sigmoid(params)

      l2_loss_iter = criterion_mse(mask, target)
      iou_loss_iter = iou_loss(mask, target)
      total_loss_iter = l2_loss_iter + iou_loss_iter
      total_loss_iter_list.append(total_loss_iter.item())

      optimizer.zero_grad()
      total_loss_iter.backward()
      optimizer.step()

      if idx % 20 == 0:
        print(
          f"Epoch [{epoch}/{num_epochs}], Step [{idx}/{len(train_loader)}], L2 Loss: {l2_loss_iter.item():.4f}, IoU Loss: {iou_loss_iter.item():.4f}, Total Loss: {total_loss_iter.item():.4f}")
        logging.info(f"Epoch [{epoch}/{num_epochs}], Step [{idx}/{len(train_loader)}], L2 Loss: {l2_loss_iter.item():.4f}, IoU Loss: {iou_loss_iter.item():.4f}, Total Loss: {total_loss_iter.item():.4f}")

      if idx % 200 == 0:
        # save_generated_image(target, epoch, idx, checkpoint_dir=checkpoint_dir, image_type='true_correction')
        save_generated_image(mask, epoch, idx, checkpoint_dir = checkpoint_dir, image_type = 'generated_correction')

      l2_loss_epoch += l2_loss_iter.item()
      iou_loss_epoch += iou_loss_iter.item()
      total_loss_epoch += total_loss_iter.item()

      if idx % 500 == 0:
        # checkpoint_path = os.path.join(, "last_checkpoint.pth")
        checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint at {checkpoint_path}")
        logging.info(f"Saved checkpoint at {checkpoint_path}")

    print(
      f'[Losses per {epoch}/{num_epochs} epoch:] L2 loss={l2_loss_epoch/len(train_loader):.4f}, IoU loss={iou_loss_epoch/len(train_loader):.4f}, Total loss={total_loss_epoch/len(train_loader):.4f}')
    logging.info(f'[Losses per {epoch}/{num_epochs} epoch:] L2 loss={l2_loss_epoch/len(train_loader):.4f}, IoU loss={iou_loss_epoch/len(train_loader):.4f}, Total loss={total_loss_epoch/len(train_loader):.4f}')
    total_loss_epoch_list_train.append(total_loss_epoch / len(train_loader))
	
    total_loss_epoch_val = validate_model(model, val_loader, current_epoch = epoch, num_epochs=num_epochs, checkpoint_dir = checkpoint_dir, device = device)
    total_loss_epoch_list_val.append(total_loss_epoch_val)

    plt.figure(figsize=(8, 6))
    plt.plot(total_loss_epoch_list_train, marker='o', linestyle='-', label='train_loss')
    plt.plot(total_loss_epoch_list_val, marker='o', linestyle='-', color='b', label='valid_loss')
    plt.title('Epoch loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'epoch_loss.jpg'))
    plt.close()

    scheduler.step()

  print('Traning complete')
  logging.info('Traning complete')

CHECKPOINT_DIR = get_next_experiment_folder('checkpoints')
print(f'Experiment logs will be saved in: {CHECKPOINT_DIR}')
setup_logging(CHECKPOINT_DIR)
logging.info(f'Experiment logs will be saved in: {CHECKPOINT_DIR}')

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info(f'Training device:{DEVICE}')

generator_model = Generator(in_ch = 1, out_ch = 1)
print('Model initialized:', generator_model)
logging.info('Model initialized')

test_image = torch.randn((1,3,1024,1024))
output = generator_model.forward(test_image)
print(f'Output shape: {output.shape}')
print(f'Output shape: {torch.sigmoid(output).shape}')

test_image = torch.randn((1,3,1024,1024))
# summary(generator, (3,1024,1024))

TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Grayscale(),
    BinarizeTransform(threshold=0.5)
])

BATCH_SIZE = 3
logging.info(f'Batch size:{BATCH_SIZE}')
# Define dataset
TRAIN_DATASET = OPCDataset("data/processed/gds_dataset/origin/train_origin", "data/processed/gds_dataset/correction/train_correction", transform = TRANSFORM)
VALID_DATASET = OPCDataset("data/processed/gds_dataset/origin/valid_origin", "data/processed/gds_dataset/correction/valid_correction", transform = TRANSFORM)
TEST_DATASET = OPCDataset("data/processed/gds_dataset/origin/test_origin", "data/processed/gds_dataset/correction/test_correction", transform = TRANSFORM)

# Define dataloader
TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size = BATCH_SIZE, shuffle = True)
VALID_LOADER = DataLoader(VALID_DATASET, batch_size = BATCH_SIZE, shuffle = False)
TEST_LOADER = DataLoader(TEST_DATASET, batch_size = BATCH_SIZE, shuffle = False)

print(f'Number of images in train subset:{len(TRAIN_DATASET)}\n')
print(f'Number of images in valid subset:{len(VALID_DATASET)}\n')
print(f'Number of images in test subset:{len(TEST_DATASET)}\n')
image, target = next(iter(TRAIN_LOADER))
print(f'Image shape: {image.shape}')
print(f'Target shape: {target.shape}')
print(f'Image shape after removing batch dimension: {image[0,0].shape}')

# Выполним проверку подсчёта функций потерь
output = generator_model(image)
print(f'Output shape: {output.shape}')
iou_loss_value = iou_loss(target, output.sigmoid())
mse_loss_value = torch.nn.functional.mse_loss(output.sigmoid(), target)

print(f'IoU loss:{iou_loss_value}')
print(f'L1-loss:{mse_loss_value}')
print(f'IoU loss shape:{iou_loss_value.shape}')
print(f'L1-loss shape:{mse_loss_value.shape}')

plt.imshow(target[0,0], cmap='gray')
plt.show()
# Train the model
pretrain_model(model = generator_model,
               train_loader = TRAIN_LOADER,
               val_loader = VALID_LOADER,
               num_epochs = 15,
               lr = 2e-4,
               device = DEVICE,
               start_epoch = 0,
               checkpoint_dir = CHECKPOINT_DIR,
               resume=False)
