import os

if os.getenv('ENV') == 'TRAIN_MACHINE':
  DATASET_PATH = '/home/amoskovtsev/projects/mb_opc/data/processed/gds_dataset'
  CHECKPOINT_PATH = '/mnt/data/amoskovtsev/mb_opc/checkpoints'
  CLASSIFACTION_DATA_PATH = '/home/amoskovtsev/projects/mb_opc/data/processed/gds_dataset/classifier'

else:
  DATASET_PATH = '/workarea/otdMDP/users/amoskovtsev/Pycharm_proj/MB_OPC/custom_unet/data/processed/gds_dataset'
  CLASSIFACTION_DATA_PATH = '/workarea/otdMDP/users/amoskovtsev/Pycharm_proj/MB_OPC/custom_unet/data/processed/classifier'
  CHECKPOINT_PATH = 'checkpoints'

BATCH_SIZE = 3
EPOCHS = 20
LEARNING_RATE = 2e-4
LOG_WANDB = True
RESUME = True