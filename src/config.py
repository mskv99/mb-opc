import os

if os.getenv('ENV') == 'TRAIN_MACHINE':
  DATASET_PATH = '/home/amoskovtsev/projects/mb_opc/data/processed/gds_dataset'
  CHECKPOINT_PATH = '/mnt/data/amoskovtsev/mb_opc/checkpoints'

else:
  DATASET_PATH = 'data/processed/gds_dataset'
  CHECKPOINT_PATH = 'checkpoints'

BATCH_SIZE = 3
EPOCHS = 25
LEARNING_RATE = 2e-4
