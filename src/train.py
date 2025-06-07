import os
import sys

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from models.lit_generator import LitGenerator
from src.dataset import OPCDataset, apply_transform

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def train(cfg: DictConfig):
    # We simply print the configuration
    # print(OmegaConf.to_yaml(cfg))
    DATASET_PATH = cfg["env"]["paths"]["dataset"]
    BATCH_SIZE = cfg["training"]["batch_size"]
    EPOCHS = cfg["training"]["epochs"]
    LOG_DIR = cfg["env"]["paths"]["checkpoint"]
    DEVICE = cfg["training"]["device"]
    NUM_WORKERS = cfg["training"]["num_workers"]

    TRAIN_DATASET = OPCDataset(
        os.path.join(DATASET_PATH, "origin/train_origin/"),
        os.path.join(DATASET_PATH, "correction/train_correction/"),
        transform=apply_transform(binarize_flag=True),
    )
    VALID_DATASET = OPCDataset(
        os.path.join(DATASET_PATH, "origin/valid_origin/"),
        os.path.join(DATASET_PATH, "correction/valid_correction/"),
        transform=apply_transform(binarize_flag=True),
    )
    TEST_DATASET = OPCDataset(
        os.path.join(DATASET_PATH, "origin/test_origin/"),
        os.path.join(DATASET_PATH, "correction/test_correction/"),
        transform=apply_transform(binarize_flag=True),
    )

    # Define dataloader
    TRAIN_LOADER = DataLoader(
        TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    VALID_LOADER = DataLoader(
        VALID_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    TEST_LOADER = DataLoader(
        TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    lit_gen = LitGenerator(config=cfg)

    early_stopping = EarlyStopping(
        monitor="val/iou/epoch",
        mode="max",
        patience=cfg["training"]["patience"],
        min_delta=cfg["training"]["min_delta"],
        verbose=True,
    )

    csv_logger = CSVLogger(save_dir=LOG_DIR)
    wandb.login(key=cfg["logging"]["key"])
    wandb_logger = WandbLogger(
        project=cfg["logging"]["project"], name=cfg["logging"]["name"], log_model=False
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val/iou/epoch",
        mode="max",
        save_top_k=cfg["training"]["save_top_k"],
        filename="best_checkpoint",
        dirpath=csv_logger.log_dir,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=cfg["logging"]["log_every_n_steps"],
        accelerator=DEVICE,
        callbacks=[early_stopping, checkpoint_callback],
        logger=[csv_logger, wandb_logger],
    )

    trainer.fit(lit_gen, TRAIN_LOADER, VALID_LOADER)

    print("Calculating validation metrics:")
    val_result = trainer.test(lit_gen, dataloaders=VALID_LOADER)
    print("Validation result:", val_result)
    print("Calculating test metrics:")
    test_result = trainer.test(lit_gen, dataloaders=TEST_LOADER)
    print("Test result:", test_result)


if __name__ == "__main__":
    train()
