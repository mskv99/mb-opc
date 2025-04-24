import os
from typing import Any

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.utils import save_image
from hydra.utils import instantiate

from src.metrics import IoU, PixelAccuracy


class LitGenerator(pl.LightningModule):
    def __init__(self, config, log_dir):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.model = instantiate(config["model"])
        self.criterion = instantiate(config["loss"])
        self.iou = IoU()
        self.pixel_acc = PixelAccuracy()
        self.val_example_logged = False
        self.log_dir = log_dir

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, pred, target):
        losses = self.criterion(pred, target)
        if isinstance(losses, dict):
            total_loss = sum(losses.values())
            return total_loss, losses
        else:
            return losses, {"loss": losses}  # handle non-dict criterion

    def training_step(self, batch, batch_idx):
        origin, target = batch
        pred = torch.sigmoid(self(origin))
        loss, loss_dict = self.compute_loss(pred, target)
        iou = self.iou(pred, target)
        pixel_acc = self.pixel_acc(pred, target)

        # log all loss components
        for k, v in loss_dict.items():
            self.log(
                f"train/lossG_{k}/epoch", v, prog_bar=True, on_step=False, on_epoch=True
            )
        self.log(f"train/pixel_acc/epoch", pixel_acc, on_step=False, on_epoch=True)
        self.log(f"train/iou/epoch", iou, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        origin, target = batch
        pred = torch.sigmoid(self(origin))
        loss, loss_dict = self.compute_loss(pred, target)
        iou = self.iou(pred, target)
        pixel_acc = self.pixel_acc(pred, target)
        # Сохраняем первый пример на валидации
        if not self.val_example_logged and batch_idx == 0:
            save_image(
                pred,
                os.path.join(
                    self.log_dir, f"pred_epoch{self.current_epoch}.png"
                ),
            )

            if isinstance(self.logger, pl.loggers.WandbLogger):
                columns = ["target_correction", "predicted_correction"]
                data = [[wandb.Image(target[0]), wandb.Image(pred[0])]]
                self.logger.log_table(
                    key="sample_table", columns=columns, data=data
                )

            self.val_example_logged = True

        for k, v in loss_dict.items():
            self.log(
                f"val/lossG_{k}/epoch", v, prog_bar=True, on_step=False, on_epoch=True
            )
        self.log(f"val/pixel_acc/epoch", pixel_acc, on_step=False, on_epoch=True)
        self.log(f"val/iou/epoch", iou, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        origin, target = batch
        pred = torch.sigmoid(self(origin))
        loss, loss_dict = self.compute_loss(pred, target)
        iou = self.iou(pred, target)
        pixel_acc = self.pixel_acc(pred, target)
        return {
            "iou": iou,
            "pixel_acc": pixel_acc,
        }

    def on_validation_start(self):
        self.val_example_logged = False

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self, outputs) -> None:
        ious = torch.stack([o["iou"] for o in outputs])
        pixel_accs = torch.stack([o["pixel_acc"] for o in outputs])
        avg_iou = ious.mean().item()
        avg_pixel_acc = pixel_accs.mean().item()

        return {"iou": avg_iou, "pixel_accuracy": avg_pixel_acc}

    def configure_optimizers(self):
        # 1. Instantiate optimizer (complete the partial)
        optimizer_fn = instantiate(self.config["optim"])
        optimizer = optimizer_fn(self.parameters())
        # 2. Instantiate scheduler (complete the partial)
        scheduler_fn = instantiate(self.config["sched"])
        scheduler = scheduler_fn(optimizer=optimizer)

        # 3. Return in Lightning's expected format
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or 'step' if needed
                "frequency": 1,
            },
        }
