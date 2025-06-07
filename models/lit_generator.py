import os

import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from torchvision.utils import save_image

from src.metrics import IoU, PixelAccuracy


class LitGenerator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.model = instantiate(config["model"])
        self.criterion = instantiate(config["loss"])
        self.iou = IoU()
        self.pixel_acc = PixelAccuracy()
        self.val_example_logged = (
            False  # flag for logging images during validation phase
        )

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, pred, target):
        # compute the sum of loss components for optimization
        # store different loss components in dict for logging
        losses = self.criterion(pred, target)
        if isinstance(losses, dict):
            total_loss = sum(losses.values())
            return total_loss, losses
        else:
            return losses, {"loss": losses}

    def training_step(self, batch, batch_idx):
        origin, target = batch
        pred = torch.sigmoid(self(origin))
        loss, loss_dict = self.compute_loss(pred, target)
        iou = self.iou(pred, target)
        pixel_acc = self.pixel_acc(pred, target)

        # separate logging for all loss components
        for k, v in loss_dict.items():
            self.log(
                f"train/lossG_{k}/epoch", v, prog_bar=True, on_step=False, on_epoch=True
            )
        self.log("train/pixel_acc/epoch", pixel_acc, on_step=False, on_epoch=True)
        self.log("train/iou/epoch", iou, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        origin, target = batch
        pred = torch.sigmoid(self(origin))
        loss, loss_dict = self.compute_loss(pred, target)
        iou = self.iou(pred, target)
        pixel_acc = self.pixel_acc(pred, target)
        # save first prediction on valid data
        if not self.val_example_logged and batch_idx == 0:
            save_image(
                pred,
                os.path.join(
                    self.logger.log_dir, f"pred_epoch{self.current_epoch}.png"
                ),
            )

            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log(
                    {f"val/sample_epoch{self.current_epoch}": wandb.Image(pred[0])}
                )

            self.val_example_logged = True

        for k, v in loss_dict.items():
            self.log(
                f"val/lossG_{k}/epoch", v, prog_bar=True, on_step=False, on_epoch=True
            )
        self.log("val/pixel_acc/epoch", pixel_acc, on_step=False, on_epoch=True)
        self.log("val/iou/epoch", iou, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        origin, target = batch
        pred = torch.sigmoid(self(origin))
        loss, loss_dict = self.compute_loss(pred, target)
        iou = self.iou(pred, target)
        pixel_acc = self.pixel_acc(pred, target)
        for k, v in loss_dict.items():
            self.log(
                f"test/lossG_{k}/epoch", v, prog_bar=True, on_step=False, on_epoch=True
            )
        self.log("test/pixel_acc/epoch", pixel_acc, on_step=False, on_epoch=True)
        self.log("test/iou/epoch", iou, on_step=False, on_epoch=True)

        return loss

    def on_validation_start(self):
        self.val_example_logged = False

    def configure_optimizers(self):
        # instantiate optimizer (complete the partial)
        optimizer_fn = instantiate(self.config["optim"])
        optimizer = optimizer_fn(self.parameters())
        # instantiate scheduler (complete the partial)
        scheduler_fn = instantiate(self.config["sched"])
        scheduler = scheduler_fn(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or 'step' if needed
                "frequency": 1,
            },
        }
