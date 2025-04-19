import torch
import pytorch_lightning as pl
from hydra.utils import instantiate

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

    def validation_step(self, batch):
        origin, target = batch
        pred = torch.sigmoid(self(origin))
        loss, loss_dict = self.compute_loss(pred, target)
        iou = self.iou(pred, target)
        pixel_acc = self.pixel_acc(pred, target)

        for k, v in loss_dict.items():
            self.log(
                f"val/lossG_{k}/epoch", v, prog_bar=True, on_step=False, on_epoch=True
            )
        self.log(f"val/pixel_acc/epoch", pixel_acc, on_step=False, on_epoch=True)
        self.log(f"val/iou/epoch", iou, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

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
