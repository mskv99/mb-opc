import torch
import torch.nn as nn


class IoU(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoU, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - intersection
        iou = (intersection + self.eps) / (union + self.eps)

        return iou.mean()


class PixelAccuracy(nn.Module):
    def __init__(self, eps=1e-6):
        super(PixelAccuracy, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.clone()
        pred[pred > 0.5] = 1.0
        pred[pred <= 0.5] = 0.0
        correct = (pred == target).float().sum()
        total = torch.numel(target)
        pixel_acc = (correct + self.eps) / (total + self.eps)

        return pixel_acc
