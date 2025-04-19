import torch
import torch.nn as nn

from scipy.ndimage import distance_transform_edt


def compute_distance_map(mask):
    """
    Compute dostance map for the binary mask.
    For each pixel, the distance map gives a distance to the nearest boundary
    :param mask:
    :return:
    """
    mask = mask.cpu().numpy()
    distance_map = distance_transform_edt(mask) + distance_transform_edt(1 - mask)

    return torch.tensor(distance_map).float()


class BoundaryLoss(nn.Module):
    def __init__(self, weight=1.0, device="cpu"):
        super(BoundaryLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, pred, target):
        # Convert target to distance maps
        dist_maps = torch.stack([compute_distance_map(t) for t in target]).to(
            self.device
        )
        # Compute boundary loss
        boundary_loss = torch.mean(pred * dist_maps)

        return self.weight * boundary_loss


class ContourLoss(nn.Module):
    def __init__(self, weight=1.0, device="cpu"):
        super(ContourLoss, self).__init__()
        self.weight = weight
        self.device = device
        self.sobel_kernel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device
        ).view(1, 1, 3, 3)
        self.sobel_kernel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device
        ).view(1, 1, 3, 3)

    def forward(self, pred, target):
        target_edge_x = torch.nn.functional.conv2d(
            target, self.sobel_kernel_x, padding=1
        )
        target_edge_y = torch.nn.functional.conv2d(
            target, self.sobel_kernel_y, padding=1
        )
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)

        pred_edge_x = torch.nn.functional.conv2d(pred, self.sobel_kernel_x, padding=1)
        pred_edge_y = torch.nn.functional.conv2d(pred, self.sobel_kernel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)

        # calculate difference for every predicted and target correction in a batch
        loss_per_image = torch.abs(pred_edge - target_edge).sum(
            dim=(1, 2, 3)
        )  # (N, ), sum over height, width and channels
        # calculate the contour area for every target correction in a batch
        target_edge_sum = (
            target_edge.sum(dim=(1, 2, 3)) + 1e-6
        )  # (N, ) sum over height, width, and prevent division by zero
        # normalise the difference by the contour area of target correction
        mean_loss = (loss_per_image / target_edge_sum) * self.weight

        # return mean value over a batch
        return mean_loss.mean()  # (),


class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        tv_loss = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) + torch.sum(
            torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        )

        return self.weight * tv_loss


class MobileNetPerceptualLoss(nn.Module):
    def __init__(self, feature_extractor, selected_layers, visualize=False):
        super(MobileNetPerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.selected_layers = selected_layers
        self.feature_outputs = {}
        self.visualize = visualize

        # Register forward hooks for the selected layers

        for layer_idx in self.selected_layers:
            self.feature_extractor.features[layer_idx].register_forward_hook(
                self.save_output(layer_idx)
            )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def save_output(self, layer_idx):
        """
        Hook to save the output of the specified layer
        """

        def hook(module, input, output):
            self.feature_outputs[layer_idx] = output

        return hook

    def forward(self, img1, img2):
        # Forward pass through feature extractor
        _ = self.feature_extractor(img1)
        features1 = self.feature_outputs.copy()

        _ = self.feature_extractor(img2)
        features2 = self.feature_outputs.copy()

        # Compute perceptual loss as a L1-disttance between corresponding layers
        loss = 0
        for layer_idx in self.selected_layers:
            loss_iter = torch.nn.functional.l1_loss(
                features1[layer_idx], features2[layer_idx]
            )
            loss += loss_iter

        return loss


class IouLoss(nn.Module):
    def __init__(self, weight=1.0, eps=1e-6):
        super(IouLoss, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return (1 - iou.mean()) * self.weight


class IouBceLoss(nn.Module):
    def __init__(self, iou_weight, bce_weight):
        super(IouBceLoss, self).__init__()
        self.iou_weight = iou_weight
        self.bce_weight = bce_weight
        self.iou_loss = IouLoss()
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, pred, target):
        iou_loss = self.iou_loss(pred, target) * self.iou_weight
        bce_loss = self.bce_loss(pred, target) * self.bce_weight
        return {"bce": bce_loss, "iou": iou_loss}
