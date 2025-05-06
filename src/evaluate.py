import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import fire
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import OPCDataset, apply_transform
from src.metrics import IoU, PixelAccuracy
from src.utils import load_model, set_random_seed


def evaluate_model(model, loader, device="cuda", log=False):
    model.eval()

    pixel_accuracy = PixelAccuracy()
    iou = IoU()

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
            "pixel_acc": pixel_acc_epoch,
            "iou": iou_epoch,
            "len_loader": len(loader),
        }

        # Print and log the message
    print(f"Pixel Accuracy: {log_info['pixel_acc'] / log_info['len_loader'] :.4f}, ")
    print(f"IoU: {log_info['iou'] / log_info['len_loader'] :.4f}")

    if log:
        logging.info(
            f"Pixel Accuracy: {log_info['pixel_acc'] / log_info['len_loader'] :.4f}, "
        )
        logging.info(f"IoU: {log_info['iou'] / log_info['len_loader'] :.4f}")

    return (
        log_info["pixel_acc"] / log_info["len_loader"],
        log_info["iou"] / log_info["len_loader"],
    )


def main(model_type, weights, batch_size, subset):

    set_random_seed(42)

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available() and model_type not in ["cfno", "pspnet"]:
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"Inference device: {DEVICE}")

    DATASET_PATH = "data/processed/gds_dataset/"

    TRAIN_DATASET = OPCDataset(
        os.path.join(DATASET_PATH, "origin/train_origin"),
        os.path.join(DATASET_PATH, "correction/train_correction"),
        transform=apply_transform(binarize_flag=True),
    )
    VALID_DATASET = OPCDataset(
        os.path.join(DATASET_PATH, "origin/valid_origin"),
        os.path.join(DATASET_PATH, "correction/valid_correction"),
        transform=apply_transform(binarize_flag=True),
    )
    TEST_DATASET = OPCDataset(
        os.path.join(DATASET_PATH, "origin/test_origin"),
        os.path.join(DATASET_PATH, "correction/test_correction"),
        transform=apply_transform(binarize_flag=True),
    )

    TRAIN_LOADER = DataLoader(
        TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=2
    )
    VALID_LOADER = DataLoader(
        VALID_DATASET, batch_size=batch_size, shuffle=False, num_workers=2
    )
    TEST_LOADER = DataLoader(
        TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = load_model(model_type=model_type, weights_path=weights, device=DEVICE)
    if subset == "train":
        print(f"Running evaluation on {subset} set:")
        evaluate_model(model=model, loader=TRAIN_LOADER, device=DEVICE, log=False)
    elif subset == "valid":
        print(f"Running evaluation on {subset} set:")
        evaluate_model(model=model, loader=VALID_LOADER, device=DEVICE, log=False)
    elif subset == "test":
        print(f"Running evaluation on {subset} set:")
        evaluate_model(model=model, loader=TEST_LOADER, device=DEVICE, log=False)


if __name__ == "__main__":
    fire.Fire(main)
