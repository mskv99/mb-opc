import os
import sys
import time

import fire
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.lit_generator import LitGenerator
from src.dataset import TestDataset, apply_transform
from src.utils import next_exp_folder, save_image, set_random_seed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def infer(
    weights: str,
    inference_folder: str = "data/processed/gds_dataset/origin/test_origin",
    model_type: str = "upernet",
    batch_size: int = 2,
    output_folder: str = "inference/output_img",
    num_workers: int = 4,
):
    set_random_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and model_type not in [
        "cfno",
        "pspnet",
        "upernet",
    ]:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Inference device: {device}")

    output_dir = next_exp_folder(output_folder)

    transform = apply_transform(binarize_flag=True)

    dataset = TestDataset(inference_folder, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = LitGenerator.load_from_checkpoint(checkpoint_path=weights)
    model = model.to(device)
    model.eval()

    times = []
    for batch, paths in loader:
        batch = batch.to(device)
        start = time.time()
        with torch.no_grad():
            output = model(batch)
            output = torch.sigmoid(output)
        save_image(output, checkpoint_dir=output_dir, image_type=paths)
        elapsed = time.time() - start
        print(f"Inference batch time: {elapsed:.4f} s")
        times.append(elapsed)

    times = np.array(times)
    print(f"Avg. batch time: {times.mean():.4f} s")
    print(f"Avg. image time: {times.mean() / batch_size:.4f} s")


if __name__ == "__main__":
    fire.Fire(infer)
