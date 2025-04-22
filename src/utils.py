import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import cv2
import os

from models.components.unet import Generator
from models.components.cfno import CFNONet

import segmentation_models_pytorch as smp


def next_exp_folder(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    dir_list = os.listdir(checkpoints_dir)
    give_numb = lambda x: int(x.split("_")[-1])
    dir_numbers = [
        give_numb(name)
        for name in dir_list
        if not (name.endswith(".gitkeep") or (name.endswith(("Store"))))
    ]
    max_number = max(dir_numbers)
    new_exp_folder = os.path.join(checkpoints_dir, f"exp_{max_number + 1}")
    os.makedirs(new_exp_folder)
    return new_exp_folder


def draw_plot(**kwargs):
    # plotting single variable on a plot

    if len(kwargs) == 7:
        plt.figure(figsize=(8, 6))
        plt.plot(kwargs["first_variable"], linestyle="-", label=kwargs["label"])
        plt.title(kwargs["title"])
        plt.xlabel(kwargs["xlabel"])
        plt.ylabel(kwargs["ylabel"])
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(kwargs["checkpoint_dir"], kwargs["save_name"]))
        plt.close()

    # plotting two variables on a plot
    elif len(kwargs) == 9:
        plt.figure(figsize=(8, 6))
        plt.plot(
            kwargs["first_variable"],
            linestyle="-",
            color="r",
            label=kwargs["first_label"],
        )
        plt.plot(
            kwargs["second_variable"],
            linestyle="-",
            color="b",
            label=kwargs["second_label"],
        )
        plt.title(kwargs["title"])
        plt.xlabel(kwargs["xlabel"])
        plt.ylabel(kwargs["ylabel"])
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(kwargs["checkpoint_dir"], kwargs["save_name"]))
        plt.close()


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_model(model_type: str, weights_path: str, device):
    if model_type == "unet":
        model = Generator(in_ch=1, out_ch=1, skip_con_type="concat")
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    elif model_type == "cfno":
        model = CFNONet()
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    elif model_type in ["manet", "pspnet", "deeplabv3", "unetplusplus"]:
        model = smp.from_pretrained(weights_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.to(device)
    model.eval()
    print(f"Model '{model_type}' loaded.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    return model


def save_image(output_batch, checkpoint_dir="checkpoints", image_type="output"):
    for i, output in enumerate(output_batch):
        image = output.squeeze().cpu().numpy()
        image = (image > 0.5).astype(np.uint8) * 255
        path = os.path.join(checkpoint_dir, f"{os.path.basename(image_type[i])}.jpg")
        cv2.imwrite(path, image)
        print(f"Saved to {path}")


if __name__ == "__main__":
    a = [1, 2, 3, 4, 5, 20, 30, 50, 90, 100]
    b = [0, 2, 3, 4, 10, 11, 12, 80, 110, 120]

    draw_plot(
        first_variable=a,
        label="loss",
        title="Loss plot",
        xlabel="loss value",
        ylabel="iteration",
        save_name="test_graph.jpg",
        checkpoint_dir="data/external",
    )
    draw_plot(
        first_variable=a,
        second_variable=b,
        title="Loss plot",
        xlabel="loss value",
        ylabel="iteration",
        first_label="iou_train",
        second_label="iou_val",
        save_name="iou_graph.jpg",
        checkpoint_dir="data/external",
    )
