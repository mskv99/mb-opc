import matplotlib.pyplot as plt
import os


def get_next_experiment_folder(checkpoints_dir):
    # Ensure the checkpoints directory exists
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Find the next available experiment number
    exp_number = 1
    while True:
        exp_folder = os.path.join(checkpoints_dir, f"exp_{exp_number}")
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
            return exp_folder
        exp_number += 1


def next_exp_folder(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    dir_list = os.listdir(checkpoints_dir)
    give_numb = lambda x: int(x.split("_")[-1])
    dir_numbers = [
        give_numb(name) for name in dir_list if not name.endswith(".gitkeep")
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

"""
for images, targets in dataloader:
  preds = model(images)
  loss_tv = tv_loss(preds)
  loss_boundary = boundary_loss(preds, targets)
  loss_perceptual = perceptual_loss(preds, targets)
  
  # Total loss
  total_loss = loss_tv + loss_boundary + loss_perceptual
"""
