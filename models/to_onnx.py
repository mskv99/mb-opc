import os
import sys

import fire
import numpy as np
import onnxruntime as ort
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.components.unet import Generator


def check_onnx(onnx_model_path: str):
    inputs = torch.randn(1, 1, 1024, 1024)
    ort_sess = ort.InferenceSession(onnx_model_path)
    output = ort_sess.run(None, {"RAW_DESIGN": inputs.numpy().astype(np.float32)})
    print(f"Output:{output}")
    print(f"Output shape:{output[0].shape}")


def main(
    raw_weights: str = "checkpoints/upernet.ckpt",
    onnx_weights: str = "checkpoints/exported_model.onnx",
    check: bool = True,
):
    model = Generator(in_ch=1, out_ch=1, skip_con_type="concat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(raw_weights, map_location=device)["model_state_dict"]
    )
    model = model.to(device)
    model.eval()
    model.to(device)

    dummy_input = torch.randn(1, 1, 1024, 1024, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_weights,
        opset_version=17,
        do_constant_folding=True,
        input_names=["RAW_DESIGN"],
        output_names=["CORRECTION"],
        dynamic_axes={"RAW_DESIGN": {0: "batch_size"}, "CORRECTION": {0: "batch_size"}},
    )
    print("Модель успешно экспортирована в onnx формат!")

    if check:
        check_onnx(onnx_model_path=onnx_weights)


if __name__ == "__main__":
    fire.Fire(main)
