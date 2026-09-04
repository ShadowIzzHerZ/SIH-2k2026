"""
Export the trained BiasCorrectionNet to ONNX for on-device inference.

Why export only the network, not the physics integrator: strapdown_ins.py's
integration is trivial arithmetic (a cumulative sum) that the mobile app's
own real-time loop should do sample-by-sample as new IMU data arrives —
that's also how "instant seamless mode-switching" between GNSS and pure-INS
works in practice (you can't wait for a whole window to re-integrate).
So the exported artifact's contract with the mobile app is:

    ONNX model:  (1, window_size, 6) calibrated IMU window -> (1, window_size, 2) [delta_v, delta_theta]
    App-side:    forward_accel + delta_v[-1], yaw_rate + delta_theta[-1]  ->  integrate one step

Dynamic axes are left off the batch dim on purpose — ONNX Runtime Mobile
does one window at a time on-device, so a fixed batch size of 1 keeps the
exported graph simpler and avoids surprises in the mobile runtime.

Run:
    python -m src.export_onnx --config configs/default.yaml --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse

import onnx
import onnxruntime as ort
import torch
import yaml

from src.models.bias_correction_net import BiasCorrectionNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model = BiasCorrectionNet(
        input_channels=cfg["model"]["input_channels"],
        cnn_channels=cfg["model"]["cnn_channels"],
        cnn_kernel_size=cfg["model"]["cnn_kernel_size"],
        cnn_dilations=cfg["model"].get("cnn_dilations"),
        gru_hidden=cfg["model"]["gru_hidden"],
        gru_layers=cfg["model"]["gru_layers"],
        dropout=cfg["model"]["dropout"],
        output_dim=cfg["model"]["output_dim"],
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    window_size = cfg["data"]["window_size"]
    dummy = torch.randn(1, window_size, cfg["model"]["input_channels"])
    out_path = cfg["export"]["onnx_path"]

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["imu_window"],
        output_names=["corrections"],
        opset_version=cfg["export"]["opset"],
        dynamic_axes=None,  # fixed shape, see module docstring
    )

    # sanity-check: structural validity + numerical parity vs. the torch model
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    torch_out = model(dummy).detach().numpy()
    onnx_out = sess.run(None, {"imu_window": dummy.numpy()})[0]
    max_diff = abs(torch_out - onnx_out).max()
    print(f"exported to {out_path}")
    print(f"max |torch - onnx| output diff: {max_diff:.2e} (should be ~1e-5 or smaller)")

    if max_diff > 1e-3:
        raise RuntimeError("ONNX export diverges from the PyTorch model beyond tolerance — do not ship this.")


if __name__ == "__main__":
    main()
