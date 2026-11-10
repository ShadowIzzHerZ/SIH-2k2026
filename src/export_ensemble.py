"""
Export the two-model, speed-gated ensemble the Android app runs.

The drift experiments (experiments/drift/run_duo.py on the drift-experiments
branch) found the best balance with two networks chosen per 5 s chunk by the
speed the chunk starts from:

    speed >= 6 m/s : "fast" model, checkpoints/ensemble_fast.pt
                     (speedaware: the 6 IMU channels plus a 7th channel
                     holding that start speed / 10, the same value on every
                     sample of the window)
    speed <  6 m/s : "slow" model, checkpoints/ensemble_slow.pt
                     (augbias_long: the plain 6-channel network)

Both use the same BiasCorrectionNet architecture as the single model, only
the fast one takes 7 input channels. The ONNX contracts are therefore

    model_slow.onnx: (1, 50, 6) -> (1, 50, 2)
    model_fast.onnx: (1, 50, 7) -> (1, 50, 2)

and the app adds the 7th channel itself (see EnsembleModel.kt), so the
graph has no extra inputs. The 6 m/s gate and the /10 scaling live in the
app and in this docstring; change them only together with a retrain.

Run:
    python -m src.export_ensemble
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.bias_correction_net import BiasCorrectionNet

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 50
ARCH = dict(cnn_channels=[32, 64, 64], cnn_kernel_size=5, gru_hidden=128,
            gru_layers=2, dropout=0.2, output_dim=2)
SPEED_SCALE = 10.0  # 7th channel = start speed (m/s) / SPEED_SCALE


def load(path: Path, channels: int) -> BiasCorrectionNet:
    sd = torch.load(path, map_location="cpu")
    # The speed-aware checkpoint wraps the network as `.net`.
    if any(k.startswith("net.") for k in sd):
        sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
    model = BiasCorrectionNet(input_channels=channels, **ARCH)
    model.load_state_dict(sd)
    return model.eval()


def export(model: BiasCorrectionNet, channels: int, out: Path) -> None:
    dummy = torch.randn(1, WINDOW, channels)
    torch.onnx.export(model, dummy, str(out), input_names=["imu_window"], output_names=["corrections"],
                      opset_version=17, dynamic_axes=None, dynamo=False)
    onnx.checker.check_model(onnx.load(str(out)))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ins, outs = sess.get_inputs(), sess.get_outputs()
    assert [i.shape for i in ins] == [[1, WINDOW, channels]], ins
    assert [o.shape for o in outs] == [[1, WINDOW, 2]], outs
    diff = float(np.abs(model(dummy).detach().numpy() - sess.run(None, {"imu_window": dummy.numpy()})[0]).max())
    print(f"exported {out.relative_to(ROOT)}: (1, {WINDOW}, {channels}) -> (1, {WINDOW}, 2), max |torch - onnx| {diff:.2e}")
    if diff > 1e-3:
        raise RuntimeError(f"{out.name} diverges from the PyTorch model; do not ship it")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slow", type=Path, default=ROOT / "checkpoints/ensemble_slow.pt")
    p.add_argument("--fast", type=Path, default=ROOT / "checkpoints/ensemble_fast.pt")
    p.add_argument("--out_dir", type=Path, default=ROOT / "checkpoints")
    args = p.parse_args()
    export(load(args.slow, 6), 6, args.out_dir / "model_slow.onnx")
    export(load(args.fast, 7), 7, args.out_dir / "model_fast.onnx")


if __name__ == "__main__":
    main()
