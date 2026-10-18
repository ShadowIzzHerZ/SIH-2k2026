"""Export the best round-2 model (average of three 15 s-context, speed-input
models) to one ONNX file and check it against PyTorch.

    python -m experiments.drift.export_best [name ...]

Inputs:  imu (1, 150, 6)  calibrated vehicle-frame [ax, ay, az, gx, gy, gz],
                          the 10 s before the 5 s chunk plus the chunk, 10 Hz
         v0  (1,)         speed (m/s) the integrator starts the chunk from
Output:  corr (1, 50, 2)  [delta forward accel, delta yaw rate] for the chunk,
                          added to the raw accel/gyro before integration
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from experiments.drift import ideas2, lib

NAMES = sys.argv[1:] or ["hist10_speedaware", "hist10_speedaware_s7", "hist10_speedaware_s3"]


class Exported(torch.nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.ens = ideas2.MeanEnsemble(nets)

    def forward(self, imu, v0):
        self.ens.cur_v0 = v0
        return self.ens(imu)


nets = []
for n in NAMES:
    m = ideas2.HistSpeedNet()
    m.load_state_dict(torch.load(lib.ROOT / "checkpoints" / f"{n}.pt"))
    nets.append(m.eval())
model = Exported(nets).eval()
out = lib.ROOT / "checkpoints" / "hist10_speedaware_ens3.onnx"
imu, v0 = torch.randn(1, 150, 6) * 0.5, torch.tensor([12.0])
torch.onnx.export(model, (imu, v0), out, input_names=["imu", "v0"], output_names=["corr"], opset_version=17,
                  dynamo=False)

import onnxruntime as ort
sess = ort.InferenceSession(str(out))
worst = 0.0
for _ in range(20):
    x, v = torch.randn(1, 150, 6) * 0.5, torch.rand(1) * 30
    ref = model(x, v).detach().numpy()
    got = sess.run(None, {"imu": x.numpy(), "v0": v.numpy()})[0]
    worst = max(worst, float(np.abs(ref - got).max()))
print(f"exported {out.name} ({out.stat().st_size / 1e6:.1f} MB) from {NAMES}; max |onnx - torch| = {worst:.2e}")
