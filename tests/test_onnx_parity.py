"""
BiasCorrectionNet exported the way src/export_onnx.py does it must give the
same numbers in onnxruntime as in torch (random weights, random input).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from src.models.bias_correction_net import BiasCorrectionNet  # noqa: E402


def test_onnx_export_matches_torch(tmp_path):
    torch.manual_seed(0)
    model = BiasCorrectionNet().eval()
    # non-trivial BatchNorm statistics, as a trained model would have
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)

    dummy = torch.randn(1, 50, 6)
    path = tmp_path / "model.onnx"
    extra = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(model, dummy, str(path), input_names=["imu_window"], output_names=["corrections"],
                      opset_version=17, dynamic_axes=None, **extra)
    onnx.checker.check_model(onnx.load(str(path)))

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.shape for i in sess.get_inputs()] == [[1, 50, 6]]
    assert [o.shape for o in sess.get_outputs()] == [[1, 50, 2]]
    for seed in range(3):
        x = torch.randn(1, 50, 6, generator=torch.Generator().manual_seed(seed)) * 3
        with torch.no_grad():
            expected = model(x).numpy()
        got = sess.run(None, {"imu_window": x.numpy()})[0]
        assert np.abs(expected - got).max() < 1e-4
