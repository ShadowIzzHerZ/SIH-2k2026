"""
Checks for the two-model ensemble's Python side: the exported ONNX files the
app bundles have the shapes the app requires and match the PyTorch
checkpoints, and scripts/publish_model.py rewrites only the role it is
publishing.

    python -m pytest tests/test_ensemble_publish.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from src.export_ensemble import SPEED_SCALE, load

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app/app/src/main/assets"

spec = importlib.util.spec_from_file_location("publish_model", ROOT / "scripts/publish_model.py")
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


@pytest.mark.parametrize("role,channels,ckpt", [("slow", 6, "ensemble_slow.pt"), ("fast", 7, "ensemble_fast.pt")])
def test_bundled_onnx_matches_checkpoint(role, channels, ckpt):
    for path in (ROOT / f"checkpoints/model_{role}.onnx", ASSETS / f"model_{role}.onnx"):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        assert [i.shape for i in sess.get_inputs()] == [[1, 50, channels]]
        assert [o.shape for o in sess.get_outputs()] == [[1, 50, 2]]
        x = np.random.default_rng(1).normal(0, 0.5, (1, 50, channels)).astype(np.float32)
        if role == "fast":
            x[..., 6] = 14.0 / SPEED_SCALE
        model = load(ROOT / "checkpoints" / ckpt, channels)
        with torch.no_grad():
            want = model(torch.from_numpy(x)).numpy()
        got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        assert np.abs(got - want).max() < 1e-4


def test_assets_are_the_exported_checkpoints():
    for role in ("slow", "fast"):
        assert (ASSETS / f"model_{role}.onnx").read_bytes() == (ROOT / f"checkpoints/model_{role}.onnx").read_bytes()


def test_fast_model_reads_the_speed_channel():
    sess = ort.InferenceSession(str(ASSETS / "model_fast.onnx"), providers=["CPUExecutionProvider"])
    x = np.zeros((1, 50, 7), dtype=np.float32)
    slow_out = sess.run(None, {"imu_window": x})[0]
    x[..., 6] = 2.5
    fast_out = sess.run(None, {"imu_window": x})[0]
    assert not np.allclose(slow_out, fast_out)


def test_bundled_versions_are_read_per_role():
    assert publish.bundled_version("slow") >= 1
    assert publish.bundled_version("fast") >= 1
    assert publish.bundled_version("legacy") == publish.LEGACY_BUNDLED_VERSION


def test_publishing_one_role_keeps_the_others():
    legacy = {"version": 3, "file": "dead_reckoning_model_v3.onnx", "sha256": "a" * 64, "size_bytes": 1}
    slow = {"version": 2, "file": "model_slow_v2.onnx", "sha256": "b" * 64, "size_bytes": 2}
    fast = {"version": 4, "file": "model_fast_v4.onnx", "sha256": "c" * 64, "size_bytes": 3}
    current = dict(legacy, models={"slow": slow})

    out = publish.updated_manifest(current, "fast", fast)
    assert out["models"] == {"slow": slow, "fast": fast}
    assert {k: out[k] for k in legacy} == legacy
    assert current["models"] == {"slow": slow}  # input not modified

    new_legacy = dict(legacy, version=4)
    out = publish.updated_manifest(current, "legacy", new_legacy)
    assert out["version"] == 4 and out["models"] == {"slow": slow}

    assert publish.updated_manifest({}, "slow", slow) == {"models": {"slow": slow}}


def test_role_versions():
    m = {"version": 3, "models": {"fast": {"version": 5}}}
    assert publish.role_version(m, "legacy") == 3
    assert publish.role_version(m, "fast") == 5
    assert publish.role_version(m, "slow") == 0
    assert publish.role_version({}, "fast") == 0


def test_check_model_rejects_the_wrong_role():
    publish.check_model(ASSETS / "model_slow.onnx", "slow")
    publish.check_model(ASSETS / "model_fast.onnx", "fast")
    with pytest.raises(SystemExit):
        publish.check_model(ASSETS / "model_fast.onnx", "slow")
    with pytest.raises(SystemExit):
        publish.check_model(ASSETS / "model_slow.onnx", "fast")


def test_app_constants_match_the_export_and_publish_script():
    """The gate, speed scaling and per-role channel counts are written in
    Kotlin, Python and the publish script; they have to agree."""
    import re
    kt = (ROOT / "app/app/src/main/java/com/sih26168/deadreckoning/EnsembleModel.kt").read_text()
    manifest_kt = (ROOT / "app/app/src/main/java/com/sih26168/deadreckoning/ModelManifest.kt").read_text()
    assert float(re.search(r"GATE_MPS\s*=\s*([\d.]+)f", kt).group(1)) == 6.0
    assert float(re.search(r"SPEED_SCALE\s*=\s*([\d.]+)f", kt).group(1)) == SPEED_SCALE
    roles = dict(re.findall(r'\("(\w+)", "model_\w+\.onnx", (\d+)\)', manifest_kt))
    assert {k: int(v) for k, v in roles.items()} == {k: v for k, v in publish.CHANNELS.items() if k != "legacy"}
