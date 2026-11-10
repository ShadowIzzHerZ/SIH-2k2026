"""
Reference numbers for the app's forward-axis calibration test.

The drift experiments (experiments/drift/lib.py, calibrate_fwdfix, on the
drift-experiments branch) found the phone->vehicle forward axis by fitting
the leveled horizontal accel to what GPS says the car did: longitudinal
dv/dt and centripetal v * yaw_rate. Closed form:

    psi = angle(sum(conj(ax + i*ay) * (dv/dt + i * v * wz)))   over v > 3 m/s

with every signal smoothed by an edge-normalised 10-sample moving average.
The ensemble models were trained on IMU data calibrated this way.

This runs that exact computation (copied below) over the app's bundled
replay recordings and writes psi per file, so CalibrationForwardAxisTest
can check the Kotlin port against it.

    python -m scripts.fwdfix_reference
"""
import json
from pathlib import Path

import numpy as np

from src.calibration import estimate_gravity_vector, leveling_rotation

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app/app/src/main/assets"
OUT = ROOT / "app/app/src/test/resources/fwdfix_reference.json"
DT = 0.1


def _smooth(x, k=10):
    return np.convolve(x, np.ones(k), mode="same") / np.convolve(np.ones_like(x), np.ones(k), mode="same")


def forward_axis(accel, gyro, speed, min_speed=3.0):
    """calibrate_fwdfix's psi, plus the leveling it uses and how many
    samples counted. Returns (psi or None, ok count, R)."""
    R = leveling_rotation(estimate_gravity_vector(accel))
    a = accel @ R.T
    w = gyro @ R.T
    v = speed
    dvdt = np.gradient(_smooth(v), DT)
    lat = _smooth(v) * _smooth(w[:, 2])
    ac = _smooth(a[:, 0]) + 1j * _smooth(a[:, 1])
    tc = dvdt + 1j * lat
    ok = (v > min_speed) & np.isfinite(tc) & np.isfinite(ac)
    psi = float(np.angle(np.sum(np.conj(ac[ok]) * tc[ok]))) if ok.sum() > 100 else None
    return psi, int(ok.sum()), R


def main():
    out = {}
    for path in sorted(ASSETS.glob("replay_*.json")):
        rows = json.loads(path.read_text())["rows"]
        accel = np.array([[r["ax"], r["ay"], r["az"]] for r in rows], dtype=np.float64)
        gyro = np.array([[r["gx"], r["gy"], r["gz"]] for r in rows], dtype=np.float64)
        speed = np.array([r["speed"] for r in rows], dtype=np.float64)
        psi, ok, R = forward_axis(accel, gyro, speed)
        out[path.name] = {"samples": len(rows), "ok_samples": ok, "psi": psi,
                          "max_speed": float(speed.max()), "r_level": R.reshape(-1).tolist()}
        print(f"{path.name}: {len(rows)} samples, {ok} above 3 m/s, psi "
              f"{'n/a' if psi is None else f'{np.degrees(psi):.2f} deg'}")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
