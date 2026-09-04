"""
Turns raw IO-VNBD sequences into fixed-length training windows.

Split is done at the *sequence* (trip/file) level, not the window level —
windows from the same drive are highly correlated, so splitting after
windowing would leak train data into val/test and give an optimistic drift
number that won't hold up on demo day. That would be exactly the kind of
mistake worth catching before judges do.

Each window carries everything the training loop needs:
  - calibrated accel/gyro samples (network input)
  - forward-accel / yaw-rate the physics integrator would use on its own
    (network input's physical counterpart, so the residual correction has
    something to be a residual *of*)
  - GT speed at each step (supervises the per-step loss)
  - GT position relative to the window's start (supervises the drift loss)
  - v0 / theta0: the vehicle's true starting speed and heading for this
    window. A window is a random 5s slice of a drive — the vehicle is
    essentially never stopped and facing due-east right at the slice
    boundary, so the integrator MUST start from the real initial state,
    not from zero. (Confirmed the hard way: training without this
    plateaued around 85% drift no matter how long it ran — the model was
    being asked to predict trajectories from a starting condition that
    was wrong for nearly every window.)
"""
from __future__ import annotations

import glob
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .io_vnbd_loader import (
    ImuSequence,
    compass_deg_to_xy_unit,
    derive_speed_from_gps,
    latlon_to_local_xy,
    load_sequence,
)
from ..calibration import calibrate, estimate_gravity_vector, estimate_yaw_misalignment, leveling_rotation


@dataclass
class Window:
    accel: np.ndarray       # (T, 3) calibrated, vehicle frame
    gyro: np.ndarray        # (T, 3) calibrated, vehicle frame
    speed_gt: np.ndarray    # (T,)
    pos_gt: np.ndarray      # (T, 2) relative to window start
    v0: float                # true starting speed, m/s
    theta0: float            # true starting heading, rad (world frame, matches pos_gt's axes)
    lat0: float               # true starting lat/lon — lets a predicted local-xy trajectory be
    lon0: float               # converted back to real coordinates for map-matching against OSM
    dt: float


def resample_uniform(seq: ImuSequence, target_hz: float) -> ImuSequence:
    """Resample an (possibly irregular-rate) sequence to a fixed dt via
    linear interpolation. IO-VNBD's phone stream is documented at 10Hz;
    the vehicle stream may differ, so this makes both trainable with the
    same window logic."""
    t0, t1 = seq.time[0], seq.time[-1]
    n = int((t1 - t0) * target_hz)
    if n < 2:
        return seq
    t_new = np.linspace(t0, t1, n)

    def interp(arr):
        if arr is None:
            return None
        if arr.ndim == 1:
            return np.interp(t_new, seq.time, arr)
        return np.stack([np.interp(t_new, seq.time, arr[:, i]) for i in range(arr.shape[1])], axis=1)

    def interp_heading_deg(deg):
        # Plain np.interp on raw degrees breaks at the 0/360 wraparound
        # (e.g. 358° -> 3° would linearly blend through 180°, the opposite
        # direction). Interpolate the compass bearing as a unit vector
        # instead — no wraparound to break — then convert back to degrees.
        if deg is None:
            return None
        rad = np.radians(deg)
        c = np.interp(t_new, seq.time, np.cos(rad))
        s = np.interp(t_new, seq.time, np.sin(rad))
        return np.degrees(np.arctan2(s, c)) % 360

    return ImuSequence(
        path=seq.path,
        time=t_new,
        accel=interp(seq.accel),
        gyro=interp(seq.gyro),
        speed_gt=interp(seq.speed_gt),
        lat=interp(seq.lat),
        lon=interp(seq.lon),
        heading_gt=interp_heading_deg(seq.heading_gt),
    )


def calibrate_sequence(seq: ImuSequence, stationary_samples: int = 20,
                        min_confident_samples: int = 3000, min_speed_mps: float = 2.0) -> ImuSequence:
    """Leveling (roll/pitch) + yaw calibration.

    First attempt at yaw calibration used the GPS track's own raw
    frame-to-frame position bearing as ground truth, and that made things
    worse — a single 0.1s GPS position delta at 10Hz is the same order of
    magnitude as consumer GPS position noise, so the estimates came back
    wildly inconsistent (-123° to +99° across similar files). This version
    uses the phone's own GPS-orientation field instead (course-over-ground,
    computed by the GPS chip itself, not our own differencing — confirmed
    far smoother: holds steady ~90% of samples between real fixes, updates
    sensibly when it does change) and filters "is it actually moving" by
    the GPS speed field directly rather than a noisy position-diff — both
    changes replace a self-inflicted noise source with the dataset's own,
    presumably better-filtered signal.
    """
    n = min(stationary_samples, len(seq.accel))
    gravity = estimate_gravity_vector(seq.accel[:n])
    R_level = leveling_rotation(gravity)

    psi_yaw = 0.0
    if seq.heading_gt is not None and seq.speed_gt is not None and len(seq.heading_gt) > min_confident_samples:
        confident = seq.speed_gt > min_speed_mps
        if confident.sum() >= min_confident_samples:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)  # see note below
                level_accel = seq.accel @ R_level.T
            heading_xy = compass_deg_to_xy_unit(seq.heading_gt)
            psi_yaw = estimate_yaw_misalignment(level_accel[confident, :2], heading_xy[confident])

    # macOS's Accelerate BLAS backend emits spurious "divide by zero" /
    # "invalid value" RuntimeWarnings on these small (Nx3 @ 3x3) matmuls —
    # a known false-positive (verified: output stays fully finite; the FPE
    # flag it's reacting to comes from Accelerate's internal computation,
    # not the actual result). Doesn't happen on Linux/OpenBLAS (e.g. Colab).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        accel_v, gyro_v = calibrate(seq.accel, seq.gyro, R_level, psi_yaw)
    assert np.isfinite(accel_v).all() and np.isfinite(gyro_v).all(), (
        f"{seq.path}: calibration produced non-finite values for real — "
        f"this one isn't the benign Accelerate warning, don't ignore it."
    )
    seq.accel[:] = accel_v
    seq.gyro[:] = gyro_v
    return seq


def build_windows(seq: ImuSequence, window_size: int, stride: int, dt: float,
                   min_distance_m: float = 2.0) -> list[Window]:
    """min_distance_m excludes windows the vehicle was essentially parked/idle
    for. Real driving logs have a lot of this (confirmed: ~47% of windows in
    this dataset travel under 10cm in a 5s window — engine idling, traffic
    stops, before/after the actual drive). This isn't a minor edge case to
    smooth over: dividing a position error by near-zero distance travelled
    (drift_metric's whole definition, matching the PS's own "% of distance
    travelled" wording) blows up to enormous, meaningless percentages for
    those windows — confirmed empirically: one batch's *mean* train drift
    came out at 134,735% before this filter, entirely from a handful of such
    windows dominating the average, not from the model being that wrong.
    It also isn't the scenario the PS benchmarks anyway — blackout tracking
    while the vehicle is moving, not parked. 2m over a 5s window is a clean
    cut confirmed against the real distribution (bimodal: windows are either
    ~0m or already tens of metres, nothing in between)."""
    n = seq.n
    if seq.lat is not None and seq.lon is not None:
        xy = latlon_to_local_xy(seq.lat, seq.lon)
    else:
        xy = None

    if seq.speed_gt is not None:
        speed_gt = seq.speed_gt
    elif xy is not None:
        speed_gt = derive_speed_from_gps(seq.lat, seq.lon, seq.time)
    else:
        speed_gt = None

    windows = []
    for start in range(0, n - window_size, stride):
        end = start + window_size
        w_accel = seq.accel[start:end]
        w_gyro = seq.gyro[start:end]
        w_speed = speed_gt[start:end] if speed_gt is not None else None
        w_pos = (xy[start:end] - xy[start]) if xy is not None else None

        if w_speed is None or w_pos is None:
            continue  # can't supervise this window without GT — skip

        step_dist = np.linalg.norm(np.diff(w_pos, axis=0), axis=1)
        if step_dist.sum() < min_distance_m:
            continue  # essentially stationary — see docstring

        v0 = float(w_speed[0])
        # true starting heading: prefer the GPS-orientation field (GPS
        # chip's own course-over-ground, confirmed smooth — see
        # calibrate_sequence) at the window's start sample when the
        # vehicle's actually moving there. Falls back to bearing from the
        # position track's own first ~1s (short lookahead approximates
        # instantaneous direction of travel; the full window's endpoint
        # would blend in whatever turning happens later in the window,
        # which isn't "starting" heading) if heading_gt isn't available or
        # the window happens to start from a near-stop.
        if seq.heading_gt is not None and w_speed[0] > 1.0:
            unit = compass_deg_to_xy_unit(seq.heading_gt[start])
            theta0 = float(np.arctan2(unit[1], unit[0]))
        else:
            lookahead = min(10, window_size - 1)
            d = w_pos[lookahead] - w_pos[0]
            if np.linalg.norm(d) < 0.5:  # too little motion in that short a span — widen it
                d = w_pos[-1] - w_pos[0]
            theta0 = float(np.arctan2(d[1], d[0]))

        lat0 = float(seq.lat[start]) if seq.lat is not None else float("nan")
        lon0 = float(seq.lon[start]) if seq.lon is not None else float("nan")

        windows.append(Window(accel=w_accel, gyro=w_gyro, speed_gt=w_speed, pos_gt=w_pos,
                               v0=v0, theta0=theta0, lat0=lat0, lon0=lon0, dt=dt))
    return windows


def engineer_features(accel: np.ndarray, gyro: np.ndarray, dt: float, smooth_win: int = 5) -> np.ndarray:
    """Extra per-timestep channels appended after the raw 6 calibrated
    accel/gyro axes.

    NOT currently wired into IOVNBDWindowDataset.__getitem__ — kept here,
    defined but unused, rather than deleted. Tried once (model.input_channels
    12 in config, a matching checkpoints/best_12ch* set): the first
    cold-start cycle reached 71.73% val drift at epoch 2 then early-stopped
    at epoch 10, noisier and no better than the 6-channel baseline at a
    comparable point — but on only 10 epochs (patience=8) vs. the 57 the
    6-channel run took to reach its eventual best, so this wasn't a fair
    trial, not a disproof. Revisit with a higher patience for the first
    cycle, and/or per-channel normalization (these features are on very
    different scales — magnitudes vs. jerk vs. local std — unlike train.py's
    reasonably-scaled raw 6 that trained fine into BatchNorm1d as-is)
    before concluding either way. Reverted to the working 6-channel baseline
    (checkpoints/best.pt) in the meantime given the SIH deadline.

    Four training cycles of warm-restarting the 6-raw-channel model against
    the same data all plateaued in the exact same ~68-70% val-drift band
    (see results/train_history_6ch_baseline.json), and train drift
    plateaued right along with it — evidence of underfitting from limited
    input signal (motivating this), not of overfitting or a bad LR.

    Appended *after* the 6 raw channels (never inserted before) so
    `train.py`'s `imu[..., 0]` (forward accel) and `imu[..., 5]` (yaw rate)
    indexing into the physics baseline stays correct, if re-enabled.
    """
    ax, ay, az = accel[:, 0], accel[:, 1], accel[:, 2]
    gx, gy, gz = gyro[:, 0], gyro[:, 1], gyro[:, 2]

    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    accel_horiz_mag = np.sqrt(ax**2 + ay**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # Sharp brake/throttle transitions look different from steady
    # acceleration only in their rate of change — a single raw sample can't
    # tell them apart.
    jerk_x = np.gradient(ax, dt)

    # Local smoothing/roughness of the two channels the physics baseline
    # actually integrates: a short moving average (separates sustained
    # motion from a single noisy sample) and local std (vibration
    # intensity/roughness) over a ~0.5s window.
    kernel = np.ones(smooth_win) / smooth_win
    ax_smooth = np.convolve(ax, kernel, mode="same")
    ax_local_std = np.sqrt(np.clip(np.convolve(ax**2, kernel, mode="same") - ax_smooth**2, 0, None))
    gz_smooth = np.convolve(gz, kernel, mode="same")
    gz_local_std = np.sqrt(np.clip(np.convolve(gz**2, kernel, mode="same") - gz_smooth**2, 0, None))

    return np.stack(
        [accel_mag, accel_horiz_mag, gyro_mag, jerk_x, ax_local_std, gz_local_std], axis=1
    ).astype(np.float32)


class IOVNBDWindowDataset(Dataset):
    def __init__(self, windows: list[Window]):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        imu = np.concatenate([w.accel, w.gyro], axis=1).astype(np.float32)  # (T, 6)
        return {
            "imu": torch.from_numpy(imu),
            "speed_gt": torch.from_numpy(w.speed_gt.astype(np.float32)),
            "pos_gt": torch.from_numpy(w.pos_gt.astype(np.float32)),
            "v0": torch.tensor(w.v0, dtype=torch.float32),
            "theta0": torch.tensor(w.theta0, dtype=torch.float32),
            "lat0": w.lat0,
            "lon0": w.lon0,
            "dt": w.dt,
        }


def load_dataset_splits(data_root: str, variant: str, column_map: dict, sample_rate_hz: float,
                         window_size: int, window_stride: int,
                         train_split: float, val_split: float, seed: int = 0,
                         file_prefix: str = ""):
    """Discover CSV files, load+calibrate+resample each sequence, window
    them, then split at the *file* level into train/val/test.

    file_prefix filters by filename prefix — set to "S-" (see config) to
    train only on smartphone recordings. IO-VNBD mixes vehicle CAN-bus
    files (`V-*.csv`, no 3-axis IMU) into the same folders, and those will
    never resolve accel/gyro columns no matter what — filtering them out
    up front avoids a wall of noisy [skip] lines for files that were never
    going to work, instead of relying on the per-file error to catch it.
    """
    root = Path(data_root) / variant
    csv_paths = sorted(glob.glob(str(root / "**" / f"{file_prefix}*.csv"), recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found under {root} matching prefix '{file_prefix}' — check data_root/variant/file_prefix in config.")

    rng = np.random.default_rng(seed)
    paths = np.array(csv_paths)
    rng.shuffle(paths)
    n = len(paths)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    split_paths = {
        "train": paths[:n_train],
        "val": paths[n_train:n_train + n_val],
        "test": paths[n_train + n_val:],
    }

    dt = 1.0 / sample_rate_hz
    out = {}
    for split, files in split_paths.items():
        windows: list[Window] = []
        for p in files:
            try:
                seq = load_sequence(Path(p), column_map)
                seq = resample_uniform(seq, sample_rate_hz)
                seq = calibrate_sequence(seq)
                windows.extend(build_windows(seq, window_size, window_stride, dt))
            except Exception as e:
                print(f"[skip] {p}: {e}")
        out[split] = IOVNBDWindowDataset(windows)
        print(f"{split}: {len(files)} files -> {len(windows)} windows")
    return out
