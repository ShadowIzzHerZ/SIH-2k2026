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

    IS now wired in — via IOVNBDWindowDataset's extra_features=True /
    src.train's --extra_features flag — after a proper, conclusive retry
    (2026-09-05) fixed both gaps the first inconclusive attempt had: it now
    gets real per-channel z-score normalization (computed from train-split
    stats only, see compute_channel_norm_stats — these 6 extra channels are
    on very different scales from each other and from the raw 6, unlike
    train.py's reasonably-scaled raw accel/gyro that trained fine into
    BatchNorm1d unnormalized) and a fair, generous early_stop_patience=15
    for the first cycle (configs/engineered_features.yaml) instead of the
    original attempt's default 8, which had cut it off at epoch 10 — nowhere
    near the 57 epochs the 6-channel baseline took to reach its own best.

    Conclusive result this time, combined with comma2k19 mixed into
    training (checkpoints/best_engineered.pt, results/eval_report saved
    under that run): **worse than the 6-channel baseline on the metric that
    matters most** — 62.75% combined / 64.28% IO-VNBD-only test drift, vs.
    60.92%/62.33% without the extra channels. The one genuine win: comma2k19
    -only test drift improved further to 14.48% mean / **5.42% median**,
    71.45% pass rate (vs. 16.49%/8.94%/54.5% with just the raw 6 channels)
    — the extra engineered channels (magnitude, jerk, local roughness) do
    help the model on already-easy, clean highway driving, but make the
    harder IO-VNBD urban/stop-and-go case worse, not better. **Reverted to
    the 6-channel baseline as the production checkpoint** (`checkpoints/
    best.pt`) given this; `checkpoints/best_engineered.pt` and its eval
    reports are kept as a reference, same treatment as best_6ch_baseline.pt.

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
    def __init__(self, windows: list[Window], extra_features: bool = False,
                 norm_mean: np.ndarray | None = None, norm_std: np.ndarray | None = None):
        """extra_features=True appends engineer_features()'s 6 extra channels
        (12 total). norm_mean/norm_std (per-channel, computed once from the
        *train* split only — see compute_channel_norm_stats — and reused
        across val/test/comma2k19_test_only) z-score the raw+engineered
        channels before they hit the network. Both were flagged as missing
        in the first, inconclusive attempt at this (see engineer_features()'s
        docstring) — the 6 raw channels are all similar-scale accel/gyro
        already, so they trained fine unnormalized, but jerk/local-std/
        magnitude channels are on very different scales and need it."""
        self.windows = windows
        self.extra_features = extra_features
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        raw_imu = np.concatenate([w.accel, w.gyro], axis=1).astype(np.float32)  # (T, 6), real physical units
        imu = raw_imu
        if self.extra_features:
            extra = engineer_features(w.accel, w.gyro, w.dt)  # (T, 6)
            imu = np.concatenate([imu, extra], axis=1)  # (T, 12)
        if self.norm_mean is not None:
            imu = (imu - self.norm_mean) / self.norm_std
        # "imu" (possibly extended + z-score normalized) feeds the network;
        # "imu_raw" (always real accel m/s^2 / gyro rad/s) is what
        # train.py's forward_pass integrates through the physics baseline —
        # normalizing that too would integrate z-scored numbers instead of
        # real acceleration, breaking dead-reckoning entirely, not just
        # feature scale. When extra_features/norm are both off, imu_raw and
        # imu are identical (same values as before this was added).
        return {
            "imu": torch.from_numpy(imu),
            "imu_raw": torch.from_numpy(raw_imu),
            "speed_gt": torch.from_numpy(w.speed_gt.astype(np.float32)),
            "pos_gt": torch.from_numpy(w.pos_gt.astype(np.float32)),
            "v0": torch.tensor(w.v0, dtype=torch.float32),
            "theta0": torch.tensor(w.theta0, dtype=torch.float32),
            "lat0": w.lat0,
            "lon0": w.lon0,
            "dt": w.dt,
        }


def compute_channel_norm_stats(windows: list[Window], extra_features: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over every timestep of every window, meant to be
    called on the *train* split only and reused (not recomputed) for
    val/test/etc — computing it per-split would leak each split's own
    distribution into its own normalization and isn't how the network would
    see data at real inference time anyway (no access to future/other
    windows' statistics)."""
    all_imu = []
    for w in windows:
        imu = np.concatenate([w.accel, w.gyro], axis=1).astype(np.float32)
        if extra_features:
            imu = np.concatenate([imu, engineer_features(w.accel, w.gyro, w.dt)], axis=1)
        all_imu.append(imu)
    stacked = np.concatenate(all_imu, axis=0)  # (n_windows * T, C)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0  # avoid divide-by-zero on a degenerate channel
    return mean.astype(np.float32), std.astype(np.float32)


def load_dataset_splits(data_root: str, variant: str, column_map: dict, sample_rate_hz: float,
                         window_size: int, window_stride: int,
                         train_split: float, val_split: float, seed: int = 0,
                         file_prefix: str = "", extra_features: bool = False):
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
    split_windows: dict[str, list[Window]] = {}
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
        split_windows[split] = windows
        print(f"{split}: {len(files)} files -> {len(windows)} windows")

    # Normalization stats computed from train only (see
    # compute_channel_norm_stats's docstring) — only when extra_features is
    # on, so the existing plain-6-channel path's numbers stay bit-for-bit
    # unchanged from before this was added.
    norm_mean, norm_std = (None, None)
    if extra_features:
        norm_mean, norm_std = compute_channel_norm_stats(split_windows["train"], extra_features=True)

    out = {}
    for split, windows in split_windows.items():
        out[split] = IOVNBDWindowDataset(windows, extra_features=extra_features,
                                          norm_mean=norm_mean, norm_std=norm_std)
    return out


def load_combined_dataset_splits(comma2k19_dir: str | None = None, seed: int = 0, **iovnbd_kwargs):
    """load_dataset_splits(**iovnbd_kwargs), optionally with comma2k19
    windows mixed into the same train/val/test splits — split at the
    *segment* level (same leakage-avoidance reasoning as IO-VNBD's
    file-level split above), so no comma2k19 segment's windows cross a
    split boundary either.

    comma2k19_dir=None (or no parquet files found there) behaves exactly
    like load_dataset_splits — this is an additive, opt-in extension, not
    a replacement.

    Returns the combined splits dict, plus "iovnbd_test_only" and
    "comma2k19_test_only" — the same two test sets kept separate — so a
    caller can check whether mixing comma2k19 into training measurably
    helped/hurt *IO-VNBD* test drift specifically, not just report a
    combined number that could hide either direction.
    """
    combined = load_dataset_splits(seed=seed, **iovnbd_kwargs)
    if not comma2k19_dir:
        return combined

    comma_paths = sorted(glob.glob(f"{comma2k19_dir}/*.parquet"))
    if not comma_paths:
        print(f"[combined dataset] no comma2k19 parquet files under {comma2k19_dir} — skipping, IO-VNBD only")
        return combined

    from .comma2k19_loader import load_all_segments  # local import: extra deps (pyarrow, huggingface_hub)

    sample_rate_hz = iovnbd_kwargs["sample_rate_hz"]
    window_size = iovnbd_kwargs["window_size"]
    window_stride = iovnbd_kwargs["window_stride"]
    train_split = iovnbd_kwargs["train_split"]
    val_split = iovnbd_kwargs["val_split"]

    seqs = load_all_segments(comma_paths, target_hz=sample_rate_hz)
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(seqs))
    rng.shuffle(idxs)
    n = len(idxs)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    split_idxs = {"train": idxs[:n_train], "val": idxs[n_train:n_train + n_val], "test": idxs[n_train + n_val:]}

    dt = 1.0 / sample_rate_hz
    comma_windows_by_split: dict[str, list[Window]] = {}
    for split, ii in split_idxs.items():
        windows: list[Window] = []
        for i in ii:
            seq = seqs[i]
            try:
                calibrate_sequence(seq)
                windows.extend(build_windows(seq, window_size, window_stride, dt))
            except Exception as e:
                print(f"[skip] comma2k19 segment {seq.path}: {e}")
        comma_windows_by_split[split] = windows
        print(f"comma2k19 {split}: {len(ii)} segments -> {len(windows)} windows")

    # Reuse whatever extra_features/norm stats load_dataset_splits already
    # set up on the IO-VNBD side (computed from IO-VNBD *train* windows only)
    # so comma2k19 windows get the identical treatment — a second,
    # independently-fit normalization would make the two datasets' channels
    # not directly comparable to the network.
    extra_features = combined["train"].extra_features
    norm_mean, norm_std = combined["train"].norm_mean, combined["train"].norm_std

    out = {}
    for split in ("train", "val", "test"):
        out[split] = IOVNBDWindowDataset(list(combined[split].windows) + comma_windows_by_split[split],
                                          extra_features=extra_features, norm_mean=norm_mean, norm_std=norm_std)
    out["iovnbd_test_only"] = combined["test"]
    out["comma2k19_test_only"] = IOVNBDWindowDataset(comma_windows_by_split["test"],
                                                      extra_features=extra_features, norm_mean=norm_mean, norm_std=norm_std)
    return out
