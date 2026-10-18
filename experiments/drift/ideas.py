"""The drift-reduction ideas, one entry each. Every entry is a small change on
top of the v3 recipe (lib.train) so it can be compared row by row."""
from __future__ import annotations

import math
import numpy as np
import torch

from experiments.drift import lib
from src.data.io_vnbd_loader import compass_deg_to_xy_unit
from src.models.strapdown_ins import dead_reckon_position, integrate_heading


# ------------------------------------------------ idea: bias from pre-blackout GPS
def make_bias_fn(splits, lookback_s=30.0, min_s=10.0, min_speed=3.0, gyro=True, accel=True):
    """In the app a blackout always follows a stretch WITH GPS. Over the last
    `lookback_s` before the blackout: forward-accel bias = mean(a_fwd) -
    (speed change / time); yaw-rate bias = mean(gz) - (GPS course change /
    time), the latter only when moving fast enough for the GPS course to be
    trusted. Uses only data from before the chain starts."""
    seqs = splits["_seqs"]
    cache = {}

    def fn(w):
        key = (w.src, w.start)
        if key in cache:
            return cache[key]
        s = seqs[w.src]
        L = int(lookback_s / lib.DT)
        a0 = max(0, w.start - L)
        b = np.zeros(6, dtype=np.float32)
        if w.start - a0 >= int(min_s / lib.DT):
            T = (w.start - a0) * lib.DT
            v = s.speed_gt[a0:w.start]
            if accel:
                b[0] = s.accel[a0:w.start, 0].mean() - (v[-1] - v[0]) / T
            if gyro and s.heading_gt is not None and v.mean() > min_speed:
                u = compass_deg_to_xy_unit(s.heading_gt[a0:w.start])
                h = np.unwrap(np.arctan2(u[:, 1], u[:, 0]))
                b[5] = s.gyro[a0:w.start, 2].mean() - (h[-1] - h[0]) / T
        cache[key] = b
        return b
    return fn


def bias_chain_fn(splits, **kw):
    bf = make_bias_fn(splits, **kw)

    def chain_fn(model, ds, device, dt, **k):
        return lib.chained_drift_x(model, ds, device, dt, bias_fn=bf, **k)
    return chain_fn


# ------------------------------------------------ idea: zero-velocity (stop) detection
def zupt_step_fn(acc_std_thr=0.06, gz_thr=0.005, k=10):
    """Speed forced to 0 where the last 1 s of IMU looks parked (low accel-norm
    std and low yaw rate). Thresholds picked from TRAIN-split stationary vs
    moving distributions (comma2k19 separates cleanly, IO-VNBD barely)."""
    def step(raw, c, v, th, imu):
        acc = raw[..., 0] + c[..., 0]
        an = torch.linalg.norm(raw[..., :3], dim=-1)
        # rolling std / mean over the trailing k samples inside the chunk
        pad = torch.cat([an[:, :1].expand(-1, k - 1), an], 1)
        win = pad.unfold(1, k, 1)
        astd = win.std(-1)
        gz = torch.cat([raw[:, :1, 5].abs().expand(-1, k - 1), raw[..., 5].abs()], 1).unfold(1, k, 1).mean(-1)
        still = (astd < acc_std_thr) & (gz < gz_thr)
        still[:, : k - 1] = False  # padded start of the chunk: not enough samples to judge
        vs, vv = [], v
        for i in range(acc.shape[1]):
            vv = (vv + acc[:, i] * lib.DT).clamp(0.0, 50.0)
            vv = torch.where(still[:, i], torch.zeros_like(vv), vv)
            vs.append(vv)
        speed = torch.stack(vs, 1)
        heading = th.unsqueeze(1) + torch.cumsum((raw[..., 5] + c[..., 1]) * lib.DT, 1)
        return speed, heading
    return step


def zupt_chain_fn(**kw):
    sf = zupt_step_fn(**kw)

    def chain_fn(model, ds, device, dt, **k):
        return lib.chained_drift_x(model, ds, device, dt, step_fn=sf, **k)
    return chain_fn


# ------------------------------------------------ idea: learned absolute-speed anchor
class SpeedAnchorNet(torch.nn.Module):
    """Base net with a 3rd output: an absolute speed estimate v_hat (softplus).
    Speed is a complementary filter: v_k = (1-a)(v_{k-1} + acc_k dt) + a v_hat_k,
    a = sigmoid(learned scalar), so integration drift is pulled back toward a
    speed the network reads from the IMU (vibration, turning dynamics) instead
    of growing without bound over a long blackout."""

    def __init__(self):
        super().__init__()
        self.net = lib.new_model(output_dim=3)
        self.alpha_logit = torch.nn.Parameter(torch.tensor(-3.0))

    def forward(self, x):
        return self.net(x)

    def integrate(self, raw, c, v0):
        a = torch.sigmoid(self.alpha_logit)
        acc = raw[..., 0] + c[..., 0]
        vhat = torch.nn.functional.softplus(c[..., 2]) * 10.0
        vs, vv = [], v0
        for i in range(acc.shape[1]):
            vv = ((1 - a) * (vv + acc[:, i] * lib.DT) + a * vhat[:, i]).clamp(min=0.0)
            vs.append(vv)
        return torch.stack(vs, 1), vhat


def anchor_forward(model, batch, dt, device):
    raw = batch["imu_raw"]
    c = model(batch["imu"])
    speed, _ = model.integrate(raw, c, batch["v0"])
    heading = integrate_heading(raw[..., 5] + c[..., 1], dt, theta0=batch["theta0"])
    return speed, dead_reckon_position(speed, heading, dt), batch["speed_gt"], batch["pos_gt"]


def anchor_loss(model, splits):
    from src.train import compute_loss

    def fn(b):
        raw = b["imu_raw"]
        c = model(b["imu"])
        speed, vhat = model.integrate(raw, c, b["v0"])
        heading = integrate_heading(raw[..., 5] + c[..., 1], lib.DT, theta0=b["theta0"])
        pos = dead_reckon_position(speed, heading, lib.DT)
        loss, m = compute_loss(speed, pos, b["speed_gt"], b["pos_gt"], lib.LOSS_WEIGHTS)
        return loss + 0.1 * torch.nn.functional.smooth_l1_loss(vhat, b["speed_gt"]), m
    return fn


def anchor_chain_loss(model, splits):
    from src.train import compute_loss

    def fn(b):
        imu, raw, sgt, pgt = b["imu"], b["imu_raw"], b["speed_gt"], b["pos_gt"]
        v, th = b["v0"], b["theta0"]
        B, K = imu.shape[:2]
        p_cum = torch.zeros(B, 2)
        gt_cum = torch.zeros(B, 2)
        dist = torch.zeros(B)
        total = 0.0
        for j in range(K):
            c = model(imu[:, j])
            speed, vhat = model.integrate(raw[:, j], c, v)
            heading = integrate_heading(raw[:, j, :, 5] + c[..., 1], lib.DT, theta0=th)
            pos = dead_reckon_position(speed, heading, lib.DT)
            lj, _ = compute_loss(speed, pos, sgt[:, j], pgt[:, j], w)
            p_cum = p_cum + pos[:, -1]
            gt_cum = gt_cum + pgt[:, j, -1]
            dist = dist + torch.linalg.norm(pgt[:, j, 1:] - pgt[:, j, :-1], dim=-1).sum(1)
            cd = 100.0 * torch.linalg.norm(p_cum - gt_cum, dim=-1) / dist.clamp(min=1e-6)
            total = total + lj + cd.mean() + 0.1 * torch.nn.functional.smooth_l1_loss(vhat, sgt[:, j])
            v, th = speed[:, -1], heading[:, -1]
        return total / K, {}
    return fn


def anchor_step(model):
    def step(raw, c, v, th, imu):
        speed, _ = model.integrate(raw, c, v)
        heading = th.unsqueeze(1) + torch.cumsum((raw[..., 5] + c[..., 1]) * lib.DT, 1)
        return speed, heading
    return step


# ------------------------------------------------ idea: bias/noise augmentation
class AugDataset(torch.utils.data.Dataset):
    """Adds a random constant bias per window (accel ±0.15 m/s^2, gyro ±0.01
    rad/s per axis) and small white noise to BOTH the physics input and the
    network input, so the network learns to estimate and cancel bias from what
    it sees instead of memorising each training drive's bias."""

    def __init__(self, base, acc_b=0.15, gyro_b=0.01, noise=0.02):
        self.base, self.acc_b, self.gyro_b, self.noise = base, acc_b, gyro_b, noise
        self.windows = base.windows

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = dict(self.base[i])
        b = torch.cat([torch.empty(3).uniform_(-self.acc_b, self.acc_b),
                       torch.empty(3).uniform_(-self.gyro_b, self.gyro_b)])
        raw = it["imu_raw"] + b + self.noise * torch.randn_like(it["imu_raw"])
        it["imu_raw"] = raw
        it["imu"] = raw
        return it


# ------------------------------------------------ idea: start from physics
def zero_init_model():
    """Last layer zeroed so training starts from pure physics (zero correction)
    instead of random corrections; the fixed-calibration physics alone is
    already ~18% at 30 s, better than early-epoch models."""
    m = lib.new_model()
    torch.nn.init.zeros_(m.head[-1].weight)
    torch.nn.init.zeros_(m.head[-1].bias)
    return m


# ------------------------------------------------ registry
def _calib_splits():
    return lib.build_splits("fwdfix")


EXPERIMENTS = {
    "baseline": {
        "desc": "v3 recipe (heading + end-heading losses, chained fine-tune K=3), original calibration, clean split",
        "calib": "orig",
    },
    "calibfix": {
        "desc": "baseline + fixed phone->vehicle calibration (whole-drive leveling, forward axis fitted to GPS accel)",
        "calib": "fwdfix",
    },
    "zeroinit": {
        "desc": "calibfix + correction head initialised to zero (training starts from physics)",
        "calib": "fwdfix",
        "model": zero_init_model,
    },
    "aug_chain6": {
        "desc": "calibfix + bias augmentation + 30 s (6-chunk) chained fine-tune",
        "calib": "fwdfix",
        "train_ds": lambda model, sp: AugDataset(sp["train"]),
        "train_kw": {"chunks": 6, "max_chains": 6000},
    },
    "chain6": {
        "desc": "calibfix + chained fine-tune over 6 chunks (30 s) instead of 3",
        "calib": "fwdfix",
        "train_kw": {"chunks": 6, "max_chains": 6000},
    },
    "augbias": {
        "desc": "calibfix + random IMU bias/noise augmentation in training",
        "calib": "fwdfix",
        "train_ds": lambda model, sp: AugDataset(sp["train"]),
    },
    "anchor": {
        "desc": "calibfix + learned absolute-speed anchor (complementary filter on speed)",
        "calib": "fwdfix",
        "model": SpeedAnchorNet,
        "fwd": anchor_forward,
        "loss_fn": anchor_loss,
        "chain_loss_fn": anchor_chain_loss,
        "chain_fn": lambda model, sp: (lambda m, ds, dev, dt, **k: lib.chained_drift_x(m, ds, dev, dt, step_fn=anchor_step(m), **k)),
    },
}


# ------------------------------------------------ idea: speed from turning (centripetal observer)
def centripetal_step_fn(w_thr=0.08, gain=0.05, k=10, vmax=40.0):
    """In a turn the car's lateral accel is v * yaw_rate, so v = a_lat / yaw_rate
    is a direct speed measurement that does not drift. Whenever the (1 s
    smoothed) yaw rate is above w_thr, nudge the integrated speed toward it
    with a small per-sample gain (a 1-state complementary filter)."""
    def step(raw, c, v, th, imu):
        acc = raw[..., 0] + c[..., 0]
        w = raw[..., 5] + c[..., 1]
        def trail(x):
            pad = torch.cat([x[:, :1].expand(-1, k - 1), x], 1)
            return pad.unfold(1, k, 1).mean(-1)
        ay, ws = trail(raw[..., 1]), trail(w)
        vobs = (ay / ws.where(ws.abs() > 1e-3, torch.full_like(ws, 1e-3))).clamp(0.0, vmax)
        use = (ws.abs() > w_thr) & (ay * ws > 0)
        vs, vv = [], v
        for i in range(acc.shape[1]):
            vv = (vv + acc[:, i] * lib.DT).clamp(0.0, 50.0)
            vv = torch.where(use[:, i], (1 - gain) * vv + gain * vobs[:, i], vv)
            vs.append(vv)
        speed = torch.stack(vs, 1)
        heading = th.unsqueeze(1) + torch.cumsum(w * lib.DT, 1)
        return speed, heading
    return step


def centripetal_chain_fn(**kw):
    sf = centripetal_step_fn(**kw)

    def chain_fn(model, ds, device, dt, **k):
        return lib.chained_drift_x(model, ds, device, dt, step_fn=sf, **k)
    return chain_fn


# ------------------------------------------------ idea: tell the network the current speed
class SpeedAwareNet(torch.nn.Module):
    """Same net with a 7th input channel: the speed the integrator starts the
    chunk from (true v0 for a single window, the model's own estimate when
    chained). Vibration, engine and turning signatures mean different things
    at 5 m/s and 25 m/s; the base net has to guess the speed from 5 s of IMU."""

    def __init__(self):
        super().__init__()
        self.net = lib.new_model(input_channels=7)
        self.cur_v0 = torch.zeros(1)

    def forward(self, x):
        v = self.cur_v0.reshape(-1, 1, 1).expand(x.shape[0], x.shape[1], 1) / 10.0
        return self.net(torch.cat([x, v.to(x.dtype)], -1))


def speedaware_forward(model, batch, dt, device):
    from src.train import forward_pass
    model.cur_v0 = batch["v0"]
    return forward_pass(model, batch, dt, device)


def speedaware_loss(model, splits, weights=None):
    from src.train import compute_loss
    w = weights or lib.LOSS_WEIGHTS

    def fn(b):
        return compute_loss(*speedaware_forward(model, b, lib.DT, "cpu"), w)
    return fn


def speedaware_chain_loss(model, splits, weights=None):
    from src.train import compute_loss
    w = weights or lib.LOSS_WEIGHTS

    def fn(b):
        imu, raw, sgt, pgt = b["imu"], b["imu_raw"], b["speed_gt"], b["pos_gt"]
        v, th = b["v0"], b["theta0"]
        B, K = imu.shape[:2]
        p_cum, gt_cum, dist = torch.zeros(B, 2), torch.zeros(B, 2), torch.zeros(B)
        total = 0.0
        for j in range(K):
            model.cur_v0 = v
            c = model(imu[:, j])
            speed = torch.clamp(v.unsqueeze(1) + torch.cumsum((raw[:, j, :, 0] + c[..., 0]) * lib.DT, 1), min=0.0)
            heading = integrate_heading(raw[:, j, :, 5] + c[..., 1], lib.DT, theta0=th)
            pos = dead_reckon_position(speed, heading, lib.DT)
            lj, _ = compute_loss(speed, pos, sgt[:, j], pgt[:, j], w)
            p_cum = p_cum + pos[:, -1]
            gt_cum = gt_cum + pgt[:, j, -1]
            dist = dist + torch.linalg.norm(pgt[:, j, 1:] - pgt[:, j, :-1], dim=-1).sum(1)
            total = total + lj + (100.0 * torch.linalg.norm(p_cum - gt_cum, dim=-1) / dist.clamp(min=1e-6)).mean()
            v, th = speed[:, -1], heading[:, -1]
        return total / K, {}
    return fn


EXPERIMENTS["speedaware"] = {
    "desc": "calibfix + bias augmentation + current speed fed to the network as an input",
    "calib": "fwdfix",
    "model": SpeedAwareNet,
    "train_ds": lambda model, sp: AugDataset(sp["train"]),
    "fwd": speedaware_forward,
    "loss_fn": speedaware_loss,
    "chain_loss_fn": speedaware_chain_loss,
    "chain_fn": lambda model, sp: lib.chained_drift_x,
}


EXPERIMENTS["augbias_long"] = {
    "desc": "augbias with double the training budget (12 single-window + 12 chained epochs)",
    "calib": "fwdfix",
    "train_ds": lambda model, sp: AugDataset(sp["train"]),
}


# ------------------------------------------------ round 3: richer sensor-error augmentation
class RichAugDataset(torch.utils.data.Dataset):
    """AugDataset plus the other errors a phone mount actually has: per-axis
    scale error (±3%), a few degrees of leftover yaw misalignment (leaks
    lateral accel into forward accel), a road-grade gravity leak that ramps
    across the window (±3% grade), and a random noise level per window."""

    def __init__(self, base, acc_b=0.15, gyro_b=0.01, scale=0.03, yaw_deg=4.0, grade=0.03, noise=(0.01, 0.04)):
        self.base, self.windows = base, base.windows
        self.acc_b, self.gyro_b, self.scale, self.yaw, self.grade, self.noise = \
            acc_b, gyro_b, scale, math.radians(yaw_deg), grade, noise

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = dict(self.base[i])
        raw = it["imu_raw"].clone()
        T = raw.shape[0]
        psi = float(torch.empty(1).uniform_(-self.yaw, self.yaw))
        c, s = math.cos(psi), math.sin(psi)
        for a, b in ((0, 1), (3, 4)):
            x, y = raw[:, a].clone(), raw[:, b].clone()
            raw[:, a], raw[:, b] = c * x - s * y, s * x + c * y
        raw = raw * (1 + torch.empty(6).uniform_(-self.scale, self.scale))
        g0, g1 = torch.empty(2).uniform_(-self.grade, self.grade)
        raw[:, 0] += 9.81 * torch.linspace(float(g0), float(g1), T)
        raw += torch.cat([torch.empty(3).uniform_(-self.acc_b, self.acc_b),
                          torch.empty(3).uniform_(-self.gyro_b, self.gyro_b)])
        sd = float(torch.empty(1).uniform_(*self.noise))
        raw += torch.randn_like(raw) * torch.tensor([sd] * 3 + [sd / 2] * 3)
        it["imu_raw"] = raw
        it["imu"] = raw
        return it


EXPERIMENTS["richaug"] = dict(EXPERIMENTS["speedaware"],
    desc="speedaware + richer augmentation (scale error, yaw misalignment, road-grade leak, varied noise)",
    train_ds=lambda model, sp: RichAugDataset(sp["train"]))


EXPERIMENTS["moredata"] = dict(EXPERIMENTS["speedaware"],
    desc="speedaware + 18 recovered IO-VNBD training drives (+537 min, prep_extra.py); val/test unchanged",
    splits=lambda: lib.build_splits("fwdfix", extra=True))


EXPERIMENTS["richmore"] = dict(EXPERIMENTS["moredata"],
    desc="speedaware + richer augmentation + 18 recovered training drives",
    train_ds=lambda model, sp: RichAugDataset(sp["train"]))


def slow_windows(ds, vmax=8.0):
    """Training windows whose mean speed is below vmax (stop-and-go traffic)."""
    from src.data.windowing import IOVNBDWindowDataset
    return IOVNBDWindowDataset([w for w in ds.windows if float(w.speed_gt.mean()) < vmax])


EXPERIMENTS["slowspec"] = dict(EXPERIMENTS["moredata"],
    desc="slow specialist: speedaware + richer augmentation trained only on windows below 8 m/s (with recovered drives); meant for the duo's low-speed half",
    train_ds=lambda model, sp: RichAugDataset(slow_windows(sp["train"])))


EXPERIMENTS["richmore_long"] = dict(EXPERIMENTS["richmore"],
    desc="richmore with double the training budget (12 single-window + 12 chained epochs)")


EXPERIMENTS["moredata_long"] = dict(EXPERIMENTS["moredata"],
    desc="moredata with double the training budget (12 single-window + 12 chained epochs)")


EXPERIMENTS["slowspec_plain"] = dict(EXPERIMENTS["moredata"],
    desc="slow specialist with plain bias augmentation (no richer augmentation), windows below 8 m/s, recovered drives",
    train_ds=lambda model, sp: AugDataset(slow_windows(sp["train"])))


# ------------------------------------------------ round 3: noisy speed input
def speedjitter_loss(model, splits, rel=0.1, absn=0.5):
    """Stage-1 loss where the speed fed to the network is the true v0 plus
    noise (10% + 0.5 m/s). In a real blackout that input is the model's own
    drifting estimate, so training only on the exact value overtrusts it.
    The integrator still starts from the true v0."""
    from src.train import compute_loss, forward_pass

    def fn(b):
        v = b["v0"]
        model.cur_v0 = (v * (1 + rel * torch.randn_like(v)) + absn * torch.randn_like(v)).clamp(min=0)
        return compute_loss(*forward_pass(model, b, lib.DT, "cpu"), lib.LOSS_WEIGHTS)
    return fn


EXPERIMENTS["speedjitter"] = dict(EXPERIMENTS["moredata"],
    desc="moredata + noisy speed input during single-window training (10% + 0.5 m/s)",
    loss_fn=speedjitter_loss)


EXPERIMENTS["moredata_swa"] = dict(EXPERIMENTS["moredata"],
    desc="moredata + weight average of the 6 chained epochs (kept if val agrees); targets the big seed-to-seed swings",
    train_kw={"swa": True})


EXPERIMENTS["trainval"] = dict(EXPERIMENTS["moredata"],
    desc="moredata trained on train + val drives (5 more drives), last epoch kept: val scores do not track test, so spend val on training",
    splits=lambda: lib.build_splits("fwdfix", extra=True, val_in_train=True),
    train_kw={"keep_last": True})
EXPERIMENTS["lastepoch"] = dict(EXPERIMENTS["moredata"],
    desc="moredata with the last epoch kept instead of the best-val epoch (control for trainval)",
    train_kw={"keep_last": True})


EXPERIMENTS["lastepoch_noextra"] = dict(EXPERIMENTS["speedaware"],
    desc="speedaware (no recovered drives) with the last epoch kept: checks whether the extra drives help once epoch picking is removed",
    train_kw={"keep_last": True})


EXPERIMENTS["lastepoch_chain6"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata with the chained fine-tune over 6 chunks (30 s) instead of 3",
    train_kw={"keep_last": True, "chunks": 6, "max_chains": 6000})


class BigSpeedAwareNet(SpeedAwareNet):
    """SpeedAwareNet with a wider net (CNN 64-128-128, GRU 256): more capacity
    now that the recovered drives nearly double the training data."""

    def __init__(self):
        super().__init__()
        self.net = lib.new_model(input_channels=7, cnn_channels=[64, 128, 128], gru_hidden=256)


EXPERIMENTS["lastepoch_big"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata with a wider network (CNN 64-128-128, GRU 256)",
    model=BigSpeedAwareNet, train_kw={"keep_last": True})


EXPERIMENTS["lastepoch_richaug"] = dict(EXPERIMENTS["richmore"],
    desc="last-epoch speedaware + recovered drives + richer augmentation (richmore with the last epoch kept)",
    train_kw={"keep_last": True})


SPEED_HEAVY = dict(lib.LOSS_WEIGHTS, speed=5.0)
EXPERIMENTS["lastepoch_speedw"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata with speed loss weight 5 (was 1): speed integration is the dominant error",
    loss_fn=lambda model, sp: speedaware_loss(model, sp, SPEED_HEAVY),
    chain_loss_fn=lambda model, sp: speedaware_chain_loss(model, sp, SPEED_HEAVY),
    train_kw={"keep_last": True})


EXPERIMENTS["lastepoch_s1heavy"] = dict(EXPERIMENTS["lastepoch"],
    desc="last-epoch moredata with the budget shifted to single-window training (run with --stage1 9 --stage2 3)")
EXPERIMENTS["lastepoch_s2heavy"] = dict(EXPERIMENTS["lastepoch"],
    desc="last-epoch moredata with the budget shifted to chained training (run with --stage1 3 --stage2 9)")
EXPERIMENTS["lastepoch_chain12"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata with the chained fine-tune over 12 chunks (60 s)",
    train_kw={"keep_last": True, "chunks": 12, "max_chains": 3000})
EXPERIMENTS["lastepoch_lowlr2"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata with a 3x lower chained fine-tune learning rate (1e-4)",
    train_kw={"keep_last": True, "lr2": 1e-4})
EXPERIMENTS["lastepoch_chain6_lowlr2"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata, chained fine-tune rate 1e-4 and the chain-6 setup combined",
    train_kw={"keep_last": True, "lr2": 1e-4, "chunks": 6, "max_chains": 6000})
EXPERIMENTS["lastepoch_hilr2"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata with a 2x higher chained fine-tune learning rate (6e-4)",
    train_kw={"keep_last": True, "lr2": 6e-4})
EXPERIMENTS["lastepoch_wd"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata with 100x stronger weight decay (1e-3)",
    train_kw={"keep_last": True, "wd": 1e-3})


class DropSpeedAwareNet(SpeedAwareNet):
    """SpeedAwareNet with GRU dropout 0.4 instead of 0.2."""

    def __init__(self):
        super().__init__()
        self.net = lib.new_model(input_channels=7, dropout=0.4)


EXPERIMENTS["lastepoch_drop4"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata with GRU dropout 0.4 instead of 0.2",
    model=DropSpeedAwareNet, train_kw={"keep_last": True})
EXPERIMENTS["lastepoch_chain6_drop4"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata, GRU dropout 0.4, chained fine-tune over 6 chunks (30 s)",
    model=DropSpeedAwareNet, train_kw={"keep_last": True, "chunks": 6, "max_chains": 6000})


class Drop6SpeedAwareNet(SpeedAwareNet):
    """SpeedAwareNet with GRU dropout 0.6."""

    def __init__(self):
        super().__init__()
        self.net = lib.new_model(input_channels=7, dropout=0.6)


EXPERIMENTS["lastepoch_drop6"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata with GRU dropout 0.6 (trend check after 0.4 helped comma)",
    model=Drop6SpeedAwareNet, train_kw={"keep_last": True})


class SmallSpeedAwareNet(SpeedAwareNet):
    """SpeedAwareNet with half the width (CNN 16-32-32, GRU 64): checks whether
    the 30-odd training drives are overfit by the default size."""

    def __init__(self):
        super().__init__()
        self.net = lib.new_model(input_channels=7, cnn_channels=[16, 32, 32], gru_hidden=64)


EXPERIMENTS["lastepoch_small"] = dict(EXPERIMENTS["moredata"],
    desc="last-epoch moredata with a half-width net (CNN 16-32-32, GRU 64)",
    model=SmallSpeedAwareNet, train_kw={"keep_last": True})
