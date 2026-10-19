"""Second round of drift ideas (drift-search-2). Same split, recipe and
metric as ideas.py; each entry is one change on top of augbias (calibfix +
bias/noise augmentation), the best single-model recipe from round 1."""
from __future__ import annotations

import math

import numpy as np
import torch

from experiments.drift import aligned, ideas, lib


# ------------------------------------------------ test time: ensembles
class RotEnsemble(torch.nn.Module):
    """Runs the base net on copies of the IMU rotated by small mounting
    errors (yaw about vertical, pitch about lateral) and averages the
    *effective* forward accel and yaw rate the copies imply. Returned as a
    correction relative to the unrotated input, so every eval path that adds
    corrections to imu_raw works unchanged. Needs net input == imu_raw
    (true here: no normalisation)."""

    def __init__(self, base, yaw_deg=(0.0,), pitch_deg=(0.0,)):
        super().__init__()
        self.base = base
        self.rots = [(math.radians(y), math.radians(p)) for y in yaw_deg for p in pitch_deg]

    @staticmethod
    def _rot(x, yaw, pitch):
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        Rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=x.dtype)
        Ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=x.dtype)
        R = Ry @ Rz
        return torch.cat([x[..., :3] @ R.T, x[..., 3:6] @ R.T, x[..., 6:]], -1)

    def forward(self, x):
        acc, yaw = 0.0, 0.0
        for y, p in self.rots:
            xr = self._rot(x, y, p)
            if hasattr(self.base, "cur_v0"):
                self.base.cur_v0 = self.cur_v0
            c = self.base(xr)
            acc = acc + xr[..., 0] + c[..., 0]
            yaw = yaw + xr[..., 5] + c[..., 1]
        n = len(self.rots)
        return torch.stack([acc / n - x[..., 0], yaw / n - x[..., 5]], -1)


class MeanEnsemble(torch.nn.Module):
    """Average of several nets' corrections (e.g. two seeds of one recipe)."""

    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.cur_v0 = torch.zeros(1)

    def forward(self, x):
        out = 0.0
        for n in self.nets:
            if hasattr(n, "cur_v0"):
                n.cur_v0 = self.cur_v0
            out = out + n(x)
        return out / len(self.nets)


# ------------------------------------------------ idea: speed as an auxiliary task
class SpeedHeadNet(torch.nn.Module):
    """Base net with a 3rd output head read as absolute speed (m/s). Unlike
    round 1's anchor, the head is NOT used by the integrator: it is only an
    extra training target, so the shared CNN/GRU features have to encode how
    fast the car is going (the dominant error source). At inference the
    model is exactly the 2-output base net."""

    def __init__(self):
        super().__init__()
        self.net = lib.new_model(output_dim=3)
        self.aux = None

    def forward(self, x):
        c = self.net(x)
        self.aux = torch.nn.functional.softplus(c[..., 2]) * 10.0
        return c[..., :2]


# ------------------------------------------------ idea: longer IMU context
class HistDataset(torch.utils.data.Dataset):
    """Network input = the H samples of IMU BEFORE the window plus the window
    itself (IMU keeps running during a GPS blackout, so this history is
    always available in the app). imu_raw / targets stay the 5 s window.
    Windows too close to the drive start get edge-padded history. The same
    random bias/noise as AugDataset is applied to history and window."""

    def __init__(self, base, seqs, hist=100, aug=True, acc_b=0.15, gyro_b=0.01, noise=0.02):
        self.base, self.seqs, self.hist, self.aug = base, seqs, hist, aug
        self.acc_b, self.gyro_b, self.noise = acc_b, gyro_b, noise
        self.windows = base.windows

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = dict(self.base[i])
        w = self.windows[i]
        s = self.seqs[w.src]
        a0 = w.start - self.hist
        h = np.concatenate([s.accel[max(a0, 0):w.start], s.gyro[max(a0, 0):w.start]], 1).astype(np.float32)
        if a0 < 0:
            pad = np.repeat(it["imu_raw"][:1].numpy(), -a0, 0) if len(h) == 0 else np.repeat(h[:1], -a0, 0)
            h = np.concatenate([pad, h], 0)
        raw = it["imu_raw"]
        full = torch.cat([torch.from_numpy(h), raw], 0)
        if self.aug:
            b = torch.cat([torch.empty(3).uniform_(-self.acc_b, self.acc_b),
                           torch.empty(3).uniform_(-self.gyro_b, self.gyro_b)])
            full = full + b + self.noise * torch.randn_like(full)
            raw = full[self.hist:]
        it["imu_raw"] = raw.contiguous()
        it["imu"] = full
        return it


class HistNet(torch.nn.Module):
    """Base net run over history + window; only the window's corrections are used."""

    def __init__(self, T=lib.WS):
        super().__init__()
        self.net = lib.new_model()
        self.T = T

    def forward(self, x):
        return self.net(x)[:, -self.T:]


class HistSpeedNet(ideas.SpeedAwareNet):
    """Speed-aware net (current speed as an extra input) over history + window."""

    def forward(self, x):
        return super().forward(x)[:, -lib.WS:]


class HistSpeedAuxNet(HistSpeedNet):
    """HistSpeedNet plus the auxiliary absolute-speed head of SpeedHeadNet."""

    def __init__(self):
        super().__init__()
        self.net = lib.new_model(input_channels=7, output_dim=3)
        self.aux = None

    def forward(self, x):
        v = self.cur_v0.reshape(-1, 1, 1).expand(x.shape[0], x.shape[1], 1) / 10.0
        c = self.net(torch.cat([x, v.to(x.dtype)], -1))[:, -lib.WS:]
        self.aux = torch.nn.functional.softplus(c[..., 2]) * 10.0
        return c[..., :2]


class HistSpeedBigNet(HistSpeedNet):
    """HistSpeedNet with a wider GRU (192 hidden) and CNN (48/96/96)."""

    def __init__(self):
        super().__init__()
        self.net = lib.new_model(input_channels=7, gru_hidden=192, cnn_channels=[48, 96, 96])


class HistRotDataset(HistDataset):
    """HistDataset plus a random mounting rotation (yaw and pitch, up to
    +/-deg) of history and window alike, so the net learns to undo small
    calibration errors instead of only seeing perfectly aligned axes."""

    def __init__(self, *a, deg=3.0, **k):
        super().__init__(*a, **k)
        self.deg = deg

    def __getitem__(self, i):
        it = super().__getitem__(i)
        y, p = (math.radians(float(np.random.uniform(-self.deg, self.deg))) for _ in range(2))
        full = RotEnsemble._rot(it["imu"], y, p)
        it["imu"], it["imu_raw"] = full, full[self.hist:].contiguous()
        return it


class DistillNet(HistSpeedNet):
    """One HistSpeedNet trained to match the 3-model average (the current
    pick) as well as the truth, so the phone runs one model instead of three.
    While training, self.distill holds w * MSE(student, teacher) for the
    loss to add; the frozen teacher is not part of the saved weights."""

    TEACHERS = ("hist10_speedaware", "hist10_speedaware_s7", "hist10_speedaware_s3")

    def __init__(self, w=1000.0):
        super().__init__()
        nets = []
        for n in self.TEACHERS:
            m = HistSpeedNet()
            m.load_state_dict(torch.load(lib.ROOT / "checkpoints" / f"{n}.pt"))
            nets.append(m.eval().requires_grad_(False))
        object.__setattr__(self, "teacher", MeanEnsemble(nets))
        self.w, self.distill = w, None

    def forward(self, x):
        c = super().forward(x)
        if self.training:
            self.teacher.cur_v0 = self.cur_v0
            with torch.no_grad():
                t = self.teacher(x)
            self.distill = self.w * torch.nn.functional.mse_loss(c, t)
        else:
            self.distill = None
        return c


class SlowOversample(torch.utils.data.Dataset):
    """Single-window training set where windows slower than lib.SLOW (the
    weakest case, ~35% drift at 30 s) appear `k` times per epoch. The first
    len(windows) indices are the original windows, so chains built from
    .windows for the fine-tune stage are unchanged."""

    def __init__(self, base, k=3):
        self.base, self.windows = base, base.windows
        slow = [i for i, w in enumerate(self.windows) if float(np.mean(w.speed_gt)) < lib.SLOW]
        self.idx = list(range(len(self.windows))) + slow * (k - 1)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


def hist_splits(hist, extra=False):
    def make():
        sp = aligned.build_splits(extra=extra)
        for k in ("val", "test", "test_iovnbd", "test_comma"):
            sp[k] = HistDataset(sp[k], sp["_seqs"], hist=hist, aug=False)
        sp["_hist"] = hist
        return sp
    return make


def _aug_train(model, sp):
    return ideas.AugDataset(sp["train"])


def recipe(desc, model=lib.new_model, splits=aligned.build_splits, train_ds=_aug_train, set_v0=False,
           aux_w=0.0, end_w=None, **train_kw):
    """augbias recipe with the aligned (off-by-one fixed) integration in every
    loss and every evaluation."""
    return {
        "desc": desc, "model": model, "splits": splits, "train_ds": train_ds,
        "fwd": aligned.make_fwd(set_v0),
        "loss_fn": lambda m, sp: aligned.make_loss(m, set_v0=set_v0, aux_w=aux_w),
        "chain_loss_fn": lambda m, sp: aligned.make_chain_loss(m, end_w=end_w, aux_w=aux_w, set_v0=set_v0),
        "chain_fn": lambda m, sp: aligned.chained_drift,
        "train_kw": train_kw,
    }


EXPERIMENTS = {
    "aligned": recipe("augbias recipe with prediction and truth aligned (off-by-one fix) in training and scoring"),
    "aligned_speedaware": recipe("speedaware recipe (current speed as input), aligned", model=ideas.SpeedAwareNet,
                                 set_v0=True),
    "speedhead": recipe("aligned + auxiliary absolute-speed head (multi-task; not used by the integrator)",
                        model=SpeedHeadNet, aux_w=0.5),
    "hist10": recipe("aligned + network sees the 10 s of IMU before each 5 s window (15 s context)",
                     model=HistNet, splits=hist_splits(100),
                     train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"])),
    "hist25": recipe("aligned + network sees the 25 s of IMU before each 5 s window (30 s context)",
                     model=HistNet, splits=hist_splits(250),
                     train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"])),
    "hist10_endweight": recipe("15 s context + 6-chunk chained fine-tune weighted toward 30 s (hist10 + endweight)",
                               model=HistNet, splits=hist_splits(100),
                               train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"]),
                               end_w=[0.25, 0.25, 0.5, 0.5, 1.5, 3.0], chunks=6, max_chains=6000),
    "hist10_speedaware": recipe("15 s context + current speed as an input (hist10 + speedaware)",
                                model=HistSpeedNet, splits=hist_splits(100), set_v0=True,
                                train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"])),
    "hist5_speedaware": recipe("5 s context + current speed as an input (context-length sweep)",
                               model=HistSpeedNet, splits=hist_splits(50), set_v0=True,
                               train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"])),
    "hist15_speedaware": recipe("20 s context + current speed as an input (context-length sweep)",
                                model=HistSpeedNet, splits=hist_splits(150), set_v0=True,
                                train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"])),
    "hist10_speedaux": recipe("15 s context + speed input + auxiliary absolute-speed head",
                              model=HistSpeedAuxNet, splits=hist_splits(100), set_v0=True, aux_w=0.5,
                              train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"])),
    "hist10_speedbig": recipe("15 s context + speed input, wider network (GRU 192, CNN 48/96/96)",
                              model=HistSpeedBigNet, splits=hist_splits(100), set_v0=True,
                              train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"])),
    # from the drift-experiments thread: keep the last epoch (val does not flag bad
    # seeds) and add the 18 training-only drives recovered by prep_extra.py
    "hist10_speedaware_lx": recipe("15 s context + speed input, last epoch kept, + 18 recovered training drives",
                                   model=HistSpeedNet, splits=hist_splits(100, extra=True), set_v0=True,
                                   train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"]),
                                   keep_last=True),
    "hist10_speedaware_swa": recipe("15 s context + speed input, weights averaged over the chained fine-tune epochs (SWA)",
                                    model=HistSpeedNet, splits=hist_splits(100), set_v0=True,
                                    train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"]),
                                    swa=True),
    "hist10_speedaware_rotaug": recipe("15 s context + speed input, random +/-3 deg mounting rotation in training",
                                       model=HistSpeedNet, splits=hist_splits(100), set_v0=True,
                                       train_ds=lambda m, sp: HistRotDataset(sp["train"], sp["_seqs"], hist=sp["_hist"])),
    "hist10_speedaware_ch10": recipe("15 s context + speed input, 10 chained fine-tune epochs instead of 6",
                                     model=HistSpeedNet, splits=hist_splits(100), set_v0=True,
                                     train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"])),  # run with --stage2 10
    "hist10_speedaware_distill": recipe("one 15 s context + speed input model trained to match the 3-model average",
                                        model=DistillNet, splits=hist_splits(100), set_v0=True,
                                        train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"])),
    "hist10_speedaware_slow3": recipe("15 s context + speed input, slow (< 6 m/s) windows 3x as often in training",
                                      model=HistSpeedNet, splits=hist_splits(100), set_v0=True,
                                      train_ds=lambda m, sp: SlowOversample(
                                          HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"]))),
    # suggested by the drift-experiments thread: its best recipe (keep_last, extra
    # drives, 6-chunk chained fine-tune) plus this branch's 15 s history
    "hist10_speedaware_lx_chain6": recipe("15 s context + speed input, last epoch kept, + 18 extra drives, 6-chunk chained fine-tune",
                                          model=HistSpeedNet, splits=hist_splits(100, extra=True), set_v0=True,
                                          train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"]),
                                          keep_last=True, chunks=6, max_chains=6000),
    "hist10_speedaware_lx_chain6_ew": recipe("new best recipe with the 6-chunk chain loss weighted toward 30 s",
                                             model=HistSpeedNet, splits=hist_splits(100, extra=True), set_v0=True,
                                             train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"]),
                                             end_w=[0.25, 0.25, 0.5, 0.5, 1.5, 3.0],
                                             keep_last=True, chunks=6, max_chains=6000),
    "hist10_speedaware_lx_chain12": recipe("new best recipe with the chained fine-tune over 60 s (12 chunks) instead of 30 s",
                                           model=HistSpeedNet, splits=hist_splits(100, extra=True), set_v0=True,
                                           train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"]),
                                           keep_last=True, chunks=12, max_chains=3000),
    "hist15_speedaware_lx_chain6": recipe("new best recipe with 20 s of IMU context instead of 15 s",
                                          model=HistSpeedNet, splits=hist_splits(150, extra=True), set_v0=True,
                                          train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"]),
                                          keep_last=True, chunks=6, max_chains=6000),
    "hist10_speedaware_lx_chain6_ew2": recipe("new best recipe with the 6-chunk chain loss weighted harder toward 30 s",
                                              model=HistSpeedNet, splits=hist_splits(100, extra=True), set_v0=True,
                                              train_ds=lambda m, sp: HistDataset(sp["train"], sp["_seqs"], hist=sp["_hist"]),
                                              end_w=[0.0, 0.0, 0.25, 0.5, 2.0, 4.0],
                                              keep_last=True, chunks=6, max_chains=6000),
    "endweight": recipe("aligned, chained fine-tune over 6 chunks with the cumulative-drift loss weighted toward 30 s",
                        end_w=[0.25, 0.25, 0.5, 0.5, 1.5, 3.0], chunks=6, max_chains=6000),
}
