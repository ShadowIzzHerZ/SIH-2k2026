"""
BiasCorrectionNet — the learned part of the hybrid dead-reckoning pipeline.

Design (matches the "AI speed/vibration filter" line item in the PS):
    Raw MEMS IMU on a phone has time-varying bias and vibration-induced
    noise that a fixed physics model can't remove. Instead of trying to
    hand-tune that away, this network looks at a sliding window of
    calibrated IMU samples and predicts a *residual correction* to what
    the physics-only integrator (strapdown_ins.py) would have produced:

        delta_v      -- correction to forward-speed rate (m/s^2 equiv.)
        delta_theta  -- correction to yaw rate (rad/s)

    These residuals get added to the calibrated accel/gyro *before*
    integration in the training loop (see train.py), so the whole thing
    is trained end-to-end against real trajectory drift, not just against
    a hand-picked "bias" target that we don't actually have ground truth
    for.

Architecture v2 (bigger, on purpose — see rationale below): a dilated,
residual 1D-CNN (TCN-style) feeding a GRU feeding a linear head. Same family
as v1 and as published IMU-only odometry work (IONet, RIDI), just deeper.

Why bigger, and why *this* bigger:
    Six warm-restart training cycles on v1 (32/64/64 CNN, 128-hidden 2-layer
    GRU, 214K params) all plateaued in the same ~68-70% val-drift band —
    and critically, *train* drift plateaued too (~58-61%, see
    results/train_history_6ch_baseline.json). A train-set plateau across
    repeated restarts with a healthy LR schedule is underfitting, not an
    optimization or overfitting problem: the network doesn't have enough
    capacity/context to fit the correction function, not that it's
    memorizing noise. So the fix tried here is capacity and receptive
    field, not different features or more regularization.
    - Residual blocks (two convs + a shortcut) instead of a plain conv
      stack, so going deeper doesn't just make gradients vanish.
    - Dilation growing per block (1, 2, 4 by default) — a plain 3-layer,
      kernel-5 stack only sees ~13 raw samples (~1.3s @ 10Hz) of context
      per output step; dilated residual blocks reach far wider (~tens of
      samples) for the same kernel size, letting the network use a whole
      braking/turning event's shape, not just a small local window of it.
    - GRU stays *unidirectional* (not bidirectional) on purpose despite
      training on whole 5s windows at once here: this network needs to run
      causally on-device as IMU samples stream in during a live GPS
      blackout, and a bidirectional GRU would need the *end* of a window
      before it could correct its *start* — a real latency cost this PS's
      real-time navigation use case shouldn't pay for an offline accuracy
      number. Depth (3 layers) and width (256 hidden) went up instead.

Result: ~1.9M params (~7.6MB fp32) vs. v1's 214K (~1MB) — still light
enough for ONNX -> ONNX Runtime Mobile / TFLite on a phone at 10Hz, just no
longer sub-megabyte. See configs/default.yaml's model section for the
exact sizes in use.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualConvBlock1D(nn.Module):
    """Two dilated Conv1d + BatchNorm + ReLU, with a residual shortcut
    (1x1 conv if the channel count changes, identity otherwise). Standard
    TCN-style block — lets the CNN go deeper (and reach a wider receptive
    field via dilation) without the plain stack's vanishing-gradient cost.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # keeps sequence length fixed (odd kernel_size)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.relu(h + self.shortcut(x))
        return self.dropout(h)


class BiasCorrectionNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 6,
        cnn_channels: list[int] | None = None,
        cnn_kernel_size: int = 5,
        cnn_dilations: list[int] | None = None,
        gru_hidden: int = 256,
        gru_layers: int = 3,
        dropout: float = 0.2,
        output_dim: int = 2,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [64, 128, 256]
        cnn_dilations = cnn_dilations or [2**i for i in range(len(cnn_channels))]  # 1, 2, 4, ...
        assert len(cnn_dilations) == len(cnn_channels), "cnn_dilations must match cnn_channels length"

        blocks = []
        in_ch = input_channels
        for out_ch, dilation in zip(cnn_channels, cnn_dilations):
            blocks.append(ResidualConvBlock1D(in_ch, out_ch, cnn_kernel_size, dilation, dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        # Unidirectional — see module docstring: this must run causally
        # on-device as samples stream in, not just perform well offline.
        self.gru = nn.GRU(
            input_size=in_ch,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, C) calibrated IMU window — [ax, ay, az, gx, gy, gz]
               in the leveled vehicle frame (see calibration.py), plus
               engineered channels appended after (see windowing.py's
               engineer_features()) when C > 6. input_channels is a config
               value, not hardcoded, so this works either way.

        Returns:
            corrections: (batch, T, output_dim) per-timestep residuals
            [delta_v, delta_theta], same length as the input window so it
            can be added directly to the physics integrator's inputs.
        """
        # Conv1d wants (batch, C, T)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # -> (batch, T, C')
        h, _ = self.gru(h)                                 # -> (batch, T, gru_hidden)
        out = self.head(h)                                 # -> (batch, T, output_dim)
        return out
