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

Architecture: small 1D-CNN (local vibration/shape features) feeding a GRU
(temporal context across the window) feeding a linear head. This is the
same family used in published IMU-only odometry work (IONet, RIDI) and is
light enough to convert to ONNX -> ONNX Runtime Mobile / TFLite and run
at 10Hz on a phone with room to spare.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiasCorrectionNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 6,
        cnn_channels: list[int] | None = None,
        cnn_kernel_size: int = 5,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 2,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [32, 64, 64]

        conv_layers = []
        in_ch = input_channels
        for out_ch in cnn_channels:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*conv_layers)

        self.gru = nn.GRU(
            input_size=in_ch,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
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
