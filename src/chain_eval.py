"""
Chained-blackout evaluation: the metric the live app actually lives or dies by.

Per-window drift (train.py / evaluate.py) scores each 5 s window on its own,
starting from the TRUE heading and speed, and only looks at the final
*position*. The fusion engine (fusion.py, FusionEngine.kt) instead chains
5 s chunks, each one starting from the previous chunk's own final
heading/speed. A model can look great per window and still fail that way:
the cleanlabels checkpoint's yaw correction was ~0.03 rad/s over the first
10 samples of a window but ~2 rad/s over the last 10 (nothing in the loss
looked at final heading), so a 30 s chained blackout drifted ~90-100% while
its per-window median was 11%. This module reproduces the chained regime
directly from the held-out windows, so training can select on it.

Chains are rebuilt from back-to-back windows of the same recording
(Window.src / Window.start): chunk j is the window starting exactly
window_size*j samples after chunk 0. Chunk 0 starts from the true state;
chunks 1.. start from the model's own previous final speed/heading/position,
exactly as run_fusion does. Drift = final position error / true distance
travelled over the whole chain, the same definition as the PS metric.
"""
from __future__ import annotations

import numpy as np
import torch


def build_chains(dataset, window_size: int, n_chunks: int, chain_stride_chunks: int = 2,
                 max_chains: int | None = 800, seed: int = 0,
                 speed_range: tuple[float, float] | None = None) -> list[list[int]]:
    """Dataset indices of n_chunks back-to-back windows, per recording.
    chain_stride_chunks spaces chain starts so chains from one recording
    overlap less. Chains with any missing window (dropped as parked / no GT)
    are skipped, never bridged. speed_range=(lo, hi) keeps only chains whose
    mean ground-truth speed (m/s) is in [lo, hi): how a regime-specific model
    is scored only on the kind of driving it was trained for."""
    by_src: dict[str, dict[int, int]] = {}
    for idx, w in enumerate(dataset.windows):
        if w.start >= 0:
            by_src.setdefault(w.src, {})[w.start] = idx
    chains = []
    for starts in by_src.values():
        for st in sorted(starts):
            if (st // window_size) % chain_stride_chunks:
                continue
            ids = [starts.get(st + j * window_size) for j in range(n_chunks)]
            if all(i is not None for i in ids):
                if speed_range is not None:
                    mean_v = float(np.mean([dataset.windows[i].speed_gt.mean() for i in ids]))
                    if not (speed_range[0] <= mean_v < speed_range[1]):
                        continue
                chains.append(ids)
    if max_chains is not None and len(chains) > max_chains:
        rng = np.random.default_rng(seed)
        chains = [chains[i] for i in sorted(rng.choice(len(chains), max_chains, replace=False))]
    return chains


@torch.no_grad()
def chained_drift(model, dataset, device, dt: float, window_size: int = 50, n_chunks: int = 6,
                  max_chains: int | None = 800, v_clamp_max: float = 50.0, batch_size: int = 128,
                  speed_range: tuple[float, float] | None = None,
                  model_hi=None, switch_mps: float = 6.0) -> dict:
    """Chained n_chunks*window_size/10 s blackout drift over the dataset.
    Returns {} if no complete chain exists.

    model_hi: optional second ("duo") model. Each chunk is run by `model`
    (the urban/low-speed model) when the chain's CURRENT estimated speed is
    below switch_mps, else by model_hi (the highway model) — decided from the
    model's own state at the start of the chunk, exactly as fusion.py does."""
    chains = build_chains(dataset, window_size, n_chunks, max_chains=max_chains, speed_range=speed_range)
    if not chains:
        return {}
    model.eval()
    if model_hi is not None:
        model_hi.eval()
    drifts = []
    for b0 in range(0, len(chains), batch_size):
        cb = chains[b0:b0 + batch_size]
        B = len(cb)
        items = [[dataset[i] for i in ids] for ids in cb]
        v = torch.tensor([it[0]["v0"] for it in items], dtype=torch.float32, device=device)
        th = torch.tensor([it[0]["theta0"] for it in items], dtype=torch.float32, device=device)
        p = torch.zeros(B, 2, device=device)
        gt_end = torch.zeros(B, 2, device=device)
        dist = torch.zeros(B, device=device)
        for j in range(n_chunks):
            imu = torch.stack([it[j]["imu"] for it in items]).to(device)
            raw = torch.stack([it[j]["imu_raw"] for it in items]).to(device)
            pos_gt = torch.stack([it[j]["pos_gt"] for it in items]).to(device)
            c = model(imu)
            if model_hi is not None:
                use_hi = (v >= switch_mps).view(-1, 1, 1)
                c = torch.where(use_hi, model_hi(imu), c)
            acc = raw[..., 0] + c[..., 0]
            yaw = raw[..., 5] + c[..., 1]
            # per-step clamp like run_fusion (a window-level cumsum can't clamp mid-way)
            vs = []
            vv = v
            for k in range(acc.shape[1]):
                vv = (vv + acc[:, k] * dt).clamp(0.0, v_clamp_max)
                vs.append(vv)
            speed = torch.stack(vs, dim=1)
            heading = th.unsqueeze(1) + torch.cumsum(yaw * dt, dim=1)
            step = torch.stack([speed * torch.cos(heading), speed * torch.sin(heading)], dim=-1) * dt
            p = p + step.sum(dim=1)
            v, th = speed[:, -1], heading[:, -1]
            gt_end = gt_end + pos_gt[:, -1, :]
            dist = dist + torch.linalg.norm(pos_gt[:, 1:] - pos_gt[:, :-1], dim=-1).sum(dim=1)
        drifts.extend((100.0 * torch.linalg.norm(p - gt_end, dim=-1) / dist.clamp(min=1e-6)).cpu().tolist())
    d = np.array(drifts)
    return {"n_chains": int(len(d)), "chain_seconds": n_chunks * window_size * dt,
            "mean_drift_pct": float(d.mean()), "median_drift_pct": float(np.median(d)),
            "p90_drift_pct": float(np.percentile(d, 90)), "pass_rate_pct": float((d < 10.0).mean() * 100)}


class ChainTrainDataset(torch.utils.data.Dataset):
    """K back-to-back windows of one recording per item, for chained-chunk
    training (train.py --chain_train_chunks). Chunk 0 starts from the true
    state; the training loop carries the model's own end state into chunk 1..
    (see train.chain_loss), so the network practises recovering from its own
    heading/speed errors instead of always being handed the truth."""

    def __init__(self, dataset, chains: list[list[int]]):
        self.dataset = dataset
        self.chains = chains

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, i):
        items = [self.dataset[j] for j in self.chains[i]]
        out = {k: torch.stack([it[k] for it in items]) for k in ("imu", "imu_raw", "speed_gt", "pos_gt")}
        out["v0"], out["theta0"] = items[0]["v0"], items[0]["theta0"]
        return out
