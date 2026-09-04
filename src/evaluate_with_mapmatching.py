"""
Measures the real improvement map-matching gives on top of the trained
dead-reckoning model's own output — not a synthetic drift test (that's
tests/test_map_matching.py, which validates the matcher itself works),
but the actual model's predictions on real held-out test windows.

Pipeline per window:
    1. run the trained model -> pos_pred (local xy, relative to window start)
    2. convert pos_pred to lat/lon using the window's true starting lat0/lon0
    3. download (or reuse a cached) OSM road graph around that area
    4. map-match the predicted trajectory onto it
    5. convert the snapped points back to local xy
    6. compare drift_pct before vs after snapping, against the same pos_gt

Only processes a sample of test windows (default 40), not the whole test
set — this makes real network calls (one OSM download per geographic area,
cached by rounded lat/lon so windows from the same drive mostly share one
download) and running it over all 18k+ test windows would be slow for what
is fundamentally a validation/demo run, not a production evaluation loop.

Run:
    python -m src.evaluate_with_mapmatching --config configs/default.yaml --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
import yaml

from src.data.windowing import load_dataset_splits
from src.map_matching import download_road_graph, match_trajectory, osmnx_graph_to_inmem_map
from src.models.bias_correction_net import BiasCorrectionNet
from src.models.strapdown_ins import drift_metric
from src.train import forward_pass, pick_device

R_EARTH = 6371000.0


def xy_to_latlon(xy: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Inverse of io_vnbd_loader.latlon_to_local_xy."""
    lat0_rad = np.radians(lat0)
    lat = lat0 + np.degrees(xy[:, 1] / R_EARTH)
    lon = lon0 + np.degrees(xy[:, 0] / (R_EARTH * np.cos(lat0_rad)))
    return np.stack([lat, lon], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--n_windows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_request_gap", type=float, default=5.0,
                         help="Minimum seconds between successful Overpass downloads, not just "
                              "retry backoff — the free API rate-limited/refused connections after "
                              "a handful of requests in quick succession on a prior run even when "
                              "each individual request succeeded, so pacing every request (not just "
                              "retries after a failure) matters.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = pick_device(cfg["train"]["device"])

    splits = load_dataset_splits(
        data_root=cfg["data"]["root"], variant=cfg["data"]["variant"],
        column_map=cfg["data"]["column_map"], sample_rate_hz=cfg["data"]["sample_rate_hz"],
        window_size=cfg["data"]["window_size"], window_stride=cfg["data"]["window_stride"],
        train_split=cfg["data"]["train_split"], val_split=cfg["data"]["val_split"],
        file_prefix=cfg["data"].get("file_prefix", ""),
    )
    test_ds = splits["test"]

    model = BiasCorrectionNet(
        input_channels=cfg["model"]["input_channels"], cnn_channels=cfg["model"]["cnn_channels"],
        cnn_kernel_size=cfg["model"]["cnn_kernel_size"],
        gru_hidden=cfg["model"]["gru_hidden"],
        gru_layers=cfg["model"]["gru_layers"], dropout=cfg["model"]["dropout"],
        output_dim=cfg["model"]["output_dim"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dt = 1.0 / cfg["data"]["sample_rate_hz"]

    random.seed(args.seed)
    idxs = random.sample(range(len(test_ds)), min(args.n_windows, len(test_ds)))
    print(f"evaluating {len(idxs)} sampled test windows")

    graph_cache: dict[tuple[int, int], "InMemMap"] = {}
    last_request_t = [0.0]  # mutable cell, closed over by get_map

    def get_map(lat0: float, lon0: float):
        # ~11km grid — coarser than a first pass (~1km) to raise the cache-hit
        # rate: the free Overpass API rate-limits/refuses connections after a
        # handful of requests in quick succession (confirmed — most downloads
        # in the first run failed with connection-refused after the first
        # couple succeeded), so minimizing total requests matters more here
        # than precision in which nearby windows share a graph.
        key = (round(lat0, 1), round(lon0, 1))
        if key not in graph_cache:
            # Proactive pacing, not just retry-after-failure backoff — the
            # free API rate-limited even between individually-successful
            # requests on a prior run, so every distinct download (not only
            # ones that just failed) waits out the minimum gap first.
            gap = time.time() - last_request_t[0]
            if gap < args.min_request_gap:
                time.sleep(args.min_request_gap - gap)

            last_err = None
            for attempt in range(3):
                try:
                    print(f"  downloading OSM graph for {key} (attempt {attempt + 1})...", flush=True)
                    G = download_road_graph(lat0, lon0, dist_m=1500)
                    graph_cache[key] = osmnx_graph_to_inmem_map(G, name=f"{key}")
                    last_request_t[0] = time.time()
                    break
                except Exception as e:
                    last_err = e
                    print(f"  download failed ({e}); backing off...", flush=True)
                    time.sleep(5 * (attempt + 1))  # backoff — be gentle with the free API
                    last_request_t[0] = time.time()
            else:
                raise last_err
        return graph_cache[key]

    before_drifts, after_drifts, match_rates = [], [], []

    with torch.no_grad():
        for n, idx in enumerate(idxs):
            print(f"[{n + 1}/{len(idxs)}] window {idx}...", flush=True)
            item = test_ds[idx]
            batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in item.items()}
            speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
            before = drift_metric(pos_pred, pos_gt).item()

            pos_pred_np = pos_pred[0].cpu().numpy()
            pos_gt_np = pos_gt[0].cpu().numpy()
            lat0, lon0 = item["lat0"], item["lon0"]
            if not np.isfinite(lat0) or not np.isfinite(lon0):
                print(f"  [skip] non-finite lat0/lon0")
                continue

            # Isolate each window's own failures (a download timeout/retry
            # exhaustion, or anything unexpected from the matcher itself) so
            # one bad window can't take down an otherwise-long unattended run.
            try:
                map_con = get_map(lat0, lon0)

                latlon_pred = xy_to_latlon(pos_pred_np, lat0, lon0)
                # subsample to ~1/sec for the matcher (matching every 10Hz sample is
                # unnecessary and slow; the matcher fills in the path between them).
                # drift_metric scores ONLY the trajectory's final point — a plain
                # range(0, T, step) doesn't land on T-1 for T=50, step=10, so the
                # one point that actually matters for the metric would never get
                # corrected (confirmed: that exact bug made before==after look
                # identical to 2 decimal places on the first run). np.linspace
                # guarantees both endpoints are included.
                n_sel = max(2, len(latlon_pred) // max(1, int(round(1.0 / dt))))
                sel = sorted(set(np.linspace(0, len(latlon_pred) - 1, n_sel, dtype=int).tolist()))
                path = [tuple(latlon_pred[i]) for i in sel]

                result = match_trajectory(map_con, path, obs_noise=30.0, max_dist=100.0)
            except Exception as e:
                print(f"  [skip] map-matching failed: {e}")
                continue
            match_rates.append(result.match_rate)

            snapped_xy = pos_pred_np.copy()  # fall back to unmatched prediction where snapping failed
            lat0_rad = np.radians(lat0)
            for local_i, global_i in enumerate(sel):
                p = result.snapped_latlon[local_i]
                if p is not None:
                    slat, slon = p
                    snapped_xy[global_i] = [
                        np.radians(slon - lon0) * R_EARTH * np.cos(lat0_rad),
                        np.radians(slat - lat0) * R_EARTH,
                    ]

            after = drift_metric(
                torch.from_numpy(snapped_xy).unsqueeze(0).float(),
                torch.from_numpy(pos_gt_np).unsqueeze(0).float(),
            ).item()
            print(f"  before {before:.1f}% -> after {after:.1f}% (match_rate {result.match_rate:.2f})", flush=True)

            before_drifts.append(before)
            after_drifts.append(after)

    before_drifts, after_drifts = np.array(before_drifts), np.array(after_drifts)
    print(f"\nwindows evaluated: {len(before_drifts)} | avg match rate: {np.mean(match_rates):.2f}")
    print(f"BEFORE map-matching — mean drift: {before_drifts.mean():.2f}% | median: {np.median(before_drifts):.2f}%")
    print(f"AFTER  map-matching — mean drift: {after_drifts.mean():.2f}% | median: {np.median(after_drifts):.2f}%")
    improvement = 100 * (1 - after_drifts.mean() / max(before_drifts.mean(), 1e-6))
    print(f"relative improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
