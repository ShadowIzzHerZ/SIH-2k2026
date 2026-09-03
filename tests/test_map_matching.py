"""
Validates map_matching.py against a real road network with a known-answer
drift-recovery scenario — not just "it runs without error."

Uses a real GPS segment from the IO-VNBD test data (so the "true" road
path is real, not synthetic), applies a synthetic drift to it (simulating
what the dead-reckoning model's imperfect output looks like), and checks
that map-matching pulls the drifted path substantially back toward the
true road path. This kind of test matters here specifically because a
first pass at this same check silently returned "no improvement" — not
because matching didn't work, but because of a wrong-API bug (reading the
deduplicated route geometry instead of each observation's actual snapped
point). A test that just asserts "no exception" would never have caught
that; asserting a *quantitative expected improvement* would.

Needs network access (downloads a small real OSM extract). Not run as
part of a fast/offline check — run explicitly when touching map_matching.py.

Run:
    python -m tests.test_map_matching
"""
from __future__ import annotations

import numpy as np

from src.data.io_vnbd_loader import latlon_to_local_xy, load_sequence
from src.map_matching import download_road_graph, match_trajectory, osmnx_graph_to_inmem_map

# A real ~740m driving segment from S-Vta1a.csv (Leicester, UK area),
# picked for steady >3 m/s driving (not idle/noisy). Saved as raw
# coordinates so this test doesn't depend on the full dataset being present.
_SEG_PATH = "data/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-Vta1a.csv"
_SEG_START, _SEG_LEN = 10200, 300


def _load_test_segment():
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(open("configs/default.yaml"))
    seq = load_sequence(Path(_SEG_PATH), cfg["data"]["column_map"])
    lat = seq.lat[_SEG_START:_SEG_START + _SEG_LEN]
    lon = seq.lon[_SEG_START:_SEG_START + _SEG_LEN]
    return lat, lon


def test_map_matching_recovers_synthetic_drift():
    lat, lon = _load_test_segment()
    true_xy = latlon_to_local_xy(lat, lon)

    np.random.seed(0)
    n = len(lat)
    t = np.linspace(0, 1, n)
    drift_direction = np.array([1.0, 0.3])   # roughly perpendicular to this road segment
    drift_magnitude = 40.0                    # metres, growing over the segment — comparable
                                               # to real drift over a several-hundred-metre blackout
    drifted_xy = true_xy + np.outer(t, drift_direction) * drift_magnitude \
        + np.random.normal(0, 3, true_xy.shape)

    lat0, lon0 = lat[0], lon[0]
    R = 6371000.0
    lat0_rad = np.radians(lat0)
    drifted_lat = lat0 + np.degrees(drifted_xy[:, 1] / R)
    drifted_lon = lon0 + np.degrees(drifted_xy[:, 0] / (R * np.cos(lat0_rad)))

    G = download_road_graph(lat.mean(), lon.mean(), dist_m=1200)
    map_con = osmnx_graph_to_inmem_map(G)

    idx = np.arange(0, n, 10)
    drifted_path = list(zip(drifted_lat[idx].tolist(), drifted_lon[idx].tolist()))
    true_xy_sub, drifted_xy_sub = true_xy[idx], drifted_xy[idx]

    result = match_trajectory(map_con, drifted_path, obs_noise=25.0, max_dist=80.0)
    print(f"match_rate: {result.match_rate:.2f}")

    before_err = np.linalg.norm(drifted_xy_sub - true_xy_sub, axis=1)

    after_errs = []
    for i, p in enumerate(result.snapped_latlon):
        if p is None:
            continue
        slat, slon = p
        sxy = np.array([
            np.radians(slon - lon0) * R * np.cos(lat0_rad),
            np.radians(slat - lat0) * R,
        ])
        after_errs.append(np.linalg.norm(sxy - true_xy_sub[i]))
    after_errs = np.array(after_errs)

    print(f"BEFORE mean err: {before_err.mean():.1f}m | AFTER mean err: {after_errs.mean():.1f}m "
          f"(n={len(after_errs)}/{len(idx)} matched)")

    assert result.match_rate > 0.9, "matcher should match almost all observations on a real road segment"
    assert len(after_errs) > 0
    assert after_errs.mean() < before_err.mean() * 0.7, (
        "map-matching should meaningfully reduce drift error, not just re-trace the drifted path "
        "(if this fails, check you're reading edge_m.pi per-observation, not the deduplicated route)"
    )


if __name__ == "__main__":
    test_map_matching_recovers_synthetic_drift()
    print("\n✅ map-matching drift-recovery test passed")
