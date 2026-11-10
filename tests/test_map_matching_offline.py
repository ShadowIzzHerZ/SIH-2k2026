"""
Offline check of match_trajectory's per-observation output, on a small
hand-built road (no OSM download, no dataset).

The matcher inserts non-emitting states between observations that are
further apart than the road's node spacing, so the i-th entry of its best
state sequence is not observation i. Each snapped point must still be the
projection of ITS OWN observation onto the road.
"""
from __future__ import annotations

import pytest

pytest.importorskip("osmnx")
pytest.importorskip("leuvenmapmatching")

from leuvenmapmatching.map.inmem import InMemMap  # noqa: E402

from src.map_matching import match_trajectory  # noqa: E402

LAT0, LON0 = 52.0, 4.0


def _straight_road(n_nodes: int = 30, spacing_deg: float = 0.001) -> InMemMap:
    m = InMemMap("straight", use_latlon=True, use_rtree=True, index_edges=True)
    for i in range(n_nodes):
        m.add_node(i, (LAT0, LON0 + spacing_deg * i))
    for i in range(n_nodes - 1):
        m.add_edge(i, i + 1)
    m.setup_index()
    return m


def test_snapped_points_line_up_with_their_own_observations():
    road = _straight_road()
    # ~200 m between observations (road nodes every ~70 m), ~11 m north of the road
    path = [(LAT0 + 0.0001, LON0 + 0.003 * k + 0.0005) for k in range(8)]
    result = match_trajectory(road, path, obs_noise=30.0, max_dist=100.0)

    assert len(result.snapped_latlon) == len(path)
    assert result.snapped_latlon[-1] is not None
    for (lat, lon), snapped in zip(path, result.snapped_latlon):
        assert snapped is not None
        s_lat, s_lon = snapped
        assert s_lat == pytest.approx(LAT0, abs=1e-6)     # on the road
        assert s_lon == pytest.approx(lon, abs=1e-5)      # straight below its own observation
