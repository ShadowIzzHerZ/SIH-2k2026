"""
Validates the two non-holonomic guarantees match_trajectory() now makes
explicit (see map_matching.py's match_trajectory docstring/avoid_goingback
comment) against a small synthetic road, not a real OSM download — so this
runs fast and offline, unlike tests/test_map_matching.py's real-network
drift-recovery check.

  1. A one-way street structurally cannot be matched the wrong way: the
     graph fed to the matcher only ever has the directed edge(s) osmnx
     itself resolved, so there is no reverse edge to snap onto at all.
     Checked directly against osmnx_graph_to_inmem_map(), not a hand-rolled
     substitute for it, so this actually covers the function used in
     production.

  2. A noisy observation that appears to jump backward along a road
     shouldn't make the *matched* path teleport backward with it —
     leuvenmapmatching's avoid_goingback penalty (now explicitly enabled,
     not left as an implicit library default) should hold the match near
     its forward-most point instead. A soft penalty, not a hard ban — the
     assertion below checks the backward move is much smaller than the raw
     injected jump, not that it's exactly zero.

Run:
    python -m tests.test_non_holonomic_matching
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.map_matching import match_trajectory, osmnx_graph_to_inmem_map

R_EARTH = 6371000.0
LAT0 = 40.0  # arbitrary reference latitude — only cos(LAT0) matters, for the lon->metres scale


def _straight_one_way_road(n_nodes: int = 8, spacing_m: float = 15.0) -> nx.MultiDiGraph:
    """A straight one-way road running east, n_nodes apart by spacing_m,
    as a osmnx-shaped MultiDiGraph (node attrs y=lat, x=lon; directed edges
    only forward) — exactly what osmnx_graph_to_inmem_map expects, so this
    exercises the real function instead of building an InMemMap by hand."""
    G = nx.MultiDiGraph()
    lat0_rad = np.radians(LAT0)
    for i in range(n_nodes):
        lon = np.degrees((i * spacing_m) / (R_EARTH * np.cos(lat0_rad)))
        G.add_node(i, y=LAT0, x=lon)
    for i in range(n_nodes - 1):
        G.add_edge(i, i + 1)  # one direction only — a one-way street
    return G


def _progress_m(lat: float, lon: float) -> float:
    """Distance east (metres) from the road's start node — since the road
    is a perfectly straight east-west line, this is a direct, exact measure
    of how far along it a point is, independent of which edge it snapped to."""
    lat0_rad = np.radians(LAT0)
    return np.radians(lon) * R_EARTH * np.cos(lat0_rad)


def test_one_way_edges_have_no_reverse():
    G = _straight_one_way_road()
    map_con = osmnx_graph_to_inmem_map(G)

    for i in range(len(G.nodes) - 1):
        forward_nbrs = map_con.graph[i][1]
        backward_nbrs = map_con.graph[i + 1][1]
        assert i + 1 in forward_nbrs, f"expected forward edge {i}->{i + 1} to exist"
        assert i not in backward_nbrs, (
            f"found a reverse edge {i + 1}->{i} on what should be a one-way street — "
            f"osmnx_graph_to_inmem_map is adding edges the source graph doesn't have"
        )
    print("test_one_way_edges_have_no_reverse: OK")


def test_matcher_avoids_backward_teleport_along_one_way():
    G = _straight_one_way_road(n_nodes=8, spacing_m=15.0)
    map_con = osmnx_graph_to_inmem_map(G)
    lat0_rad = np.radians(LAT0)

    def to_latlon(progress_m: float) -> tuple[float, float]:
        lon = np.degrees(progress_m / (R_EARTH * np.cos(lat0_rad)))
        return LAT0, lon

    # Forward progress with small noise, then one deliberate backward jump
    # (simulating GPS/dead-reckoning jitter), then forward again.
    true_progress = np.array([0, 12, 24, 36, 48, 30, 60, 72, 84])  # index 5 jumps back 18m
    rng = np.random.default_rng(0)
    noisy_progress = true_progress + rng.normal(0, 1.5, size=len(true_progress))
    path = [to_latlon(p) for p in noisy_progress]

    # obs_noise=30 (same order of magnitude as evaluate_with_mapmatching.py
    # already uses for a dead-reckoning-derived path) tells the matcher a
    # ~15-20m gap between an observation and its matched point isn't
    # unusual — which is what actually gives avoid_goingback's prior room
    # to win against a single noisy backward-looking observation, instead
    # of positional accuracy alone dominating the decision.
    result = match_trajectory(map_con, path, obs_noise=30.0, max_dist=60.0)
    assert result.match_rate > 0.8, f"expected most points to match a road right on top of them, got {result.match_rate:.2f}"

    matched_progress = [
        _progress_m(*p) for p in result.snapped_latlon if p is not None
    ]
    assert len(matched_progress) >= 2

    backward_steps = [
        matched_progress[i] - matched_progress[i + 1]
        for i in range(len(matched_progress) - 1)
        if matched_progress[i + 1] < matched_progress[i]
    ]
    max_backward = max(backward_steps, default=0.0)
    raw_backward_jump = true_progress[4] - true_progress[5]  # 18m, the injected blip

    print(f"matched progress: {[round(p, 1) for p in matched_progress]}")
    print(f"max backward step in matched path: {max_backward:.1f}m (raw injected jump: {raw_backward_jump:.1f}m)")

    assert max_backward < 2.0, (
        f"matched path moved backward by {max_backward:.1f}m along a one-way road for a single "
        f"noisy observation — avoid_goingback doesn't seem to be taking effect "
        f"(check match_trajectory still passes it)"
    )


if __name__ == "__main__":
    test_one_way_edges_have_no_reverse()
    test_matcher_avoids_backward_teleport_along_one_way()
    print("\n✅ non-holonomic matching tests passed")
