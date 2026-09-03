"""
Snaps a trajectory (predicted or raw GPS) onto the real road network.

This is the PS's explicitly-named "map-matching against OpenStreetMap with
non-holonomic constraints" component — a separate piece from the dead-
reckoning network. The two are meant to work together: the network gives a
continuously-updated position estimate through a GPS blackout, and map-
matching corrects that estimate against the actual road geometry (a car
can't be in the middle of a building, and can't move sideways off a road),
which is exactly the kind of error dead-reckoning alone tends to accumulate
over a blackout.

Uses an HMM-based matcher (leuvenmapmatching's DistanceMatcher — a Python
implementation of the same family of algorithm as Newson & Krumm 2009,
the standard approach for this problem) rather than "snap to nearest road
point," which breaks badly on noisy trajectories: nearest-point snapping
has no memory of the route so far, so a single noisy sample can jump the
match onto a completely wrong nearby road (e.g. a parallel street) with no
way to recover. The HMM instead scores whole candidate paths by how well
they explain the observed trajectory *and* how plausible the implied route
is on the actual road network (favoring routes that don't require
impossible turns or backtracking), which is what makes it robust to the
kind of noisy, drifting input this pipeline actually produces.

Road network comes from OpenStreetMap via osmnx — free, no API key, but
needs network access at data-prep time (pre-download the extract for your
actual demo route/venue before the event, per sih.md — don't rely on venue
wifi).
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import osmnx as ox
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher


def download_road_graph(lat: float, lon: float, dist_m: float = 2000,
                         network_type: str = "drive") -> nx.MultiDiGraph:
    """Fetch a road network graph centered on a point from OpenStreetMap.

    Args:
        lat, lon: center point (e.g. your demo route's midpoint, or the
            GNSS-blackout location itself).
        dist_m: radius in metres to fetch around the center.
        network_type: "drive" excludes footpaths/etc — right for a vehicle.

    Returns:
        A networkx MultiDiGraph in osmnx's standard format. Cache this
        (e.g. `ox.save_graphml`) for the actual demo route ahead of time —
        don't fetch live at the venue.

    simplify=False on purpose: osmnx's simplification collapses a curving
    road down to just its intersection endpoints (confirmed: a 1km-radius
    area went from 108 shape-point nodes to 8 once simplified), which turns
    every road into a straight line between distant points for our
    purposes — we only use each node's lat/lon, not the edge geometry
    attribute simplification preserves. Keeping the unsimplified graph
    means edges connect closely-spaced points that trace the road's real
    curve, which matters a lot for how well the matcher can snap to it.
    """
    return ox.graph_from_point((lat, lon), dist=dist_m, network_type=network_type, simplify=False)


def osmnx_graph_to_inmem_map(G: nx.MultiDiGraph, name: str = "road_map") -> InMemMap:
    """Convert an osmnx road graph into the InMemMap format leuvenmapmatching's
    matchers expect. osmnx nodes carry (y=lat, x=lon); edges are directed
    (osmnx already encodes one-way streets correctly, which we keep — a
    non-holonomic vehicle can't match onto a road the wrong way)."""
    map_con = InMemMap(name, use_latlon=True, use_rtree=True, index_edges=True)
    for node_id, data in G.nodes(data=True):
        map_con.add_node(node_id, (data["y"], data["x"]))
    for u, v in G.edges():
        map_con.add_edge(u, v)
    map_con.setup_index()
    return map_con


@dataclass
class MapMatchResult:
    snapped_latlon: list[tuple[float, float] | None]  # per-observation corrected position,
                                                         # same length/order as the input path;
                                                         # None where that observation never got matched
    route_node_path: list       # deduplicated road-graph nodes for the whole route (for drawing the path, not per-step correction)
    n_observations: int
    n_matched: int               # how many input points got matched before any early stop

    @property
    def match_rate(self) -> float:
        return self.n_matched / max(1, self.n_observations)


def match_trajectory(map_con: InMemMap, latlon_path: list[tuple[float, float]],
                      obs_noise: float = 10.0, max_dist: float = 100.0) -> MapMatchResult:
    """Snap a (possibly noisy/drifted) lat/lon trajectory onto the road graph.

    Args:
        map_con: from osmnx_graph_to_inmem_map().
        latlon_path: the trajectory to snap — e.g. the dead-reckoning
            model's output converted back to lat/lon, or raw noisy GPS.
        obs_noise: expected GPS/position noise in metres — how far off the
            true road position an observation is likely to be. Larger for
            a dead-reckoning-derived path (which can drift tens of metres)
            than for raw GPS.
        max_dist: max distance (metres) from a point to a road candidate to
            even consider it a match — keeps the search space sane.

    Returns:
        MapMatchResult. The per-observation corrected position comes from
        each matched state's `edge_m.pi` — the point's actual projection
        onto its matched road edge — NOT from `path_pred_onlynodes`, which
        is the deduplicated *route geometry* (repeated/adjacent
        observations matched to the same node collapse to one entry, so
        it's a different length than the input and not indexable
        per-observation). Conflating the two was a real bug caught by
        checking a synthetic drift-recovery test that came back with
        implausibly no improvement — always sanity-check a matcher against
        a case where you know what "better" should look like, not just
        that it runs without error.
    """
    matcher = DistanceMatcher(map_con, obs_noise=obs_noise, max_dist=max_dist,
                               non_emitting_states=True)
    states, last_idx = matcher.match(latlon_path)

    snapped: list[tuple[float, float] | None] = [None] * len(latlon_path)
    for i, m in enumerate(matcher.lattice_best[:len(latlon_path)]):
        if m is not None and m.edge_m is not None:
            snapped[i] = m.edge_m.pi

    route_nodes = list(matcher.path_pred_onlynodes) if hasattr(matcher, "path_pred_onlynodes") else []
    return MapMatchResult(
        snapped_latlon=snapped,
        route_node_path=route_nodes,
        n_observations=len(latlon_path),
        n_matched=last_idx + 1 if states else 0,
    )
