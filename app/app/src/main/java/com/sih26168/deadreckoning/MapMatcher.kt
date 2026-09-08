package com.sih26168.deadreckoning

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

data class MapMatchResult(val lat: Double, val lon: Double, val edgeIdx: Int, val t: Double)

/**
 * Lightweight on-device counterpart to src/map_matching.py's
 * match_trajectory — snaps the live fused position onto the bundled
 * RoadGraph, not a raw "always trust the network's own estimate" line.
 *
 * This is deliberately NOT a port of leuvenmapmatching's full HMM/Viterbi
 * matcher (match_trajectory.py's DistanceMatcher) — that scores whole
 * candidate *paths* using the complete observed trajectory, which isn't
 * available yet in a live, one-point-at-a-time stream (a real deployed
 * system has the same constraint: it can't wait for future points either).
 * Instead this is a simpler greedy sequential matcher: for each new point,
 * score nearby candidate edges by distance *and* continuity with the
 * previous match, and take the best one now — a real, working
 * implementation of the same two ideas the offline matcher is built on
 * (map_matching.py's own docstring and avoid_goingback reasoning), not the
 * literal same algorithm.
 *
 *   - distance: how far the point is from the candidate edge.
 *   - continuity: bonus for staying on the same edge or moving to a
 *     directly-connected one; penalty for jumping to a disconnected edge
 *     or moving *backward* along the same edge — this is the
 *     non-holonomic constraint (a vehicle can't teleport sideways onto an
 *     unconnected road, or instantaneously reverse), same reasoning as
 *     map_matching.py's avoid_goingback and osmnx_graph_to_inmem_map's
 *     directed-edges-only structure (a one-way street's reverse direction
 *     genuinely isn't a candidate edge at all, not just a penalized one).
 *   - heading consistency: penalty for candidate edges whose own direction
 *     (node `from` -> `to`) points far from the vehicle's current fused
 *     heading. Added after a real bug found on a real device: at a
 *     junction/roundabout, several short edges can all sit within
 *     maxDistM with similar distance+continuity scores, and without this
 *     term the matcher would happily snap onto a perpendicular or
 *     doubling-back loop edge just because it was a few metres closer —
 *     visibly "going backwards" on the map relative to the vehicle's
 *     actual direction of travel. Continuity alone doesn't catch this:
 *     it only checks whether edges are topologically connected, not
 *     whether the *direction* of travel implied by hopping onto one makes
 *     any sense.
 *
 * Linear scan over all edges every call — fine for a graph this size
 * (~10k edges, matched at 10Hz is ~100k point-to-segment distance checks/
 * sec, trivial on a modern phone) without needing a spatial index; would
 * need one for a much larger bundled area.
 */
class MapMatcher(private val graph: RoadGraph, private val maxDistM: Double = 60.0) {

    private val refLat = graph.nodes[0].lat
    private val refLon = graph.nodes[0].lon
    private val nodeXY: Array<DoubleArray> = Array(graph.nodes.size) { i -> toXY(graph.nodes[i].lat, graph.nodes[i].lon) }

    private var lastEdgeIdx: Int = -1
    private var lastT: Double = 0.0

    private fun toXY(lat: Double, lon: Double): DoubleArray {
        val r = 6371000.0
        val lat0Rad = Math.toRadians(refLat)
        val x = Math.toRadians(lon - refLon) * r * cos(lat0Rad)
        val y = Math.toRadians(lat - refLat) * r
        return doubleArrayOf(x, y)
    }

    private fun toLatLon(x: Double, y: Double): DoubleArray {
        val r = 6371000.0
        val lat0Rad = Math.toRadians(refLat)
        val lat = refLat + Math.toDegrees(y / r)
        val lon = refLon + Math.toDegrees(x / (r * cos(lat0Rad)))
        return doubleArrayOf(lat, lon)
    }

    /** Distance from point p to segment [a,b], plus the projection
     * parameter t in [0,1] (0=a, 1=b) and the projected point itself. */
    private fun pointToSegment(p: DoubleArray, a: DoubleArray, b: DoubleArray): Triple<Double, Double, DoubleArray> {
        val abx = b[0] - a[0]; val aby = b[1] - a[1]
        val lenSq = abx * abx + aby * aby
        val t = if (lenSq < 1e-9) 0.0 else (((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / lenSq).coerceIn(0.0, 1.0)
        val projX = a[0] + t * abx
        val projY = a[1] + t * aby
        val dist = kotlin.math.sqrt((p[0] - projX) * (p[0] - projX) + (p[1] - projY) * (p[1] - projY))
        return Triple(dist, t, doubleArrayOf(projX, projY))
    }

    private fun angleDiff(a: Double, b: Double): Double {
        var d = a - b
        while (d > Math.PI) d -= 2 * Math.PI
        while (d < -Math.PI) d += 2 * Math.PI
        return abs(d)
    }

    /** Snap one live (lat, lon) onto the graph, or null if nothing is
     * within maxDistM (e.g. the bundled extract doesn't cover this area —
     * a real deployment would need the demo route's own extract).
     * headingRad is the vehicle's current fused heading (radians, 0=east,
     * ccw+, same convention as FusionState.heading) — see class doc for
     * why this matters, not just distance+continuity. */
    fun match(lat: Double, lon: Double, headingRad: Float): MapMatchResult? {
        val p = toXY(lat, lon)
        var bestIdx = -1
        var bestScore = Double.MAX_VALUE
        var bestT = 0.0
        var bestProj = p

        for (i in graph.edges.indices) {
            val e = graph.edges[i]
            val a = nodeXY[e.from]; val b = nodeXY[e.to]
            val (dist, t, proj) = pointToSegment(p, a, b)
            if (dist > maxDistM) continue

            var cost = dist

            val edgeHeading = atan2(b[1] - a[1], b[0] - a[0])
            val misalignment = angleDiff(edgeHeading, headingRad.toDouble())  // 0..pi
            cost += 12.0 * misalignment

            if (lastEdgeIdx >= 0) {
                when {
                    i == lastEdgeIdx && t < lastT -> cost += 15.0  // backward on the same edge — non-holonomic penalty
                    i == lastEdgeIdx -> cost -= 5.0                // staying on the same edge — mild continuity bonus
                    graph.edges[lastEdgeIdx].to == e.from -> cost -= 5.0  // directly connected, forward
                    else -> cost += 25.0                            // disconnected jump
                }
            }
            if (cost < bestScore) {
                bestScore = cost; bestIdx = i; bestT = t; bestProj = proj
            }
        }

        if (bestIdx < 0) return null
        lastEdgeIdx = bestIdx; lastT = bestT
        val ll = toLatLon(bestProj[0], bestProj[1])
        return MapMatchResult(ll[0], ll[1], bestIdx, bestT)
    }

    fun reset() {
        lastEdgeIdx = -1
        lastT = 0.0
    }
}
