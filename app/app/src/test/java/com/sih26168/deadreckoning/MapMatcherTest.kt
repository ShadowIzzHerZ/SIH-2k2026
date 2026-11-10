package com.sih26168.deadreckoning

import kotlin.math.PI
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

/** MapMatcher's greedy snapping and RoadGraph's JSON parsing, on small
 * hand-built graphs (no Android Context needed). */
class MapMatcherTest {

    // ~1.11 m per 1e-5 degree of latitude; at the equator the same holds
    // for longitude, which keeps these fixtures easy to reason about.
    private val metresPerDeg = 6371000.0 * PI / 180.0
    private fun deg(m: Double) = m / metresPerDeg

    /** A 200 m eastbound road (nodes 0 -> 1) and a 200 m northbound road
     * (nodes 2 -> 3) far enough apart not to compete. */
    private fun twoRoads() = RoadGraph(
        nodes = listOf(
            MapNode(0.0, 0.0), MapNode(0.0, deg(200.0)),
            MapNode(deg(1000.0), 0.0), MapNode(deg(1200.0), 0.0),
        ),
        edges = listOf(MapEdge(0, 1), MapEdge(2, 3)),
    )

    @Test
    fun `parse reads nodes and directed edges`() {
        val g = RoadGraph.parse(
            """{"nodes":[{"lat":31.25,"lon":75.70},{"lat":31.26,"lon":75.71}],"edges":[[0,1],[1,0]]}"""
        )
        assertEquals(listOf(MapNode(31.25, 75.70), MapNode(31.26, 75.71)), g.nodes)
        assertEquals(listOf(MapEdge(0, 1), MapEdge(1, 0)), g.edges)
    }

    @Test
    fun `snaps a nearby point perpendicularly onto the road`() {
        val m = MapMatcher(twoRoads())
        // 10 m north of the eastbound road, 50 m along it, heading east.
        val r = m.match(deg(10.0), deg(50.0), headingRad = 0f)
        assertNotNull(r)
        assertEquals(0, r!!.edgeIdx)
        assertEquals(0.0, r.lat, deg(0.01))
        assertEquals(deg(50.0), r.lon, deg(0.01))
        assertEquals(0.25, r.t, 1e-6)
    }

    @Test
    fun `returns null when nothing is within maxDistM`() {
        val m = MapMatcher(twoRoads(), maxDistM = 60.0)
        assertNull(m.match(deg(500.0), deg(500.0), headingRad = 0f))
    }

    @Test
    fun `prefers the edge whose direction matches the heading`() {
        // Two-way street: the same segment as 0 -> 1 (east) and 1 -> 0
        // (west). Equal distance, so only the heading term separates them.
        val g = RoadGraph(
            nodes = listOf(MapNode(0.0, 0.0), MapNode(0.0, deg(200.0))),
            edges = listOf(MapEdge(0, 1), MapEdge(1, 0)),
        )
        assertEquals(0, MapMatcher(g).match(deg(5.0), deg(100.0), headingRad = 0f)!!.edgeIdx)
        assertEquals(1, MapMatcher(g).match(deg(5.0), deg(100.0), headingRad = PI.toFloat())!!.edgeIdx)
    }

    @Test
    fun `reset forgets the previous edge`() {
        val m = MapMatcher(twoRoads())
        m.match(deg(5.0), deg(100.0), headingRad = 0f)
        m.reset()
        // A point near the northbound road, heading north, matches it
        // without the disconnected-jump penalty mattering either way.
        val r = m.match(deg(1100.0), deg(5.0), headingRad = (PI / 2).toFloat())
        assertEquals(1, r!!.edgeIdx)
    }
}
