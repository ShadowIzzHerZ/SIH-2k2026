package com.sih26168.deadreckoning

import android.content.Context
import org.json.JSONObject

data class MapNode(val lat: Double, val lon: Double)
data class MapEdge(val from: Int, val to: Int)

/**
 * A small, pre-fetched OpenStreetMap road extract, bundled as an asset —
 * the on-device counterpart to src/map_matching.py's download_road_graph +
 * osmnx_graph_to_inmem_map, but fetched offline at build time (via
 * scripts, see app/README.md) instead of live on-device. A live download
 * mid-demo isn't reliable (map_matching.py's own docstring already flags
 * needing network access as a real constraint — "pre-download the extract
 * for your actual demo route/venue before the event... don't rely on
 * venue wifi").
 *
 * Directed edges only — same as osmnx_graph_to_inmem_map, so a one-way
 * street's reverse direction genuinely doesn't exist as an edge here
 * either (see MapMatcher's non-holonomic reasoning).
 */
class RoadGraph(context: Context, assetName: String = "road_graph.json") {
    val nodes: List<MapNode>
    val edges: List<MapEdge>

    init {
        val text = context.assets.open(assetName).bufferedReader().use { it.readText() }
        val root = JSONObject(text)
        val nodesJson = root.getJSONArray("nodes")
        val n = ArrayList<MapNode>(nodesJson.length())
        for (i in 0 until nodesJson.length()) {
            val o = nodesJson.getJSONObject(i)
            n.add(MapNode(o.getDouble("lat"), o.getDouble("lon")))
        }
        nodes = n

        val edgesJson = root.getJSONArray("edges")
        val e = ArrayList<MapEdge>(edgesJson.length())
        for (i in 0 until edgesJson.length()) {
            val pair = edgesJson.getJSONArray(i)
            e.add(MapEdge(pair.getInt(0), pair.getInt(1)))
        }
        edges = e
    }
}
