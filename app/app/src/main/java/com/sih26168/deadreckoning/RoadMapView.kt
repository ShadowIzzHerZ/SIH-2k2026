package com.sih26168.deadreckoning

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.preference.PreferenceManager
import android.util.AttributeSet
import android.widget.FrameLayout
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline

/**
 * Real OpenStreetMap-tile map view, replacing the earlier abstract scatter
 * plot — same idea as the Zen (DevStorm 2026) project's Leaflet + OSM-tile
 * map: a real "you are here" marker the user immediately understands,
 * colored by FusionMode the same way the plots already did (blue/red/
 * orange), a live-following camera, plus the map-matched (green) overlay.
 * No API key, no Google Play Services — osmdroid is the native-Android
 * equivalent of Zen's Leaflet+OSM approach, not Google Maps.
 *
 * Same simple addPoint/addMatchedPoint/clear API the old TrajectoryView
 * had, so MainActivity's tick loop barely changed switching to this.
 *
 * Needs network access to fetch OSM tiles (see AndroidManifest's INTERNET
 * permission) — the dead-reckoning pipeline itself stays fully offline;
 * only the map *background* needs a connection. Indoors/offline, the map
 * still functions (markers, trail, follow camera) on blank/uncached tiles.
 */
class RoadMapView(context: Context, attrs: AttributeSet? = null) : FrameLayout(context, attrs) {

    private val mapView: MapView
    private var youAreHereMarker: Marker? = null

    // Trail broken into mode-colored segments, like the old TrajectoryView's
    // per-point coloring but as connected polylines — a new segment starts
    // whenever the mode changes.
    private var currentSegmentMode: FusionMode? = null
    private var currentSegmentPoints = ArrayList<GeoPoint>()
    private var currentSegmentLine: Polyline? = null
    private var segmentCount = 0

    private var matchedPoints = ArrayList<GeoPoint>()
    private var matchedLine: Polyline? = null

    // Auto-follow pauses the instant the user touches the map — otherwise
    // every 100ms tick's setCenter() would snap straight back before a pan
    // gesture could even finish, making the recenter button pointless.
    // recenterOnLatest() (the FAB) turns following back on.
    private var following = true

    // One icon per mode, built once and reused — addPoint() is called at
    // 10Hz, and the old code built a brand-new GradientDrawable+Bitmap on
    // every single call, never recycling the previous one. That's real GC
    // churn (thousands of small bitmap allocations a minute on a long
    // run), a plausible real cause of the marker occasionally not
    // rendering/flickering on a live device — osmdroid's own draw pass can
    // land mid-GC-pause. Building 3 fixed icons up front and just
    // swapping between them removes the churn entirely.
    private val modeIcons: Map<FusionMode, android.graphics.drawable.BitmapDrawable> by lazy {
        FusionMode.entries.associateWith { mode ->
            android.graphics.drawable.BitmapDrawable(resources, drawableToBitmap(dotDrawable(modeColor(mode))))
        }
    }

    init {
        Configuration.getInstance().load(context, PreferenceManager.getDefaultSharedPreferences(context))
        Configuration.getInstance().userAgentValue = context.packageName

        // Explicit tile provider construction, not just the plain
        // MapView(context) single-arg constructor — belt-and-braces; the
        // actual bug found on a real device turned out to be stale
        // osmdroid SharedPreferences surviving an `adb install -r` across
        // repeated dev iterations, whose provider chain had somehow ended
        // up missing the online MapTileDownloader entirely (confirmed via
        // Configuration.isDebugTileProviders — only Assets/File-Archive/
        // SQL-Cache/Offline-Approximation ever appeared, never "Online
        // Tile Download Provider"). A clean uninstall + reinstall (fresh
        // prefs) fixed it outright; a real first-time install never hits
        // this since it never has stale prefs to begin with.
        val tileProvider = org.osmdroid.tileprovider.MapTileProviderBasic(context)
        tileProvider.tileSource = TileSourceFactory.MAPNIK
        mapView = MapView(context, tileProvider)
        mapView.setTileSource(TileSourceFactory.MAPNIK)
        mapView.setUseDataConnection(true)
        mapView.setMultiTouchControls(true)
        mapView.controller.setZoom(17.0)
        mapView.setOnTouchListener { _, event ->
            if (event.action == android.view.MotionEvent.ACTION_DOWN) following = false
            false  // don't consume — osmdroid still needs the event for pan/zoom
        }
        addView(mapView)
    }

    /**
     * Points the camera at a rough real fix before real tracking has
     * started — calibration (leveling, then yaw alignment) can take a
     * while during live GPS use, sometimes minutes without a strong
     * signal, and addPoint() (the only other place that ever calls
     * setCenter()) doesn't run until it finishes.
     *
     * Real bug found live: without this, the map view's camera never
     * moves off its default center — (0, 0), Null Island, open ocean —
     * until calibration finishes. OpenStreetMap's Mapnik style renders
     * open ocean as a flat light-blue fill, so this looked exactly like
     * "the map isn't loading", for as long as calibration was still
     * running — sometimes the entire session, during live GPS testing
     * with a weak fix. It wasn't a tile-loading failure at all: the map
     * was correctly rendering the middle of the Atlantic, because it was
     * never told where the phone actually is yet. Masked during replay
     * testing because replay's calibration completes in seconds (its GPS
     * speed is always confidently above the yaw-alignment threshold).
     *
     * Only acts before the real marker exists — once addPoint() takes
     * over, this must never fight the live follow-camera/pan state.
     */
    fun centerOnRoughLocation(lat: Double, lon: Double) {
        if (youAreHereMarker != null) return
        mapView.controller.setCenter(GeoPoint(lat, lon))
    }

    private fun modeColor(mode: FusionMode): Int = when (mode) {
        FusionMode.GNSS_TRACKING -> Color.parseColor("#2563eb")
        FusionMode.BLACKOUT -> Color.parseColor("#dc2626")
        FusionMode.BLEND -> Color.parseColor("#f59e0b")
    }

    private fun dotDrawable(colorHex: Int, sizeDp: Int = 20): GradientDrawable {
        val d = GradientDrawable()
        d.shape = GradientDrawable.OVAL
        d.setColor(colorHex)
        d.setStroke(3, Color.WHITE)
        val px = (sizeDp * resources.displayMetrics.density).toInt()
        d.setSize(px, px)
        return d
    }

    /** lat/lon — real-world coordinates (from FusionEngine.localXYToLatLon),
     * not the old view's raw local xy. */
    fun addPoint(lat: Double, lon: Double, mode: FusionMode) {
        val p = GeoPoint(lat, lon)

        if (mode != currentSegmentMode) {
            // Mode changed — start a fresh polyline segment so each leg of
            // the trip is its own solid color, same as the old dot view.
            currentSegmentMode = mode
            currentSegmentPoints = ArrayList()
            // Carry the last point over so segments connect with no gap.
            youAreHereMarker?.position?.let { currentSegmentPoints.add(it) }
            val line = Polyline(mapView)
            line.outlinePaint.color = modeColor(mode)
            line.outlinePaint.strokeWidth = 9f
            mapView.overlays.add(0, line)  // under markers
            currentSegmentLine = line
            segmentCount++
        }
        currentSegmentPoints.add(p)
        currentSegmentLine?.setPoints(currentSegmentPoints)

        if (youAreHereMarker == null) {
            val m = Marker(mapView)
            m.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
            m.title = "You are here"
            mapView.overlays.add(m)
            youAreHereMarker = m
            mapView.controller.setCenter(p)
        }
        youAreHereMarker?.position = p
        youAreHereMarker?.icon = modeIcons.getValue(mode)
        if (following) mapView.controller.setCenter(p)  // "follow me" camera, paused by user pan

        // Bound overlay count on a long-running demo — same reasoning as
        // the old view's 5000-point cap.
        if (segmentCount > 200) {
            mapView.overlays.removeAll { it is Polyline && it !== currentSegmentLine && it !== matchedLine }
            segmentCount = 1
        }
        mapView.invalidate()
    }

    fun addMatchedPoint(lat: Double, lon: Double) {
        val p = GeoPoint(lat, lon)
        if (matchedLine == null) {
            val line = Polyline(mapView)
            line.outlinePaint.color = Color.parseColor("#16a34a")
            line.outlinePaint.strokeWidth = 6f
            mapView.overlays.add(0, line)
            matchedLine = line
        }
        matchedPoints.add(p)
        if (matchedPoints.size > 3000) matchedPoints.removeAt(0)
        matchedLine?.setPoints(matchedPoints)
        mapView.invalidate()
    }

    fun clear() {
        mapView.overlays.clear()
        youAreHereMarker = null
        currentSegmentMode = null
        currentSegmentPoints = ArrayList()
        currentSegmentLine = null
        segmentCount = 0
        matchedPoints = ArrayList()
        matchedLine = null
        following = true
        mapView.invalidate()
    }

    /** Same idea as Zen's mapView.js recenter button — the camera already
     * follows automatically every tick (see addPoint), but a user who's
     * panned away needs an explicit way back. */
    fun recenterOnLatest() {
        following = true
        youAreHereMarker?.position?.let { mapView.controller.animateTo(it) }
    }

    private fun drawableToBitmap(d: GradientDrawable): android.graphics.Bitmap {
        val size = d.intrinsicWidth.coerceAtLeast(1)
        val bmp = android.graphics.Bitmap.createBitmap(size, size, android.graphics.Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(bmp)
        d.setBounds(0, 0, size, size)
        d.draw(canvas)
        return bmp
    }

    fun onResume() = mapView.onResume()
    fun onPause() = mapView.onPause()
}
