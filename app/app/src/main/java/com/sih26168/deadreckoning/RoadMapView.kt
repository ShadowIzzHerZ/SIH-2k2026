package com.sih26168.deadreckoning

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Handler
import android.os.Looper
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
 * The marker blinks (alpha, see blinkRunnable) to read as "this is live",
 * not a static pin — same idea as a recording indicator.
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

    // Which mode the marker is currently showing (its BRIGHT/DIM variant
    // swaps with blinkOn — see applyIcon). Debugged live via logcat: first
    // tried mutating the shared icon Drawable's .alpha directly, gated to
    // only reassign `.icon =` on an actual mode change (reassigning every
    // 10Hz tick, even to the same object, was silently resetting alpha —
    // confirmed via a debug log: alpha correctly went 255->80 on a blink
    // tick, but was back to 255 by the very next blink tick 550ms later
    // with nothing in blinkRunnable itself setting it back). That gate
    // alone wasn't enough — alpha still reset, meaning something in
    // osmdroid's own Marker draw/position-update path resets a Drawable's
    // alpha independent of whether `.icon =` gets reassigned. Rather than
    // chase that further, switched to a mechanism already proven to work:
    // `.icon =` reassignment itself is how mode-color switching already
    // renders correctly, so the blink swaps between two fully-baked
    // bitmaps (modeIcons / modeIconsDim) instead of mutating alpha on one.
    private var currentIconMode: FusionMode? = null

    private fun applyIcon() {
        val mode = currentIconMode ?: return
        youAreHereMarker?.icon = (if (blinkOn) modeIcons else modeIconsDim).getValue(mode)
    }

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

    // The blink's dim phase — a second, fully baked bitmap per mode (real
    // semi-transparent pixel data, composited once here), not a runtime
    // alpha mutation on modeIcons' bitmaps. See currentIconMode's doc for
    // why: mutating Drawable.alpha directly didn't stick, something in
    // osmdroid's own Marker draw path reset it independent of that.
    private val modeIconsDim: Map<FusionMode, android.graphics.drawable.BitmapDrawable> by lazy {
        FusionMode.entries.associateWith { mode ->
            android.graphics.drawable.BitmapDrawable(resources, drawableToBitmap(dotDrawable(modeColor(mode), alpha = 80)))
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
        // Real bug found live (see updateRoughLocation's own doc for the
        // exact same failure mode there): with no setCenter() call, the
        // camera sits at its default (0, 0) -- Null Island, open ocean --
        // until the first GPS fix or replay tick arrives, which can take a
        // few seconds (permission dialog, calibration leveling wait, GPS
        // acquisition indoors). OpenStreetMap's Mapnik style renders open
        // ocean as a flat light-blue fill, which reads exactly like "the
        // map failed to load," not "still waiting for a fix." Centering
        // immediately on a real, meaningful place -- Jalandhar, Punjab,
        // the same area the bundled road_graph.json and replay demos cover
        // -- means real streets are on screen from the very first frame.
        // updateRoughLocation/addPoint silently take over and recenter as
        // soon as a real fix (live or replay) exists; this is only ever
        // visible for that brief startup window.
        mapView.controller.setCenter(GeoPoint(31.2576, 75.7065))
        mapView.setOnTouchListener { _, event ->
            if (event.action == android.view.MotionEvent.ACTION_DOWN) following = false
            false  // don't consume — osmdroid still needs the event for pan/zoom
        }
        addView(mapView)
        // Blink loop starts in onResume() (called right after this, as
        // part of the normal Activity lifecycle) — declaring blinkRunnable
        // requires mapView already assigned (see its own doc), so it can't
        // be started from here, before that property even exists yet.
    }

    // Blinks the live marker (alpha, not a rebuilt bitmap — see modeIcons'
    // own doc for why per-tick bitmap churn was a real bug) so it reads as
    // "this is live" the way a recording indicator or Google Maps' own
    // pulsing blue dot does, not just a static icon. Independent of the
    // 10Hz position-update tick — runs on its own slower cycle, and always
    // re-reads youAreHereMarker.icon fresh each cycle so it applies
    // correctly no matter which mode's (shared, cached) icon happens to be
    // showing at that moment. Declared after init{} on purpose — mapView
    // is assigned there, and an anonymous class capturing it (blinkRunnable
    // below) has to come after that in declaration order, or the compiler
    // can't prove mapView is assigned yet at the capture site.
    private val blinkIntervalMs = 550L
    private var blinkOn = true
    private val blinkHandler = Handler(Looper.getMainLooper())
    private val blinkRunnable: Runnable = object : Runnable {
        override fun run() {
            blinkOn = !blinkOn
            applyIcon()
            mapView.invalidate()
            blinkHandler.postDelayed(this, blinkIntervalMs)
        }
    }

    /**
     * Moves (creating if needed) the live "you are here" marker from a
     * raw GPS fix, before the accuracy pipeline (calibration -> fusion)
     * has produced anything to plot yet. Google Maps' own blue dot works
     * this way — it shows and tracks your real position immediately, not
     * only once some app-specific pipeline finishes.
     *
     * Two real bugs found live, both from the same root cause — nothing
     * ever touched the marker/camera before calibration finished, and
     * calibration (leveling, then yaw alignment) can take a while during
     * live GPS use, sometimes the entire session without a strong signal
     * or a confident straight-line stretch:
     *   1. The map view's camera never moved off its default center —
     *      (0, 0), Null Island, open ocean. OpenStreetMap's Mapnik style
     *      renders open ocean as a flat light-blue fill, so this looked
     *      exactly like "the map isn't loading". It wasn't a tile
     *      failure: the map was correctly rendering the middle of the
     *      Atlantic, because it was never told where the phone actually
     *      is yet.
     *   2. With no marker to move, there was also no live position
     *      indicator on screen at all while calibrating — and
     *      recenterOnLatest() (the FAB), which only ever acts on the
     *      marker's position, silently did nothing when tapped, because
     *      there was no marker yet to recenter on. Not a broken button;
     *      nothing for it to act on.
     * Masked during replay testing because replay's calibration always
     * completes in seconds (its GPS speed is always confidently above
     * the yaw-alignment threshold) — this is a live-GPS-specific gap.
     *
     * Once addPoint() takes over (real tracking has started), calls here
     * are harmless no-ops in practice — MainActivity stops making them —
     * but this still respects `following` either way so it can never
     * fight a live pan.
     */
    fun updateRoughLocation(lat: Double, lon: Double) {
        val p = GeoPoint(lat, lon)
        if (youAreHereMarker == null) {
            val m = Marker(mapView)
            m.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
            m.title = "You are here"
            mapView.overlays.add(m)
            youAreHereMarker = m
            mapView.controller.setCenter(p)
        }
        youAreHereMarker?.position = p
        // Blue — same color GNSS_TRACKING uses, and the same association
        // the legend already teaches ("GPS" = blue) — this marker IS a
        // raw GPS fix, just before the app's own pipeline has anything
        // fused to show instead.
        if (currentIconMode != FusionMode.GNSS_TRACKING) {
            currentIconMode = FusionMode.GNSS_TRACKING
            applyIcon()
        }
        if (following) mapView.controller.setCenter(p)
        mapView.invalidate()
    }

    private fun modeColor(mode: FusionMode): Int = when (mode) {
        FusionMode.GNSS_TRACKING -> Color.parseColor("#2563eb")
        FusionMode.BLACKOUT -> Color.parseColor("#dc2626")
        FusionMode.BLEND -> Color.parseColor("#f59e0b")
    }

    // alpha baked directly into the fill/stroke ARGB colors (not a
    // post-hoc Canvas composite over the bright bitmap — that path,
    // tried first, produced a bitmap that still rendered fully opaque
    // once handed to Marker/BitmapDrawable, confirmed live: logcat
    // proved the dim bitmap really was being assigned, but the on-screen
    // pixel still sampled as the full-opacity color). Same construction
    // path as the bright icon either way, just a translucent color in.
    private fun dotDrawable(colorHex: Int, sizeDp: Int = 14, alpha: Int = 255): GradientDrawable {
        val d = GradientDrawable()
        d.shape = GradientDrawable.OVAL
        d.setColor((colorHex and 0x00FFFFFF) or (alpha shl 24))
        d.setStroke(2, (Color.WHITE and 0x00FFFFFF) or (alpha shl 24))
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
        if (currentIconMode != mode) {
            currentIconMode = mode
            applyIcon()
        }
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
        currentIconMode = null
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
     * follows automatically every tick (see addPoint/updateRoughLocation),
     * but a user who's panned away needs an explicit way back.
     *
     * Instant (setCenter), not animated (animateTo) — "instantly... like
     * how it works on Google Maps" was explicitly requested. Works during
     * calibration too now that updateRoughLocation keeps a real marker
     * live from the first GPS fix onward, not just once real tracking
     * starts — see that method's doc for the bug where this used to
     * silently do nothing (no marker existed yet to recenter on). */
    fun recenterOnLatest() {
        following = true
        youAreHereMarker?.position?.let { mapView.controller.setCenter(it) }
    }

    private fun drawableToBitmap(d: GradientDrawable): android.graphics.Bitmap {
        val size = d.intrinsicWidth.coerceAtLeast(1)
        val bmp = android.graphics.Bitmap.createBitmap(size, size, android.graphics.Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(bmp)
        d.setBounds(0, 0, size, size)
        d.draw(canvas)
        return bmp
    }

    fun onResume() {
        mapView.onResume()
        blinkHandler.removeCallbacks(blinkRunnable)
        blinkHandler.postDelayed(blinkRunnable, blinkIntervalMs)
    }

    fun onPause() {
        mapView.onPause()
        blinkHandler.removeCallbacks(blinkRunnable)  // don't blink (or leak the handler) while backgrounded
    }
}
