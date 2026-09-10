package com.sih26168.deadreckoning

import android.Manifest
import android.animation.ValueAnimator
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.switchmaterial.SwitchMaterial
import kotlin.math.roundToInt

/**
 * Live on-device demo of the whole pipeline this repo trains and
 * validates in Python: SensorReader (raw IMU) -> CalibrationManager
 * (Calibration.kt, matching src/calibration.py) -> FusionEngine (matching
 * the fixed src/fusion.py) -> BiasCorrectionModel (ONNX Runtime, the
 * checkpoints/dead_reckoning_model.onnx export) -> RoadMapView (a real
 * OpenStreetMap-tile map, same free-tile approach as the Zen/DevStorm-2026
 * project's Leaflet map).
 *
 * Two demo controls, orthogonal to each other:
 *   - "Replay real recorded drive" swaps the data source from live
 *     sensors/GPS to a real, previously-recorded drive bundled as an asset
 *     (see ReplayDataSource) — for demoing indoors, with no GPS reception
 *     and no room to actually drive, using real data instead of idealized
 *     synthetic motion. demoSelector picks which of two: both are genuine
 *     DevRecorder captures from live testing in Jalandhar, Punjab
 *     (data/own_recordings/openroute_20260909_{1908,1821}.csv), not
 *     comma2k19 — so the demo route, and the road-matching overlay against
 *     the bundled road_graph.json, are both real for wherever this app is
 *     actually being shown.
 *   - "Simulate GNSS blackout" forces the fusion engine into BLACKOUT
 *     regardless of data source — the same thing src/simulate_blackout.py
 *     does offline, live here on whichever source is active.
 *
 * A third, hidden control (long-press the title) reveals Developer mode —
 * DevRecorder logging real sensor+GPS samples to a CSV your team can
 * actually retrain the model with afterward (see DevRecorder.kt's
 * docstring for why this is data collection, not on-device training).
 * Hidden by default so it's not something a judge stumbles into.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var sensorReader: SensorReader
    private lateinit var locationReader: LocationReader
    // Two real recorded drives to choose from (see demoSelector) — both
    // genuine DevRecorder captures from live testing in Jalandhar, Punjab
    // (data/own_recordings/openroute_20260909_{1908,1821}.csv), not
    // comma2k19/synthetic data. Both loaded upfront so switching between
    // them mid-session (tick()) is instant, no asset I/O on the hot path.
    private lateinit var replayDataDemo1: ReplayDataSource
    private lateinit var replayDataDemo2: ReplayDataSource
    private var replayIndex = 0

    private var calibration = CalibrationManager()
    private lateinit var model: BiasCorrectionModel
    private lateinit var fusion: FusionEngine
    private lateinit var mapMatcher: MapMatcher

    private lateinit var modeText: TextView
    private lateinit var detailText: TextView
    private lateinit var statusDot: View
    private lateinit var calibrationProgress: ProgressBar
    private lateinit var legendRow: View
    private lateinit var roadMapView: RoadMapView
    private lateinit var blackoutToggle: SwitchMaterial
    private lateinit var replayToggle: SwitchMaterial
    private lateinit var demoSelector: RadioGroup
    private lateinit var recenterButton: FloatingActionButton
    private lateinit var titleText: TextView
    private lateinit var devSection: View
    private lateinit var devRecordToggle: SwitchMaterial
    private lateinit var devRecordStatus: TextView
    private lateinit var devShareButton: Button
    private lateinit var devSkipCalibrationButton: Button
    private lateinit var devRecorder: DevRecorder
    private lateinit var prefs: SharedPreferences
    private var wasReplaying = false
    private var wasDemo = 1

    private val calibratingColor = android.graphics.Color.parseColor("#94a3b8")  // neutral gray

    private val tickHandler = Handler(Looper.getMainLooper())
    private val tickIntervalMs = 100L  // 10Hz — matches configs/default.yaml's sample_rate_hz
    private var ticking = false

    private val requestPermissions = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.RequestMultiplePermissions()
    ) { granted ->
        if (granted.values.all { it }) startPipeline()
        else modeText.text = "Location permission required"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        modeText = findViewById(R.id.modeText)
        detailText = findViewById(R.id.detailText)
        statusDot = findViewById(R.id.statusDot)
        calibrationProgress = findViewById(R.id.calibrationProgress)
        legendRow = findViewById(R.id.legendRow)
        roadMapView = findViewById(R.id.roadMapView)
        blackoutToggle = findViewById(R.id.blackoutToggle)
        replayToggle = findViewById(R.id.replayToggle)
        demoSelector = findViewById(R.id.demoSelector)
        recenterButton = findViewById(R.id.recenterButton)
        recenterButton.setOnClickListener { roadMapView.recenterOnLatest() }
        titleText = findViewById(R.id.titleText)
        devSection = findViewById(R.id.devSection)
        devRecordToggle = findViewById(R.id.devRecordToggle)
        devRecordStatus = findViewById(R.id.devRecordStatus)
        devShareButton = findViewById(R.id.devShareButton)
        devSkipCalibrationButton = findViewById(R.id.devSkipCalibrationButton)

        sensorReader = SensorReader(this)
        locationReader = LocationReader(this)
        replayDataDemo1 = ReplayDataSource(this, "replay_demo1.json")
        replayDataDemo2 = ReplayDataSource(this, "replay_demo2.json")
        model = BiasCorrectionModel(this)
        fusion = FusionEngine(model)
        mapMatcher = MapMatcher(RoadGraph(this))
        devRecorder = DevRecorder(this)

        // blackoutToggle/replayToggle/demoSelector .isChecked read live in
        // tick() — only replayToggle needs a listener, purely to show/hide
        // the demo picker (which demo to read stays live-read in tick()
        // like the other toggles).
        replayToggle.setOnCheckedChangeListener { _, checked ->
            demoSelector.visibility = if (checked) View.VISIBLE else View.GONE
        }

        prefs = getSharedPreferences("dev_prefs", MODE_PRIVATE)
        devSection.visibility = if (prefs.getBoolean("dev_mode_visible", false)) View.VISIBLE else View.GONE
        titleText.setOnLongClickListener {
            val nowVisible = devSection.visibility != View.VISIBLE
            devSection.visibility = if (nowVisible) View.VISIBLE else View.GONE
            prefs.edit().putBoolean("dev_mode_visible", nowVisible).apply()
            Toast.makeText(
                this,
                getString(if (nowVisible) R.string.dev_mode_on_toast else R.string.dev_mode_off_toast),
                Toast.LENGTH_SHORT,
            ).show()
            true
        }
        setupDevRecording()

        val needed = arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
        if (needed.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            startPipeline()
        } else {
            requestPermissions.launch(needed)
        }
    }

    /** Wires the dev-mode Record/Share controls — kept separate from the
     * main tick() logic so the always-on pipeline isn't cluttered with a
     * feature most runs never touch. */
    private fun setupDevRecording() {
        devRecordToggle.setOnCheckedChangeListener { _, checked ->
            if (checked) {
                if (replayToggle.isChecked) {
                    Toast.makeText(this, R.string.dev_record_blocked_replay, Toast.LENGTH_LONG).show()
                    devRecordToggle.isChecked = false
                    return@setOnCheckedChangeListener
                }
                devShareButton.visibility = View.GONE
                val file = devRecorder.start("openroute")
                devRecordStatus.text = "Recording to ${file.name}…"
            } else {
                val file = devRecorder.stop()
                devRecordStatus.text = if (file != null) {
                    "Saved ${file.name} — ${devRecorder.sampleCount} samples"
                } else ""
                devShareButton.visibility = if (file != null) View.VISIBLE else View.GONE
            }
        }
        devShareButton.setOnClickListener {
            val file = devRecorder.currentFile ?: return@setOnClickListener
            val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "text/csv"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(intent, file.name))
        }
        devSkipCalibrationButton.setOnClickListener {
            // Testing-only shortcut past the real (deliberately slow, see
            // CalibrationManager's class doc) calibration wait — the next
            // scheduled tick() picks up isReady/isLeveled on its own, no
            // extra kick needed.
            calibration.skipForTesting()
            Toast.makeText(this, "Calibration skipped (testing) — not a real estimate", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startPipeline() {
        if (!sensorReader.available) {
            modeText.text = "Missing accelerometer/gyroscope"
            return
        }
        sensorReader.start()
        locationReader.start()
        ticking = true
        tickHandler.postDelayed(::tick, tickIntervalMs)
    }

    /** One sample, regardless of source — live sensors/GPS or the bundled
     * replay. hasFix gates fusion (needs a real speed/bearing to feed
     * GNSS_TRACKING); hasRoughFix is the weaker "do we know roughly where
     * the phone is" check the map marker uses — see
     * LocationReader.hasAnyRecentFix's doc for why these need to differ. */
    private data class Sample(
        val rawAccel: FloatArray, val rawGyro: FloatArray,
        val hasFix: Boolean, val hasRoughFix: Boolean,
        val lat: Double, val lon: Double, val speed: Float, val bearingRad: Float,
    )

    private fun readSample(usingReplay: Boolean, demo: Int): Sample {
        if (usingReplay) {
            val replayData = if (demo == 2) replayDataDemo2 else replayDataDemo1
            val row = replayData.rows[replayIndex]
            replayIndex = (replayIndex + 1) % replayData.rows.size
            return Sample(row.accel, row.gyro, true, true, row.lat, row.lon, row.speed, row.bearingRad)
        }
        val loc = locationReader.lastLocation
        val hasFix = locationReader.hasRecentFix() && loc != null
        val hasRoughFix = locationReader.hasAnyRecentFix() && loc != null
        return Sample(
            sensorReader.lastAccel, sensorReader.lastGyro, hasFix, hasRoughFix,
            loc?.latitude ?: 0.0, loc?.longitude ?: 0.0, loc?.speed ?: 0f,
            // loc.bearing is Android's compass bearing (0=N/90=E, clockwise) —
            // needs the compass->math axis swap, not just Math.toRadians().
            // See Calibration.compassDegToMathRad's doc for the real bug a
            // plain toRadians() here used to cause (blackout heading off in
            // a very different direction from the real one).
            if (loc != null) Calibration.compassDegToMathRad(loc.bearing) else 0f,
        )
    }

    private fun tick() {
        if (!ticking) return

        val usingReplay = replayToggle.isChecked
        val demo = if (demoSelector.checkedRadioButtonId == R.id.demo2Radio) 2 else 1
        if (usingReplay != wasReplaying || (usingReplay && demo != wasDemo)) {
            // Switching data source mid-flight would mix live-phone leveling
            // with a replayed drive's motion scale (or vice versa), and
            // switching demos mid-flight would splice two different real
            // routes together — restart calibration/fusion/trajectory
            // cleanly for the new source instead of producing a
            // nonsensical blend.
            calibration = CalibrationManager()
            fusion.reset()
            mapMatcher.reset()
            roadMapView.clear()
            replayIndex = 0
            wasReplaying = usingReplay
            wasDemo = demo
            calibrationProgress.visibility = View.VISIBLE
            calibrationProgress.progress = 0
            legendRow.visibility = View.GONE
        }

        val sample = readSample(usingReplay, demo)

        // Keep a real, live "you are here" marker moving from the very
        // first GPS fix — calibration (below) can take a while, sometimes
        // the entire session, during live GPS with a weak signal, and
        // nothing else moves the map's marker/camera until it finishes.
        // See RoadMapView.updateRoughLocation's doc for the two real bugs
        // this fixes (the map sitting at Null Island looking exactly like
        // a tile-loading failure, and the recenter button silently doing
        // nothing because there was no marker yet to recenter on).
        //
        // hasRoughFix, not hasFix — a real bug found live: hasFix requires
        // hasSpeed()/hasBearing(), which a lot of Android GPS chips leave
        // unset while genuinely stationary (exactly the leveling step
        // right below, which tells the user to hold the phone still), so
        // the marker was invisible for the entire leveling phase despite
        // the phone's location already being known. See
        // LocationReader.hasAnyRecentFix's doc for the full reasoning.
        if (sample.hasRoughFix) roadMapView.updateRoughLocation(sample.lat, sample.lon)

        if (devRecorder.isRecording) {
            if (usingReplay) {
                // Replay got switched on mid-recording — stop rather than
                // silently log replay data as if it were a new real drive.
                devRecordToggle.isChecked = false // triggers stop() via the listener
            } else {
                devRecorder.logSample(
                    sample.rawAccel, sample.rawGyro, sample.hasFix,
                    sample.lat, sample.lon, sample.speed,
                    // sample.bearingRad is this app's internal (math, 0=east
                    // ccw+) convention — heading_gt needs real compass
                    // degrees, same convention src/data/io_vnbd_loader.py
                    // and Calibration.kt's own yaw estimation expect. Plain
                    // Math.toDegrees() here would silently write a math-
                    // convention angle into a column downstream code reads
                    // as compass bearing.
                    Calibration.mathRadToCompassDeg(sample.bearingRad),
                )
                devRecordStatus.text = "Recording… ${devRecorder.sampleCount} samples" +
                    (if (!sample.hasFix) "  [no GPS fix]" else "")
            }
        }

        if (!calibration.isLeveled) {
            calibration.addLevelSample(sample.rawAccel)
            // Distinguish "still collecting" from "collected, but you're
            // not actually holding it still" — otherwise an unchanging
            // "leveling…" looks like a hang when it's really waiting on
            // the user (see CalibrationManager's stationarity gate).
            showCalibrating(
                getString(
                    if (calibration.levelWaitingForStillness) R.string.calib_leveling_waiting
                    else R.string.calib_leveling
                )
            )
            tickHandler.postDelayed(::tick, tickIntervalMs)
            return
        }

        val demoBlackout = blackoutToggle.isChecked
        val available = sample.hasFix && !demoBlackout

        if (!calibration.isReady) {
            if (sample.hasFix) {
                calibration.addYawSample(sample.rawAccel, sample.speed, sample.bearingRad)
            }
            showCalibrating(
                getString(
                    if (usingReplay) R.string.calib_yaw_replay else R.string.calib_yaw_live
                )
            )
            tickHandler.postDelayed(::tick, tickIntervalMs)
            return
        }

        if (legendRow.visibility != View.VISIBLE) {
            // First tick past calibration — reveal the legend, hide the
            // progress bar, once, rather than every tick. Faded in rather
            // than just flipped to VISIBLE — this is the exact moment
            // calibration hands off to real tracking, worth reading as a
            // deliberate transition rather than a layout pop.
            calibrationProgress.visibility = View.GONE
            legendRow.alpha = 0f
            legendRow.visibility = View.VISIBLE
            legendRow.animate().alpha(1f).setDuration(300L).start()
        }

        val calAccel = calibration.calibrate(sample.rawAccel)
        val calGyro = calibration.calibrate(sample.rawGyro)

        fusion.tick(
            calibratedAccel = calAccel,
            calibratedGyro = calGyro,
            available = available,
            lat = sample.lat,
            lon = sample.lon,
            gnssSpeed = sample.speed,
            gnssHeadingRad = sample.bearingRad,
        )

        val s = fusion.state
        // RoadMapView draws in real lat/lon, not FusionEngine's local xy —
        // convert once per tick using the engine's own reference fix.
        fusion.localXYToLatLon(s.x, s.y)?.let { (lat, lon) ->
            roadMapView.addPoint(lat, lon, s.mode)

            // Map-match the fused position onto the bundled road graph —
            // same idea as src/evaluate_with_mapmatching.py's before/after
            // comparison, live here instead of an offline batch report.
            mapMatcher.match(lat, lon, s.heading)?.let { m ->
                roadMapView.addMatchedPoint(m.lat, m.lon)
            }
        }

        val (statusText, dotColor) = when (s.mode) {
            FusionMode.GNSS_TRACKING -> getString(R.string.status_gnss) to 0xFF2563EB.toInt()
            FusionMode.BLACKOUT -> getString(R.string.status_blackout) to 0xFFDC2626.toInt()
            FusionMode.BLEND -> getString(R.string.status_blend) to 0xFFF59E0B.toInt()
        }
        modeText.text = statusText
        setStatusDotColor(dotColor)
        // s.heading is this app's internal (math, 0=east ccw+) convention —
        // show a real compass heading (0=N/90=E) to the user, not a plain
        // toDegrees() of the math-convention value.
        detailText.text = "speed: ${"%.0f".format(s.speed * 3.6f)} km/h  |  heading: ${Calibration.mathRadToCompassDeg(s.heading).roundToInt()}°" +
            (if (usingReplay) "  |  [REPLAY demo $demo]" else "") +
            (if (demoBlackout) "  |  [SIMULATED BLACKOUT]" else "") +
            // Everything downstream inherits a bad calibration, so say so
            // rather than presenting a degraded fix as an equal one.
            (if (calibration.isRoughCalibration) "  |  ⚠ ${getString(R.string.calib_rough)}" else "")

        tickHandler.postDelayed(::tick, tickIntervalMs)
    }

    private fun showCalibrating(detail: String) {
        modeText.text = getString(R.string.status_calibrating)
        detailText.text = detail
        setStatusDotColor(calibratingColor)
        // animate=true smoothly tweens toward the new value instead of a
        // jump-cut fill — noticeable given this gets called every 100ms
        // tick while collecting yaw samples, where the raw percent can
        // sit still for a while then jump a few points at once.
        calibrationProgress.setProgress(calibration.progressPercent, true)
    }

    // Tracks the dot's current color so setStatusDotColor can animate
    // *from* it — GradientDrawable itself has no "current color" getter.
    private var currentDotColor: Int = calibratingColor

    private fun setStatusDotColor(color: Int) {
        if (color == currentDotColor) return
        val drawable = statusDot.background as? GradientDrawable ?: return
        // Real mode changes (GNSS<->blackout<->blend) are the whole point
        // of this app to notice — an instant color-snap is easy to miss in
        // a glance at the corner of the screen; a quick tween draws the
        // eye the way the blinking map marker (RoadMapView) does.
        ValueAnimator.ofArgb(currentDotColor, color).apply {
            duration = 250L
            addUpdateListener { drawable.setColor(it.animatedValue as Int) }
            start()
        }
        currentDotColor = color
    }

    override fun onResume() {
        super.onResume()
        roadMapView.onResume()  // osmdroid tile cache lifecycle
    }

    override fun onPause() {
        super.onPause()
        roadMapView.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        ticking = false
        tickHandler.removeCallbacksAndMessages(null)
        sensorReader.stop()
        locationReader.stop()
        model.close()
        devRecorder.stop()  // flush any in-progress recording rather than losing the tail
    }
}
