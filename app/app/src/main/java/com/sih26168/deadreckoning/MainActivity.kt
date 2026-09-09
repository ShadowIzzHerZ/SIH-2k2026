package com.sih26168.deadreckoning

import android.Manifest
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
 *     sensors/GPS to a real, previously-validated comma2k19 segment
 *     bundled as an asset (see ReplayDataSource) — for demoing indoors,
 *     with no GPS reception and no room to actually drive, using real
 *     data instead of idealized synthetic motion.
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
    private lateinit var replayData: ReplayDataSource
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
    private lateinit var recenterButton: FloatingActionButton
    private lateinit var titleText: TextView
    private lateinit var devSection: View
    private lateinit var devRecordToggle: SwitchMaterial
    private lateinit var devRecordStatus: TextView
    private lateinit var devShareButton: Button
    private lateinit var devRecorder: DevRecorder
    private lateinit var prefs: SharedPreferences
    private var wasReplaying = false

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
        recenterButton = findViewById(R.id.recenterButton)
        recenterButton.setOnClickListener { roadMapView.recenterOnLatest() }
        titleText = findViewById(R.id.titleText)
        devSection = findViewById(R.id.devSection)
        devRecordToggle = findViewById(R.id.devRecordToggle)
        devRecordStatus = findViewById(R.id.devRecordStatus)
        devShareButton = findViewById(R.id.devShareButton)

        sensorReader = SensorReader(this)
        locationReader = LocationReader(this)
        replayData = ReplayDataSource(this)
        model = BiasCorrectionModel(this)
        fusion = FusionEngine(model)
        mapMatcher = MapMatcher(RoadGraph(this))
        devRecorder = DevRecorder(this)

        // blackoutToggle/replayToggle .isChecked read live in tick() — no listeners needed.

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

    /** One sample, regardless of source — live sensors/GPS or the bundled replay. */
    private data class Sample(
        val rawAccel: FloatArray, val rawGyro: FloatArray,
        val hasFix: Boolean, val lat: Double, val lon: Double, val speed: Float, val bearingRad: Float,
    )

    private fun readSample(usingReplay: Boolean): Sample {
        if (usingReplay) {
            val row = replayData.rows[replayIndex]
            replayIndex = (replayIndex + 1) % replayData.rows.size
            return Sample(row.accel, row.gyro, true, row.lat, row.lon, row.speed, row.bearingRad)
        }
        val loc = locationReader.lastLocation
        val hasFix = locationReader.hasRecentFix() && loc != null
        return Sample(
            sensorReader.lastAccel, sensorReader.lastGyro, hasFix,
            loc?.latitude ?: 0.0, loc?.longitude ?: 0.0, loc?.speed ?: 0f,
            if (loc != null) Math.toRadians(loc.bearing.toDouble()).toFloat() else 0f,
        )
    }

    private fun tick() {
        if (!ticking) return

        val usingReplay = replayToggle.isChecked
        if (usingReplay != wasReplaying) {
            // Switching data source mid-flight would mix live-phone leveling
            // with a replayed car's motion scale (or vice versa) — restart
            // calibration/fusion/trajectory cleanly for the new source
            // instead of producing a nonsensical blend of the two.
            calibration = CalibrationManager()
            fusion.reset()
            mapMatcher.reset()
            roadMapView.clear()
            replayIndex = 0
            wasReplaying = usingReplay
            calibrationProgress.visibility = View.VISIBLE
            calibrationProgress.progress = 0
            legendRow.visibility = View.GONE
        }

        val sample = readSample(usingReplay)

        if (devRecorder.isRecording) {
            if (usingReplay) {
                // Replay got switched on mid-recording — stop rather than
                // silently log replay data as if it were a new real drive.
                devRecordToggle.isChecked = false // triggers stop() via the listener
            } else {
                devRecorder.logSample(
                    sample.rawAccel, sample.rawGyro, sample.hasFix,
                    sample.lat, sample.lon, sample.speed,
                    Math.toDegrees(sample.bearingRad.toDouble()).toFloat(),
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
            // progress bar, once, rather than every tick.
            calibrationProgress.visibility = View.GONE
            legendRow.visibility = View.VISIBLE
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
        detailText.text = "speed: ${"%.0f".format(s.speed * 3.6f)} km/h  |  heading: ${Math.toDegrees(s.heading.toDouble()).roundToInt()}°" +
            (if (usingReplay) "  |  [REPLAY]" else "") +
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
        calibrationProgress.progress = calibration.progressPercent
    }

    private fun setStatusDotColor(color: Int) {
        (statusDot.background as? GradientDrawable)?.setColor(color)
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
