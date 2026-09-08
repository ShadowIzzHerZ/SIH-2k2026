package com.sih26168.deadreckoning

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.switchmaterial.SwitchMaterial
import kotlin.math.roundToInt

/**
 * Live on-device demo of the whole pipeline this repo trains and
 * validates in Python: SensorReader (raw IMU) -> CalibrationManager
 * (Calibration.kt, matching src/calibration.py) -> FusionEngine (matching
 * the fixed src/fusion.py) -> BiasCorrectionModel (ONNX Runtime, the
 * checkpoints/dead_reckoning_model.onnx export) -> TrajectoryView.
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
    private lateinit var trajectoryView: TrajectoryView
    private lateinit var blackoutToggle: SwitchMaterial
    private lateinit var replayToggle: SwitchMaterial
    private var wasReplaying = false

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
        trajectoryView = findViewById(R.id.trajectoryView)
        blackoutToggle = findViewById(R.id.blackoutToggle)
        replayToggle = findViewById(R.id.replayToggle)

        sensorReader = SensorReader(this)
        locationReader = LocationReader(this)
        replayData = ReplayDataSource(this)
        model = BiasCorrectionModel(this)
        fusion = FusionEngine(model)
        mapMatcher = MapMatcher(RoadGraph(this))

        // blackoutToggle/replayToggle .isChecked read live in tick() — no listeners needed.

        val needed = arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
        if (needed.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            startPipeline()
        } else {
            requestPermissions.launch(needed)
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
            trajectoryView.clear()
            replayIndex = 0
            wasReplaying = usingReplay
        }

        val sample = readSample(usingReplay)

        if (!calibration.isLeveled) {
            calibration.addLevelSample(sample.rawAccel)
            modeText.text = getString(R.string.status_calibrating)
            detailText.text = "leveling…"
            tickHandler.postDelayed(::tick, tickIntervalMs)
            return
        }

        val demoBlackout = blackoutToggle.isChecked
        val available = sample.hasFix && !demoBlackout

        if (!calibration.isReady) {
            if (sample.hasFix) {
                calibration.addYawSample(sample.rawAccel, sample.speed, sample.bearingRad)
            }
            modeText.text = getString(R.string.status_calibrating)
            detailText.text = if (usingReplay) "yaw alignment: collecting confident samples from the replay…"
                else "yaw alignment: collecting confident samples — drive in a straight line with GPS"
            tickHandler.postDelayed(::tick, tickIntervalMs)
            return
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
        trajectoryView.addPoint(s.x, s.y, s.mode)

        // Map-match the fused position onto the bundled road graph — same
        // idea as src/evaluate_with_mapmatching.py's before/after
        // comparison, live here instead of an offline batch report. Most
        // useful (and most likely to actually find a nearby road) during
        // BLACKOUT/BLEND, where the raw fused estimate can drift off the
        // true road; matching every mode too so the green overlay is a
        // continuous, comparable trail rather than appearing/disappearing.
        fusion.localXYToLatLon(s.x, s.y)?.let { (lat, lon) ->
            mapMatcher.match(lat, lon)?.let { m ->
                fusion.latLonToLocalXY(m.lat, m.lon)?.let { xy ->
                    trajectoryView.addMatchedPoint(xy[0], xy[1])
                }
            }
        }

        modeText.text = when (s.mode) {
            FusionMode.GNSS_TRACKING -> getString(R.string.status_gnss)
            FusionMode.BLACKOUT -> getString(R.string.status_blackout)
            FusionMode.BLEND -> getString(R.string.status_blend)
        }
        detailText.text = "speed: ${"%.1f".format(s.speed)} m/s  |  heading: ${Math.toDegrees(s.heading.toDouble()).roundToInt()}°" +
            (if (usingReplay) "  |  [REPLAY]" else "") +
            (if (demoBlackout) "  |  [SIMULATED BLACKOUT]" else "")

        tickHandler.postDelayed(::tick, tickIntervalMs)
    }

    override fun onDestroy() {
        super.onDestroy()
        ticking = false
        tickHandler.removeCallbacksAndMessages(null)
        sensorReader.stop()
        locationReader.stop()
        model.close()
    }
}
