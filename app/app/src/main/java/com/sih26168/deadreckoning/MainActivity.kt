package com.sih26168.deadreckoning

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
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
 * "Simulate GNSS blackout" exists because a live demo can't reliably walk
 * into a real tunnel on stage on cue — it forces the fusion engine into
 * BLACKOUT the same way src/simulate_blackout.py does offline, on real
 * live sensor data instead of a recorded trace.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var sensorReader: SensorReader
    private lateinit var locationReader: LocationReader
    private lateinit var calibration: CalibrationManager
    private lateinit var model: BiasCorrectionModel
    private lateinit var fusion: FusionEngine

    private lateinit var modeText: TextView
    private lateinit var detailText: TextView
    private lateinit var trajectoryView: TrajectoryView
    private lateinit var blackoutToggle: SwitchMaterial

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

        sensorReader = SensorReader(this)
        locationReader = LocationReader(this)
        calibration = CalibrationManager()
        model = BiasCorrectionModel(this)
        fusion = FusionEngine(model)

        // blackoutToggle.isChecked is read live in tick() — no listener needed here.

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

    private fun tick() {
        if (!ticking) return

        val rawAccel = sensorReader.lastAccel
        val rawGyro = sensorReader.lastGyro

        if (!calibration.isLeveled) {
            calibration.addLevelSample(rawAccel)
            modeText.text = getString(R.string.status_calibrating)
            detailText.text = "leveling…"
            tickHandler.postDelayed(::tick, tickIntervalMs)
            return
        }

        val loc = locationReader.lastLocation
        val demoBlackout = blackoutToggle.isChecked
        val realFix = locationReader.hasRecentFix() && loc != null
        val available = realFix && !demoBlackout

        if (!calibration.isReady) {
            if (realFix) {
                calibration.addYawSample(rawAccel, loc!!.speed, Math.toRadians(loc.bearing.toDouble()).toFloat())
            }
            modeText.text = getString(R.string.status_calibrating)
            detailText.text = "yaw alignment: collecting confident samples — drive in a straight line with GPS"
            tickHandler.postDelayed(::tick, tickIntervalMs)
            return
        }

        val calAccel = calibration.calibrate(rawAccel)
        val calGyro = calibration.calibrate(rawGyro)

        fusion.tick(
            calibratedAccel = calAccel,
            calibratedGyro = calGyro,
            available = available,
            lat = loc?.latitude ?: 0.0,
            lon = loc?.longitude ?: 0.0,
            gnssSpeed = loc?.speed ?: 0f,
            gnssHeadingRad = if (loc != null) Math.toRadians(loc.bearing.toDouble()).toFloat() else 0f,
        )

        val s = fusion.state
        trajectoryView.addPoint(s.x, s.y, s.mode)
        modeText.text = when (s.mode) {
            FusionMode.GNSS_TRACKING -> getString(R.string.status_gnss)
            FusionMode.BLACKOUT -> getString(R.string.status_blackout)
            FusionMode.BLEND -> getString(R.string.status_blend)
        }
        detailText.text = "speed: ${"%.1f".format(s.speed)} m/s  |  heading: ${Math.toDegrees(s.heading.toDouble()).roundToInt()}°" +
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
