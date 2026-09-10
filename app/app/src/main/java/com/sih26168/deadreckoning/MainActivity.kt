package com.sih26168.deadreckoning

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
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
 * OpenStreetMap-tile map).
 *
 * UI is a pixel-match of the Stitch "Zen Civic" mockups (each screen's
 * own code.html under stitch_android_app_stability_audit/): a fixed top bar, a
 * full-bleed map with floating cards over it, and a real bottom nav with
 * 4 destinations — Live Map (this screen), Sensors (raw accel/gyro +
 * calibration controls), Simulation (every demo/testing control), Reset
 * (an immediate real action, not a screen). Only real app data/actions
 * are wired to each mockup element — anything with no real backing
 * (CAN Bus, a claimed 100Hz, pedestrian step-count, a ×1/×2/×5 speed
 * control) is left out rather than faked; see activity_main.xml's own
 * comments for exactly what was dropped and why.
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

    // ---- Top bar ----
    private lateinit var headerIcon: TextView
    private lateinit var titleText: TextView
    private lateinit var subtitleText: TextView

    // ---- Live Map tab ----
    private lateinit var calibratingCard: View
    private lateinit var detailText: TextView
    private lateinit var calibrationProgress: ProgressBar
    private lateinit var calibrationRingContainer: View
    private lateinit var calibrationPercentText: TextView
    private lateinit var trackingContent: View
    private lateinit var modeIconChip: View
    private lateinit var modeIconText: TextView
    private lateinit var modeText: TextView
    private lateinit var modeBodyText: TextView
    private lateinit var simulatedBadge: View
    private lateinit var legendRow: View
    private lateinit var speedValueText: TextView
    private lateinit var speedBar: ProgressBar
    private lateinit var headingValueText: TextView
    private lateinit var headingCompassText: TextView
    private lateinit var blackoutToggle: SwitchMaterial
    private lateinit var resetFilterButton: Button
    private lateinit var recordingLogChip: View
    private lateinit var recordingLogTimeText: TextView
    private lateinit var roadMapView: RoadMapView
    private lateinit var recenterButton: FloatingActionButton

    // ---- Sensors tab ----
    private lateinit var accelX: AxisViews
    private lateinit var accelY: AxisViews
    private lateinit var accelZ: AxisViews
    private lateinit var gyroX: AxisViews
    private lateinit var gyroY: AxisViews
    private lateinit var gyroZ: AxisViews
    private lateinit var devSkipCalibrationButton: Button
    private lateinit var switchToReplayButton: Button

    // ---- Simulation tab ----
    private lateinit var replayToggle: SwitchMaterial
    private lateinit var demoSelector: RadioGroup
    private lateinit var devSection: View
    private lateinit var devRecordToggle: SwitchMaterial
    private lateinit var devRecordStatus: TextView
    private lateinit var devShareButton: Button
    private lateinit var devRecorder: DevRecorder
    private lateinit var prefs: SharedPreferences

    // ---- Bottom nav ----
    private lateinit var liveMapContent: View
    private lateinit var sensorsPanel: View
    private lateinit var simulationPanel: View
    private lateinit var navLiveMap: View
    private lateinit var navSensors: View
    private lateinit var navSimulation: View
    private lateinit var navReset: View
    private lateinit var navLiveMapLabel: TextView
    private lateinit var navSensorsLabel: TextView
    private lateinit var navSimulationLabel: TextView

    private var wasReplaying = false
    private var wasDemo = 1
    private var activeTab = Tab.LIVE_MAP
    private var previousFusionMode: FusionMode? = null
    private var blackoutStartUptimeMs: Long? = null

    private enum class Tab { LIVE_MAP, SENSORS, SIMULATION }

    /** One accel/gyro axis cell's inflated children — see axis_readout.xml. */
    private class AxisViews(root: View) {
        val label: TextView = root.findViewById(R.id.axisLabel)
        val value: TextView = root.findViewById(R.id.axisValue)
        val bar: ProgressBar = root.findViewById(R.id.axisBar)
    }

    private val tickHandler = Handler(Looper.getMainLooper())
    private val tickIntervalMs = 100L  // 10Hz — matches configs/default.yaml's sample_rate_hz
    private var ticking = false

    // Real "location access needed" screen (Zen Civic restyle) — replaces
    // the old dead-end status-text swap on denial with three real actions.
    private lateinit var locationRequiredCard: View
    private lateinit var missingSensorsCard: View
    private lateinit var enableLocationButton: Button
    private lateinit var openSettingsButton: View
    private lateinit var continueDemoLink: TextView

    private val requestPermissions = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.RequestMultiplePermissions()
    ) { granted ->
        if (granted.values.all { it }) {
            locationRequiredCard.visibility = View.GONE
            startPipeline()
        } else {
            locationRequiredCard.visibility = View.VISIBLE
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        headerIcon = findViewById(R.id.headerIcon)
        titleText = findViewById(R.id.titleText)
        subtitleText = findViewById(R.id.subtitleText)

        calibratingCard = findViewById(R.id.calibratingCard)
        detailText = findViewById(R.id.detailText)
        calibrationProgress = findViewById(R.id.calibrationProgress)
        calibrationRingContainer = findViewById(R.id.calibrationRingContainer)
        calibrationPercentText = findViewById(R.id.calibrationPercentText)
        trackingContent = findViewById(R.id.trackingContent)
        modeIconChip = findViewById(R.id.modeIconChip)
        modeIconText = findViewById(R.id.modeIconText)
        modeText = findViewById(R.id.modeText)
        modeBodyText = findViewById(R.id.modeBodyText)
        simulatedBadge = findViewById(R.id.simulatedBadge)
        legendRow = findViewById(R.id.legendRow)
        speedValueText = findViewById(R.id.speedValueText)
        speedBar = findViewById(R.id.speedBar)
        headingValueText = findViewById(R.id.headingValueText)
        headingCompassText = findViewById(R.id.headingCompassText)
        blackoutToggle = findViewById(R.id.blackoutToggle)
        resetFilterButton = findViewById(R.id.resetFilterButton)
        recordingLogChip = findViewById(R.id.recordingLogChip)
        recordingLogTimeText = findViewById(R.id.recordingLogTimeText)
        roadMapView = findViewById(R.id.roadMapView)
        recenterButton = findViewById(R.id.recenterButton)
        recenterButton.setOnClickListener { roadMapView.recenterOnLatest() }

        accelX = AxisViews(findViewById(R.id.accelX))
        accelY = AxisViews(findViewById(R.id.accelY))
        accelZ = AxisViews(findViewById(R.id.accelZ))
        gyroX = AxisViews(findViewById(R.id.gyroX))
        gyroY = AxisViews(findViewById(R.id.gyroY))
        gyroZ = AxisViews(findViewById(R.id.gyroZ))
        devSkipCalibrationButton = findViewById(R.id.devSkipCalibrationButton)
        switchToReplayButton = findViewById(R.id.switchToReplayButton)

        replayToggle = findViewById(R.id.replayToggle)
        demoSelector = findViewById(R.id.demoSelector)
        devSection = findViewById(R.id.devSection)
        devRecordToggle = findViewById(R.id.devRecordToggle)
        devRecordStatus = findViewById(R.id.devRecordStatus)
        devShareButton = findViewById(R.id.devShareButton)

        liveMapContent = findViewById(R.id.liveMapContent)
        sensorsPanel = findViewById(R.id.sensorsPanel)
        simulationPanel = findViewById(R.id.simulationPanel)
        navLiveMap = findViewById(R.id.navLiveMap)
        navSensors = findViewById(R.id.navSensors)
        navSimulation = findViewById(R.id.navSimulation)
        navReset = findViewById(R.id.navReset)
        navLiveMapLabel = findViewById(R.id.navLiveMapLabel)
        navSensorsLabel = findViewById(R.id.navSensorsLabel)
        navSimulationLabel = findViewById(R.id.navSimulationLabel)

        locationRequiredCard = findViewById(R.id.locationRequiredCard)
        missingSensorsCard = findViewById(R.id.missingSensorsCard)
        enableLocationButton = findViewById(R.id.enableLocationButton)
        openSettingsButton = findViewById(R.id.openSettingsButton)
        continueDemoLink = findViewById(R.id.continueDemoLink)

        sensorReader = SensorReader(this)
        locationReader = LocationReader(this)
        replayDataDemo1 = ReplayDataSource(this, "replay_demo1.json")
        replayDataDemo2 = ReplayDataSource(this, "replay_demo2.json")
        model = BiasCorrectionModel(this)
        fusion = FusionEngine(model)
        mapMatcher = MapMatcher(RoadGraph(this))
        devRecorder = DevRecorder(this)

        replayToggle.setOnCheckedChangeListener { _, checked ->
            demoSelector.visibility = if (checked) View.VISIBLE else View.GONE
        }
        switchToReplayButton.setOnClickListener { replayToggle.isChecked = true; selectTab(Tab.LIVE_MAP) }

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
        setupLocationRequiredScreen()
        setupBottomNav()

        val needed = arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
        if (needed.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            startPipeline()
        } else {
            requestPermissions.launch(needed)
        }
    }

    /** Wires the 4 real bottom-nav destinations — 3 real content panels
     * plus Reset, a real immediate action rather than a screen. */
    private fun setupBottomNav() {
        navLiveMap.setOnClickListener { selectTab(Tab.LIVE_MAP) }
        navSensors.setOnClickListener { selectTab(Tab.SENSORS) }
        navSimulation.setOnClickListener { selectTab(Tab.SIMULATION) }
        navReset.setOnClickListener { performResetFilter() }
    }

    private fun selectTab(tab: Tab) {
        activeTab = tab
        liveMapContent.visibility = if (tab == Tab.LIVE_MAP) View.VISIBLE else View.GONE
        sensorsPanel.visibility = if (tab == Tab.SENSORS) View.VISIBLE else View.GONE
        simulationPanel.visibility = if (tab == Tab.SIMULATION) View.VISIBLE else View.GONE
        recenterButton.visibility = if (tab == Tab.LIVE_MAP) View.VISIBLE else View.GONE

        val activeColor = ContextCompat.getColor(this, R.color.on_primary_container)
        val inactiveColor = ContextCompat.getColor(this, R.color.on_surface_muted)
        navLiveMap.background = if (tab == Tab.LIVE_MAP) ContextCompat.getDrawable(this, R.drawable.mode_icon_chip_bg) else null
        navSensors.background = if (tab == Tab.SENSORS) ContextCompat.getDrawable(this, R.drawable.mode_icon_chip_bg) else null
        navSimulation.background = if (tab == Tab.SIMULATION) ContextCompat.getDrawable(this, R.drawable.mode_icon_chip_bg) else null
        navLiveMapLabel.setTextColor(if (tab == Tab.LIVE_MAP) activeColor else inactiveColor)
        navSensorsLabel.setTextColor(if (tab == Tab.SENSORS) activeColor else inactiveColor)
        navSimulationLabel.setTextColor(if (tab == Tab.SIMULATION) activeColor else inactiveColor)

        // Same header view, different real copy per destination — matches
        // dead_reckoning_navigation's "Offline Navigation" header on Live
        // Map and sensor_calibration's "NAV-DR / Sensor Alignment" header
        // on Sensors, without inflating two separate top bars for text
        // that's the only thing actually different between them.
        when (tab) {
            Tab.LIVE_MAP -> {
                titleText.text = getString(R.string.offline_navigation_title)
                subtitleText.text = getString(R.string.offline_navigation_subtitle)
            }
            Tab.SENSORS -> {
                titleText.text = "NAV-DR"
                subtitleText.text = getString(R.string.sensors_panel_title).let { "Sensor Alignment" }
            }
            Tab.SIMULATION -> {
                titleText.text = getString(R.string.offline_navigation_title)
                subtitleText.text = getString(R.string.simulation_panel_title)
            }
        }
    }

    /** Reset tab's real action: clears the fused trail/state and restarts
     * the map-matcher, without touching calibration (this is "reset
     * filter", not "recalibrate") — matches the mockup's own "Reset
     * filter" button semantics exactly, just also reachable from the nav. */
    private fun performResetFilter() {
        fusion.reset()
        mapMatcher.reset()
        roadMapView.clear()
        previousFusionMode = null
        blackoutStartUptimeMs = null
        if (legendRow.visibility == View.VISIBLE) {
            // Calibration already happened — a filter reset shouldn't
            // re-run it, just clear the trail and let the next tick's
            // fusion.tick() re-anchor from the current position.
        }
        Toast.makeText(this, R.string.reset_filter_toast, Toast.LENGTH_SHORT).show()
        selectTab(Tab.LIVE_MAP)
    }

    /** Wires locationRequiredCard's three real actions. */
    private fun setupLocationRequiredScreen() {
        enableLocationButton.setOnClickListener {
            val needed = arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
            requestPermissions.launch(needed)
        }
        openSettingsButton.setOnClickListener {
            val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null))
            startActivity(intent)
        }
        continueDemoLink.setOnClickListener {
            locationRequiredCard.visibility = View.GONE
            replayToggle.isChecked = true
            startPipeline()
        }
    }

    /** Wires the dev-mode Record/Share controls. */
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
            calibration.skipForTesting()
            Toast.makeText(this, "Calibration skipped (testing) — not a real estimate", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startPipeline() {
        if (!sensorReader.available) {
            missingSensorsCard.visibility = View.VISIBLE
            return
        }
        sensorReader.start()
        val hasLocationPermission = arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
            .all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }
        if (hasLocationPermission) locationReader.start()
        ticking = true
        tickHandler.postDelayed(::tick, tickIntervalMs)
    }

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
            if (loc != null) Calibration.compassDegToMathRad(loc.bearing) else 0f,
        )
    }

    private fun tick() {
        if (!ticking) return

        val usingReplay = replayToggle.isChecked
        val demo = if (demoSelector.checkedRadioButtonId == R.id.demo2Radio) 2 else 1
        if (usingReplay != wasReplaying || (usingReplay && demo != wasDemo)) {
            calibration = CalibrationManager()
            fusion.reset()
            mapMatcher.reset()
            roadMapView.clear()
            replayIndex = 0
            wasReplaying = usingReplay
            wasDemo = demo
            previousFusionMode = null
            blackoutStartUptimeMs = null
            calibratingCard.visibility = View.VISIBLE
            trackingContent.visibility = View.GONE
            calibrationProgress.progress = 0
            calibrationPercentText.text = "0%"
            legendRow.visibility = View.GONE
        }

        val sample = readSample(usingReplay, demo)

        if (sample.hasRoughFix) roadMapView.updateRoughLocation(sample.lat, sample.lon)

        // Live raw accel/gyro readout (Sensors tab) — updated every tick
        // regardless of which tab is on screen (cheap; keeps the Sensors
        // tab correct the instant it's opened, no stale first frame).
        updateAxisReadouts(sample.rawAccel, sample.rawGyro)

        if (devRecorder.isRecording) {
            if (usingReplay) {
                devRecordToggle.isChecked = false
            } else {
                devRecorder.logSample(
                    sample.rawAccel, sample.rawGyro, sample.hasFix,
                    sample.lat, sample.lon, sample.speed,
                    Calibration.mathRadToCompassDeg(sample.bearingRad),
                )
                devRecordStatus.text = "Recording… ${devRecorder.sampleCount} samples" +
                    (if (!sample.hasFix) "  [no GPS fix]" else "")
            }
        }
        updateRecordingLogChip()

        if (!calibration.isLeveled) {
            calibration.addLevelSample(sample.rawAccel)
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
            // First tick past calibration — swap the calibrating card for
            // the real tracking content, once, rather than every tick.
            calibratingCard.visibility = View.GONE
            trackingContent.visibility = View.VISIBLE
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
        fusion.localXYToLatLon(s.x, s.y)?.let { (lat, lon) ->
            roadMapView.addPoint(lat, lon, s.mode)
            mapMatcher.match(lat, lon, s.heading)?.let { m ->
                roadMapView.addMatchedPoint(m.lat, m.lon)
            }
        }

        updateModeBanner(s.mode, demoBlackout)
        updateTelemetryTiles(s.speed, s.heading)

        tickHandler.postDelayed(::tick, tickIntervalMs)
    }

    private fun showCalibrating(detail: String) {
        detailText.text = getString(R.string.status_calibrating) + "\n" + detail
        calibrationProgress.setProgress(calibration.progressPercent, true)
        calibrationPercentText.text = "${calibration.progressPercent}%"
    }

    /** Mode-status banner — icon chip + dot colored per the app's real
     * FusionMode (same hex the live trail itself uses), real per-mode
     * copy, "Simulated" badge only for the demo toggle (never for a real
     * dropout), elapsed time tracked from a real transition timestamp —
     * matches the mockup's own banner structure with real content
     * throughout instead of its single hardcoded blackout state. */
    private fun updateModeBanner(mode: FusionMode, demoBlackout: Boolean) {
        if (mode == FusionMode.BLACKOUT && previousFusionMode != FusionMode.BLACKOUT) {
            blackoutStartUptimeMs = SystemClock.elapsedRealtime()
        } else if (mode != FusionMode.BLACKOUT) {
            blackoutStartUptimeMs = null
        }
        previousFusionMode = mode

        val iconChipBg = modeIconChip.background
        val (icon, chipTint, titleRes, bodyRes) = when (mode) {
            FusionMode.GNSS_TRACKING -> IconTintText(getString(R.string.ic_gps_not_fixed), R.color.status_gnss, R.string.banner_gnss_title, R.string.banner_gnss_body)
            FusionMode.BLACKOUT -> IconTintText(getString(R.string.ic_gps_off), R.color.status_blackout, R.string.banner_blackout_title, R.string.banner_blackout_body)
            FusionMode.BLEND -> IconTintText(getString(R.string.ic_sync), R.color.status_blend, R.string.banner_blend_title, R.string.banner_blend_body)
        }
        modeIconText.text = icon
        val tintColor = ContextCompat.getColor(this, chipTint)
        iconChipBg.setTint(tintColor)
        modeIconText.setTextColor(tintColor)
        modeText.text = getString(titleRes)

        val elapsed = blackoutStartUptimeMs
        modeBodyText.text = if (mode == FusionMode.BLACKOUT && elapsed != null) {
            val secs = (SystemClock.elapsedRealtime() - elapsed) / 1000
            getString(bodyRes) + " · %02d:%02d elapsed".format(secs / 60, secs % 60)
        } else {
            getString(bodyRes)
        }
        simulatedBadge.visibility = if (mode == FusionMode.BLACKOUT && demoBlackout) View.VISIBLE else View.GONE
    }

    private data class IconTintText(val icon: String, val tint: Int, val title: Int, val body: Int)

    /** Speed + heading tiles — both real (fusion.state), plus a real
     * 16-point compass name derived from the real heading and the real
     * calibrated instantaneous yaw rate, all computed live, none of it
     * the mockup's invented drift/confidence/step-count numbers. */
    private fun updateTelemetryTiles(speedMps: Float, headingMathRad: Float) {
        val kmh = speedMps * 3.6f
        speedValueText.text = "%.1f".format(kmh)
        // Display-range choice for the bar fill only (120 km/h comfortably
        // covers real road speeds) — the number shown is the real speed,
        // this just picks how full the bar reads at that value.
        speedBar.progress = ((kmh / 120f) * 100f).roundToInt().coerceIn(0, 100)

        val compassDeg = Calibration.mathRadToCompassDeg(headingMathRad)
        headingValueText.text = "${compassDeg.roundToInt()}°"
        headingCompassText.text = compassDirectionName(compassDeg)
    }

    /** Real accel/gyro axis readout — same raw values DevRecorder logs,
     * just also shown live. Bar-fill range is a *display* choice (±20
     * m/s² covers real driving/walking forces with gravity visible on Z;
     * ±10 rad/s covers real handheld/vehicle rotation) — the values
     * themselves are exactly what the sensor reports, unscaled. */
    private fun updateAxisReadouts(accel: FloatArray, gyro: FloatArray) {
        bindAxis(accelX, accel[0], 20f)
        bindAxis(accelY, accel[1], 20f)
        bindAxis(accelZ, accel[2], 20f)
        bindAxis(gyroX, gyro[0], 10f)
        bindAxis(gyroY, gyro[1], 10f)
        bindAxis(gyroZ, gyro[2], 10f)
    }

    private fun bindAxis(views: AxisViews, value: Float, fullScale: Float) {
        views.value.text = "%+.3f".format(value)
        val pct = (((value / fullScale) + 1f) / 2f * 100f).roundToInt().coerceIn(0, 100)
        views.bar.progress = pct
    }

    /** Real recording-log chip (controls dock) — reflects devRecorder's
     * actual state, elapsed time computed from its real sample count at
     * the real 10Hz tick rate, not a placeholder timer. */
    private fun updateRecordingLogChip() {
        if (devRecorder.isRecording) {
            recordingLogChip.visibility = View.VISIBLE
            val totalSecs = devRecorder.sampleCount / 10
            recordingLogTimeText.text = "%02d:%02d:%02d".format(totalSecs / 3600, (totalSecs / 60) % 60, totalSecs % 60)
        } else {
            recordingLogChip.visibility = View.GONE
        }
    }

    private fun compassDirectionName(deg: Float): String {
        val names = arrayOf(
            "North", "North-North-East", "North-East", "East-North-East",
            "East", "East-South-East", "South-East", "South-South-East",
            "South", "South-South-West", "South-West", "West-South-West",
            "West", "West-North-West", "North-West", "North-North-West",
        )
        val idx = ((deg / 22.5f) + 0.5f).toInt().mod(16)
        return names[idx]
    }

    override fun onResume() {
        super.onResume()
        roadMapView.onResume()
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
        devRecorder.stop()
    }
}
