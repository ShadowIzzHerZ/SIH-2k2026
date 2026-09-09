package com.sih26168.deadreckoning

import android.content.Context
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Developer-mode data recorder — logs real sensor+GPS samples to a CSV in
 * the exact column shape src/data/io_vnbd_loader.py's own fuzzy-matcher
 * already recognizes (`_FUZZY_PATTERNS`) — literal canonical names
 * (`accel_x`, `gyro_z`, `lat`, `speed_gt`, ...), so a recording made here
 * drops straight into `data/own_recordings/` and loads with zero format
 * wrangling. Same protocol data/own_recordings/README.md already
 * documents (a stationary calib_* clip, then an openroute_* or blackout_*
 * drive) — this just captures it from the app's own live pipeline
 * instead of a separate third-party sensor-logging app.
 *
 * Deliberately NOT on-device training. ONNX Runtime Mobile (this app's
 * inference engine) has no backprop; building a real training loop in
 * Kotlin isn't something to rush in the time before a hackathon deadline,
 * and wouldn't even be running the same, already-validated training code
 * this project's real numbers come from. The genuinely useful thing you
 * can do live, on a walk: capture real data now with the phone's actual
 * sensors, retrain properly offline afterward with the existing pipeline
 * (`python -m src.train --resume checkpoints/best.pt`) — same model, same
 * validated training loop, just fed a bit of your own real-world data on
 * top of IO-VNBD/comma2k19.
 */
class DevRecorder(private val context: Context) {
    private var writer: FileWriter? = null
    private var startNanos: Long = 0

    var sampleCount: Int = 0
        private set
    var currentFile: File? = null
        private set

    val isRecording: Boolean get() = writer != null

    /** Starts a new recording, closing any previous one first. `label`
     * becomes the filename prefix — "openroute" or "calib", matching
     * data/own_recordings/README.md's naming convention. */
    fun start(label: String): File {
        stop()
        val ts = SimpleDateFormat("yyyyMMdd_HHmm", Locale.US).format(Date())
        val dir = File(context.getExternalFilesDir(null), "recordings").apply { mkdirs() }
        val file = File(dir, "${label}_$ts.csv")
        val w = FileWriter(file)
        w.append("time_since_start,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,lat,lon,speed_gt,heading_gt\n")
        writer = w
        currentFile = file
        sampleCount = 0
        startNanos = System.nanoTime()
        return file
    }

    /** One row per tick. Raw (uncalibrated) accel/gyro on purpose —
     * src/calibration.py needs real raw phone-frame samples to estimate
     * the mounting angle from, same reasoning ReplayDataSource's own doc
     * gives for bundling raw (not pre-calibrated) replay data. A missing
     * GPS fix is written as blank fields, not skipped or interpolated —
     * a real blackout stretch inside a recording is expected and useful
     * (data/own_recordings/README.md says the same), not something to
     * paper over. */
    fun logSample(
        accel: FloatArray,
        gyro: FloatArray,
        hasFix: Boolean,
        lat: Double,
        lon: Double,
        speed: Float,
        bearingDeg: Float,
    ) {
        val w = writer ?: return
        val t = (System.nanoTime() - startNanos) / 1e9
        val fix = if (hasFix) "%.7f,%.7f,%.3f,%.2f".format(lat, lon, speed, bearingDeg) else ",,,"
        w.append(
            "%.3f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s\n".format(
                t, accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2], fix,
            )
        )
        sampleCount++
    }

    /** Flushes and closes the current file, returning it (null if nothing
     * was ever started). Safe to call when not recording. */
    fun stop(): File? {
        writer?.flush()
        writer?.close()
        writer = null
        return currentFile
    }
}
