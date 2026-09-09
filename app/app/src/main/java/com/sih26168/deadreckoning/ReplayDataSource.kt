package com.sih26168.deadreckoning

import android.content.Context
import org.json.JSONObject

data class ReplayRow(
    val accel: FloatArray,   // raw phone-frame [ax, ay, az]
    val gyro: FloatArray,    // raw phone-frame [gx, gy, gz]
    val lat: Double,
    val lon: Double,
    val speed: Float,        // m/s, GPS-chip-style (not position-differenced)
    val bearingRad: Float,   // radians, 0=east ccw+ (already converted from compass degrees)
)

/**
 * Replays a real, previously-validated drive (assets/replay_drive.json —
 * a raw comma2k19 highway segment, the same one that measured 2.9% drift
 * over a real 30s continuous blackout in src/simulate_blackout.py's
 * offline evaluation) through the exact same live tick loop MainActivity
 * uses for real sensors/GPS.
 *
 * Exists for two reasons: (1) it's how this app was actually verified
 * end-to-end on a real device without needing to physically drive around
 * during development, and (2) it doubles as a genuinely useful demo mode
 * — judging rooms don't have GPS reception or room to drive, so replaying
 * a real recorded drive is the only way to demo the live pipeline (not
 * just plots) indoors. Raw (uncalibrated) accel/gyro on purpose — the
 * app's own CalibrationManager runs on this exactly like it would on live
 * sensor data, not pre-calibrated data that would skip real app code.
 */
class ReplayDataSource(context: Context, assetName: String = "replay_drive.json") {
    val rows: List<ReplayRow>
    val dt: Float

    init {
        val text = context.assets.open(assetName).bufferedReader().use { it.readText() }
        val root = JSONObject(text)
        dt = root.getDouble("dt").toFloat()
        val rowsJson = root.getJSONArray("rows")
        val list = ArrayList<ReplayRow>(rowsJson.length())
        for (i in 0 until rowsJson.length()) {
            val r = rowsJson.getJSONObject(i)
            list.add(
                ReplayRow(
                    accel = floatArrayOf(r.getDouble("ax").toFloat(), r.getDouble("ay").toFloat(), r.getDouble("az").toFloat()),
                    gyro = floatArrayOf(r.getDouble("gx").toFloat(), r.getDouble("gy").toFloat(), r.getDouble("gz").toFloat()),
                    lat = r.getDouble("lat"),
                    lon = r.getDouble("lon"),
                    speed = r.getDouble("speed").toFloat(),
                    // r.getDouble("bearing") is real compass bearing degrees
                    // (confirmed against this file's own lat/lon: matches
                    // computed great-circle bearing to ~0.1°) — needs the
                    // compass->math axis swap, not just a unit conversion.
                    // See Calibration.compassDegToMathRad's doc for the real
                    // bug a plain Math.toRadians() here used to cause.
                    bearingRad = Calibration.compassDegToMathRad(r.getDouble("bearing").toFloat()),
                )
            )
        }
        rows = list
    }
}
