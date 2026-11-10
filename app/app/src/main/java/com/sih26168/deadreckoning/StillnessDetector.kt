package com.sih26168.deadreckoning

import kotlin.math.sqrt

/**
 * "Is the phone physically still right now", from the raw IMU alone, over
 * the last [windowTicks] ticks (2 s at the 10 Hz tick). Exists because GNSS
 * Doppler speed is not zero on a phone that isn't moving: on a Galaxy F17
 * indoors it showed 8-10 km/h (2-3 m/s) at rest, too high for any fixed
 * speed floor to remove without also hiding a slow walk. A walk or a drive
 * shakes the accelerometer well past [accelStdMax]; a phone on a table
 * doesn't.
 *
 * Uses the spread of the accelerometer magnitude, so gravity and the mount
 * angle drop out, and the largest gyro magnitude in the window.
 */
class StillnessDetector(
    private val windowTicks: Int = 20,
    private val accelStdMax: Float = 0.15f,   // m/s^2
    private val gyroMax: Float = 0.05f,       // rad/s (~2.9 deg/s)
) {
    private val accelMag = ArrayDeque<Float>(windowTicks)
    private val gyroMag = ArrayDeque<Float>(windowTicks)

    fun add(accel: FloatArray, gyro: FloatArray) {
        if (accelMag.size >= windowTicks) { accelMag.removeFirst(); gyroMag.removeFirst() }
        accelMag.addLast(norm(accel))
        gyroMag.addLast(norm(gyro))
    }

    fun reset() { accelMag.clear(); gyroMag.clear() }

    /** False until a full window has been seen. */
    val isStill: Boolean
        get() {
            if (accelMag.size < windowTicks) return false
            val mean = accelMag.sum() / accelMag.size
            val variance = accelMag.sumOf { ((it - mean) * (it - mean)).toDouble() }.toFloat() / accelMag.size
            return sqrt(variance) < accelStdMax && gyroMag.max() < gyroMax
        }

    private fun norm(v: FloatArray): Float = sqrt(v.sumOf { (it * it).toDouble() }).toFloat()
}

/** GNSS speed shown as 0 while the IMU says the phone is still. The
 * [maxStillSpeedMps] cap keeps a smooth constant-speed cruise (steady
 * accelerometer, no turning) from ever being zeroed. */
internal fun speedWhileStill(speedMps: Float, imuStill: Boolean, maxStillSpeedMps: Float = 4.0f): Float =
    if (imuStill && speedMps < maxStillSpeedMps) 0f else speedMps
