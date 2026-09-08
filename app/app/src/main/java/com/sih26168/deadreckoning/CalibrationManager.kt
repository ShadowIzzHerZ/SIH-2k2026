package com.sih26168.deadreckoning

/**
 * Drives Calibration.kt's two-stage leveling + yaw estimate live, on
 * startup, instead of over an already-recorded file — same two stages as
 * calibration.py's module docstring: leveling first (assumes the phone is
 * roughly still for the first `levelSamples` readings, e.g. mounted before
 * pulling off), then yaw alignment once during an initial confident
 * (moving, GPS-available) stretch, locked in after that — "leveling
 * happens continuously, yaw-alignment happens once at trip start", per
 * that module's own reasoning, which applies just as much live as offline.
 *
 * minConfidentSamples is far smaller than calibrate_sequence's offline
 * default (3000, meant for a whole recorded file) — a live app can't make
 * the user drive confidently for 5 minutes before showing anything; ~3s of
 * real motion is enough to lock in a reasonable estimate for a demo.
 */
class CalibrationManager(
    private val levelSamples: Int = 20,
    private val minConfidentSamples: Int = 30,
    private val minSpeedMps: Float = 2.0f,
) {
    private val levelBuf = ArrayList<FloatArray>(levelSamples)
    private val yawAccelBuf = ArrayList<FloatArray>()
    private val yawHeadingBuf = ArrayList<FloatArray>()

    var rLevel: FloatArray? = null
        private set
    var psiYaw: Float = 0f
        private set
    var isReady: Boolean = false
        private set

    val isLeveled: Boolean get() = rLevel != null

    /** Feed one raw accel sample while leveling hasn't completed yet. */
    fun addLevelSample(rawAccel: FloatArray) {
        if (isLeveled) return
        levelBuf.add(rawAccel)
        if (levelBuf.size >= levelSamples) {
            rLevel = Calibration.estimateGravityVector(levelBuf).let { Calibration.levelingRotation(it) }
        }
    }

    /** Feed one (leveled accel, GPS speed, GPS heading) sample while yaw
     * hasn't locked in yet — only call once isLeveled and GNSS is
     * available with speed > minSpeedMps. */
    fun addYawSample(rawAccel: FloatArray, gpsSpeed: Float, gpsHeadingRad: Float) {
        if (!isLeveled || isReady || gpsSpeed <= minSpeedMps) return
        val leveled = Calibration.matVec3(rLevel!!, rawAccel)
        yawAccelBuf.add(floatArrayOf(leveled[0], leveled[1]))
        yawHeadingBuf.add(floatArrayOf(kotlin.math.cos(gpsHeadingRad.toDouble()).toFloat(),
            kotlin.math.sin(gpsHeadingRad.toDouble()).toFloat()))
        if (yawAccelBuf.size >= minConfidentSamples) {
            psiYaw = Calibration.estimateYawMisalignment(yawAccelBuf, yawHeadingBuf)
            isReady = true
        }
    }

    /** Calibrate one raw [ax,ay,az] or [gx,gy,gz] sample — leveling-only
     * (psiYaw=0) once leveled but before yaw locks in, full calibration
     * once isReady. Returns the raw sample unchanged if not even leveled
     * yet (nothing usable to apply). */
    fun calibrate(raw: FloatArray): FloatArray {
        val r = rLevel ?: return raw
        return Calibration.calibrateSample(raw, r, psiYaw)
    }
}
