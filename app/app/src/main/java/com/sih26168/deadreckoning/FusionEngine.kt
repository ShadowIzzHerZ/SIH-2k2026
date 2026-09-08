package com.sih26168.deadreckoning

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

enum class FusionMode { GNSS_TRACKING, BLACKOUT, BLEND }

data class FusionState(
    var mode: FusionMode = FusionMode.GNSS_TRACKING,
    var x: Float = 0f,       // local east, metres
    var y: Float = 0f,       // local north, metres
    var heading: Float = 0f, // radians, 0 = east, ccw+
    var speed: Float = 0f,   // m/s
)

/**
 * Live, on-device counterpart to src/fusion.py's run_fusion — same state
 * machine (GNSS_TRACKING / BLACKOUT / BLEND), same discrete-chunk
 * BLACKOUT design (see fusion.py's docstring for the two real bugs found
 * and fixed getting there: noisy position-diff speed/heading seeding, and
 * an unclamped runaway speed integration), driven by live sensor/location
 * ticks instead of a fixed offline array.
 *
 * Accuracy-critical integration only happens at chunk boundaries (every
 * windowSize samples of a blackout), exactly like the validated offline
 * version. Between boundaries, `tick()` extrapolates the displayed
 * position forward at the last known speed/heading (dead-simple constant-
 * velocity dead reckoning) purely for a smooth live UI — the authoritative
 * per-chunk result still corrects it every 5s, same as the real algorithm.
 */
class FusionEngine(
    private val model: BiasCorrectionModel,
    private val windowSize: Int = 50,
    private val dt: Float = 0.1f,
    private val blendSeconds: Float = 2.0f,
    private val vClampMin: Float = 0f,
    private val vClampMax: Float = 50f,
) {
    var state = FusionState()
        private set

    private var refLat: Double? = null
    private var refLon: Double? = null

    private val chunkAccel = ArrayList<FloatArray>(windowSize)
    private val chunkGyro = ArrayList<FloatArray>(windowSize)
    private var chunkAnchor = FusionState()

    private val blendSamples = maxOf(1, (blendSeconds / dt).toInt())
    private var blendRemaining = 0
    private var blendFrom = FusionState()

    private var wasAvailable = true

    /** East/north metres from the first-ever fix — same formula as
     * src/data/io_vnbd_loader.py's latlon_to_local_xy. */
    private fun toLocalXY(lat: Double, lon: Double): FloatArray {
        if (refLat == null) { refLat = lat; refLon = lon }
        val r = 6371000.0
        val lat0Rad = Math.toRadians(refLat!!)
        val x = Math.toRadians(lon - refLon!!) * r * cos(lat0Rad)
        val y = Math.toRadians(lat - refLat!!) * r
        return floatArrayOf(x.toFloat(), y.toFloat())
    }

    private fun wrapAngle(a: Float): Float = atan2(sin(a.toDouble()), cos(a.toDouble())).toFloat()

    /**
     * One 10Hz tick. `calibratedAccel`/`calibratedGyro` are already in the
     * vehicle frame (see Calibration). `available`/`lat`/`lon`/`gnssSpeed`/
     * `gnssHeadingRad` should reflect a *real* GNSS fix, Doppler-derived
     * speed/course preferred — see fusion.py's run_fusion docstring for why
     * position-differencing is a known hazard, not a neutral fallback.
     * `available=false` also covers the manual demo blackout toggle.
     */
    fun tick(
        calibratedAccel: FloatArray,
        calibratedGyro: FloatArray,
        available: Boolean,
        lat: Double,
        lon: Double,
        gnssSpeed: Float,
        gnssHeadingRad: Float,
    ) {
        if (available && !wasAvailable) {
            // GNSS just came back — fold in any partial (< windowSize)
            // chunk still buffered so a short blackout isn't silently
            // dropped, *then* snapshot state as the BLEND ramp's start.
            flushPartialChunk()
            blendRemaining = blendSamples
            blendFrom = state.copy()
        }

        if (available) {
            val xy = toLocalXY(lat, lon)
            val gnssState = FusionState(FusionMode.GNSS_TRACKING, xy[0], xy[1], gnssHeadingRad, gnssSpeed)

            state = if (blendRemaining > 0) {
                val w = 1f - blendRemaining / blendSamples.toFloat()
                blendRemaining--
                FusionState(
                    FusionMode.BLEND,
                    (1 - w) * blendFrom.x + w * gnssState.x,
                    (1 - w) * blendFrom.y + w * gnssState.y,
                    wrapAngle((1 - w) * blendFrom.heading + w * gnssState.heading),
                    (1 - w) * blendFrom.speed + w * gnssState.speed,
                )
            } else {
                gnssState
            }
            chunkAccel.clear(); chunkGyro.clear()
        } else {
            if (chunkAccel.isEmpty()) chunkAnchor = state.copy()
            blendRemaining = 0

            chunkAccel.add(calibratedAccel)
            chunkGyro.add(calibratedGyro)

            // Cheap live extrapolation between chunk boundaries — see class
            // doc. Uses the anchor's own speed/heading, not the network:
            // this is a display-only convenience, not part of the accuracy
            // path.
            val elapsed = chunkAccel.size * dt
            val extrapX = chunkAnchor.x + chunkAnchor.speed * cos(chunkAnchor.heading.toDouble()).toFloat() * elapsed
            val extrapY = chunkAnchor.y + chunkAnchor.speed * sin(chunkAnchor.heading.toDouble()).toFloat() * elapsed
            state = FusionState(FusionMode.BLACKOUT, extrapX, extrapY, chunkAnchor.heading, chunkAnchor.speed)

            if (chunkAccel.size >= windowSize) {
                runChunk()
            }
        }
        wasAvailable = available
    }

    /** Call if a blackout ends (GNSS returns) with a partial (< windowSize)
     * chunk still buffered — folds it in before the BLEND ramp starts, so
     * a short blackout isn't silently dropped. */
    private fun flushPartialChunk() {
        if (chunkAccel.isNotEmpty()) runChunk()
    }

    /** One discrete BLACKOUT chunk: full-window model call, whole
     * correction sequence integrated together from chunkAnchor — see
     * fusion.py's run_fusion for the exact same logic and why it matters
     * (position-invariance across the network's own training window). */
    private fun runChunk() {
        val n = chunkAccel.size
        val window = Array(n) { i -> chunkAccel[i] + chunkGyro[i] }  // (T, 6): [ax,ay,az,gx,gy,gz]
        val corrections = model.predict(window)  // (T, 2): [delta_v, delta_theta]

        var v = chunkAnchor.speed
        var th = chunkAnchor.heading
        var x = chunkAnchor.x
        var y = chunkAnchor.y
        for (i in 0 until n) {
            val forwardAccel = chunkAccel[i][0] + corrections[i][0]
            val yawRate = chunkGyro[i][2] + corrections[i][1]
            v = (v + forwardAccel * dt).coerceIn(vClampMin, vClampMax)
            th = wrapAngle(th + yawRate * dt)
            x += v * cos(th.toDouble()).toFloat() * dt
            y += v * sin(th.toDouble()).toFloat() * dt
        }
        state = FusionState(FusionMode.BLACKOUT, x, y, th, v)
        chunkAnchor = state.copy()
        chunkAccel.clear(); chunkGyro.clear()
    }

    fun reset() {
        state = FusionState()
        refLat = null; refLon = null
        chunkAccel.clear(); chunkGyro.clear()
        blendRemaining = 0
        wasAvailable = true
    }
}
