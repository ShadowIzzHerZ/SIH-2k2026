package com.sih26168.deadreckoning

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

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
 * windowSize samples of a blackout, or when GNSS returns mid-chunk).
 * Between boundaries, `tick()` extrapolates the displayed position forward
 * at the last known speed/heading (dead-simple constant-velocity dead
 * reckoning) purely for a smooth live UI — the authoritative per-chunk
 * result still corrects it every 5s, same as the real algorithm.
 *
 * The exported ONNX model (checkpoints/dead_reckoning_model.onnx) has a
 * FIXED input shape — exactly windowSize samples, no dynamic axes (see
 * export_onnx.py's docstring: "Dynamic axes are left off ... on purpose").
 * Confirmed the hard way, crashing on a real device: a short final chunk
 * (GNSS returning mid-chunk, e.g. 25 of 50 samples in) can't just be
 * handed to the model at its own shorter length. Fixed with a continuous
 * rolling buffer of the last windowSize *real* calibrated samples,
 * updated every tick regardless of mode — a chunk resolution always feeds
 * the model that buffer's current (always exactly windowSize, always
 * real, never padded) contents, and only harvests the LAST `pending`
 * corrections (the ones whose window position actually corresponds to
 * this chunk's own new samples) for integration. A full mid-blackout
 * chunk harvests all windowSize of them, identical to before; a short
 * final chunk harvests fewer, each still computed with genuine real
 * leading context from the rolling buffer rather than fabricated/padded
 * samples.
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

    // Continuous real-sample history — always exactly windowSize once
    // filled, fed every tick regardless of mode (see class doc).
    private val rollingWindow = ArrayDeque<FloatArray>(windowSize)
    private var pending = 0             // new blackout samples since the last chunk resolution
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
        if (rollingWindow.size >= windowSize) rollingWindow.removeFirst()
        rollingWindow.addLast(calibratedAccel + calibratedGyro)

        // The model can only ever be called with a full windowSize buffer
        // (see class doc) — force GNSS_TRACKING behavior, real or
        // simulated blackout ignored, until there's enough real history to
        // safely resolve a chunk. Only matters for the first ~5s right
        // after calibration completes.
        val effectiveAvailable = available || rollingWindow.size < windowSize

        if (effectiveAvailable && !wasAvailable) {
            // GNSS just came back — fold in any pending (< windowSize)
            // blackout samples so a short blackout isn't silently
            // dropped, *then* snapshot state as the BLEND ramp's start.
            flushPending()
            blendRemaining = blendSamples
            blendFrom = state.copy()
        }

        if (effectiveAvailable) {
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
            pending = 0
        } else {
            if (pending == 0) chunkAnchor = state.copy()
            blendRemaining = 0
            pending++

            // Cheap live extrapolation between chunk boundaries — see class
            // doc. Uses the anchor's own speed/heading, not the network:
            // this is a display-only convenience, not part of the accuracy
            // path.
            val elapsed = pending * dt
            val extrapX = chunkAnchor.x + chunkAnchor.speed * cos(chunkAnchor.heading.toDouble()).toFloat() * elapsed
            val extrapY = chunkAnchor.y + chunkAnchor.speed * sin(chunkAnchor.heading.toDouble()).toFloat() * elapsed
            state = FusionState(FusionMode.BLACKOUT, extrapX, extrapY, chunkAnchor.heading, chunkAnchor.speed)

            if (pending >= windowSize) resolveChunk()
        }
        wasAvailable = effectiveAvailable
    }

    /** Call if a blackout ends (GNSS returns) with pending (< windowSize)
     * samples still unresolved — folds them in before the BLEND ramp
     * starts, so a short blackout isn't silently dropped. */
    private fun flushPending() {
        if (pending > 0) resolveChunk()
    }

    /**
     * Resolves `pending` new blackout samples: one model call on the
     * rolling buffer's current windowSize real samples (always full —
     * see class doc for why that matters), harvesting only the last
     * `pending` corrections (the ones with real leading context from this
     * chunk and whatever real data preceded it) and integrating them
     * together from chunkAnchor — same "whole-window, not last-position"
     * reasoning as the offline fusion.py's run_fusion.
     */
    private fun resolveChunk() {
        val window = rollingWindow.toTypedArray()  // always windowSize once filled
        val corrections = model.predict(window)     // (windowSize, 2)
        val startIdx = window.size - pending

        var v = chunkAnchor.speed
        var th = chunkAnchor.heading
        var x = chunkAnchor.x
        var y = chunkAnchor.y
        for (i in startIdx until window.size) {
            val forwardAccel = window[i][0] + corrections[i][0]
            val yawRate = window[i][5] + corrections[i][1]
            v = (v + forwardAccel * dt).coerceIn(vClampMin, vClampMax)
            th = wrapAngle(th + yawRate * dt)
            x += v * cos(th.toDouble()).toFloat() * dt
            y += v * sin(th.toDouble()).toFloat() * dt
        }
        state = FusionState(FusionMode.BLACKOUT, x, y, th, v)
        chunkAnchor = state.copy()
        pending = 0
    }

    fun reset() {
        state = FusionState()
        refLat = null; refLon = null
        rollingWindow.clear()
        pending = 0
        blendRemaining = 0
        wasAvailable = true
    }
}
