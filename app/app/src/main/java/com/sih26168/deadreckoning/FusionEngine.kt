package com.sih26168.deadreckoning

import kotlin.math.atan2
import kotlin.math.cos
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
 *
 * ZUPT (zero-velocity update): a real bug found live on-device, not in the
 * validated offline eval. The bias-correction network was trained on real
 * *vehicle* driving (comma2k19/IO-VNBD — mounted, road speeds) — it has
 * never seen "phone picked up and carried by hand while basically
 * standing still", which is wildly out-of-distribution. During a blackout
 * that domain mismatch means its correction for near-zero real motion
 * isn't reliably near-zero either, and a small constant residual yaw-rate
 * bias integrated every tick for an entire chunk compounds, chunk after
 * chunk, into a full circular loop — visibly "wandering to random places"
 * while the phone was actually just sitting there. Fixed with a standard
 * INS stabilization technique: when the raw (pre-correction) accel/gyro
 * for a sample says the device is essentially at rest, don't integrate
 * the network's correction for that instant at all — decay speed toward
 * zero and hold heading instead of trusting a correction computed in a
 * regime the network was never trained on.
 *
 * That first version gated purely on accel/gyro magnitude, which is a
 * real bug of its own: Newton's first law means a vehicle cruising at a
 * genuinely constant speed also reads near-zero net accel — it's
 * indistinguishable from "at rest" using accel/gyro alone. Found live via
 * the replay demo: a real ~110 km/h highway stretch (comma2k19, gently
 * curving, no real stop) has small enough instantaneous accel/yaw-rate to
 * pass the accel/gyro-only check, so ZUPT decayed a genuinely-110km/h
 * speed toward zero — the fused trail visibly reversed course mid-cruise,
 * confirmed against the raw recorded data (speed held 30-33 m/s and gz
 * stayed under 0.06 rad/s the entire time — nothing in the real data
 * justified treating it as a stop). Fixed by also requiring the
 * currently-tracked speed to already be low (see isQuiet's zuptMaxSpeed
 * doc) — a cruising vehicle's speed isn't low, so it's exempt regardless
 * of how quiet the instantaneous reading looks, while a phone actually at
 * rest (speed already near zero, however it got there) still gets ZUPT
 * exactly as before.
 */
class FusionEngine(
    private val model: Predictor,
    private val windowSize: Int = 50,
    private val dt: Float = 0.1f,
    private val blendSeconds: Float = 2.0f,
    private val vClampMin: Float = 0f,
    private val vClampMax: Float = 50f,
    private val quietAccelThresh: Float = 0.35f,  // m/s^2, horizontal (ax,ay) magnitude
    private val quietGyroThresh: Float = 0.05f,   // rad/s (~2.9 deg/s), full 3-axis magnitude
    // Only samples where the currently-tracked speed is already below
    // this count as ZUPT-eligible, regardless of how quiet accel/gyro
    // look — see isQuiet's doc. 3 m/s (~11 km/h) sits comfortably above a
    // brisk walk (the real "holding the phone" scenario this exists for)
    // and comfortably below any real vehicle cruising speed.
    private val zuptMaxSpeed: Float = 3.0f,
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

    /** Inverse of toLocalXY — lets a caller (e.g. MapMatcher, which works
     * in real lat/lon) convert the fused local position back to a real
     * coordinate, using this engine's own reference fix. Returns null
     * before any GNSS fix has ever been seen (no reference point yet). */
    fun localXYToLatLon(x: Float, y: Float): DoubleArray? {
        val lat0 = refLat ?: return null
        val lon0 = refLon ?: return null
        val r = 6371000.0
        val lat0Rad = Math.toRadians(lat0)
        val lat = lat0 + Math.toDegrees(y / r)
        val lon = lon0 + Math.toDegrees(x / (r * cos(lat0Rad)))
        return doubleArrayOf(lat, lon)
    }

    /** Public counterpart to toLocalXY, for converting a *result* (e.g. a
     * MapMatcher-snapped lat/lon) back into this same local frame for
     * rendering alongside the fused trajectory. Returns null before any
     * GNSS fix has ever been seen. */
    fun latLonToLocalXY(lat: Double, lon: Double): FloatArray? {
        if (refLat == null) return null
        return toLocalXY(lat, lon)
    }

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
            if (isQuiet(window[i], v)) {
                // ZUPT — raw sensors say the device is at rest right now;
                // don't trust the network's correction for this instant
                // (out-of-distribution for anything but real driving), just
                // decay any stale speed and hold heading. See class doc.
                v *= 0.7f
            } else {
                val forwardAccel = window[i][0] + corrections[i][0]
                val yawRate = window[i][5] + corrections[i][1]
                v = (v + forwardAccel * dt).coerceIn(vClampMin, vClampMax)
                th = wrapAngle(th + yawRate * dt)
            }
            x += v * cos(th.toDouble()).toFloat() * dt
            y += v * sin(th.toDouble()).toFloat() * dt
        }
        state = FusionState(FusionMode.BLACKOUT, x, y, th, v)
        chunkAnchor = state.copy()
        pending = 0
    }

    /** True if a raw (pre-correction) calibrated sample [ax,ay,az,gx,gy,gz]
     * indicates the device is essentially at rest right now — see class
     * doc's ZUPT note for why resolveChunk uses this to distrust the
     * network's correction instead of applying it.
     *
     * currentSpeed is load-bearing, not a tiebreaker: accel/gyro alone
     * cannot tell "at rest" apart from "moving at a constant velocity" —
     * Newton's first law says both read as ~zero net force. Found live on
     * a real device via the replay demo: real ~110 km/h highway cruising
     * (comma2k19, genuinely constant speed, gently curving) has small
     * enough horizontal accel and yaw rate that it passed the
     * accel/gyro-only check, so ZUPT decayed a genuinely-110km/h speed
     * toward zero and the fused trail visibly reversed course — confirmed
     * against the raw recorded data (speed held 30-33 m/s, gz stayed under
     * 0.06 rad/s throughout; there was no real stop or sharp turn there).
     * Gating on currentSpeed already being low fixes the ambiguity: a
     * cruising vehicle's speed is not low, so it's exempt regardless of
     * how quiet the instantaneous accel/gyro looks. */
    private fun isQuiet(sample: FloatArray, currentSpeed: Float): Boolean {
        if (currentSpeed >= zuptMaxSpeed) return false
        val accelMag = sqrt((sample[0] * sample[0] + sample[1] * sample[1]).toDouble()).toFloat()
        val gyroMag = sqrt((sample[3] * sample[3] + sample[4] * sample[4] + sample[5] * sample[5]).toDouble()).toFloat()
        return accelMag < quietAccelThresh && gyroMag < quietGyroThresh
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
