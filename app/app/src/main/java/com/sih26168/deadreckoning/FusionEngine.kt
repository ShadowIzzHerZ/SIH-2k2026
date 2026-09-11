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
 *
 * That second version gated on the *live* integrated speed, which turned
 * out to be a one-way door and a real bug of its own — found live via
 * `adb logcat` reproducing a reported "speed jumps to ~37 km/h a second
 * into a blackout": phone essentially stationary, GNSS logged at ~1 m/s
 * the whole time, yet fusedSpeed climbed unbounded chunk after chunk, past
 * 37 km/h and still rising past 130 km/h before being caught. Once the
 * model's own out-of-distribution correction pushed v above zuptMaxSpeed
 * even once, ZUPT stopped firing for the rest of that blackout — exactly
 * when it was needed most — letting every later chunk's bias compound
 * unchecked. Fixed by gating on blackoutEntrySpeed (the last real
 * GNSS-confirmed speed, frozen once at the blackout transition) instead of
 * the live v — see isQuiet's doc for why that breaks the feedback loop
 * without reintroducing the 110 km/h reversed-course bug above.
 *
 * That fix alone wasn't sufficient, confirmed with the ZUPT gate itself
 * instrumented live: for a real handheld/walking-pace recording, isQuiet's
 * per-sample accel/gyro thresholds (quietAccelThresh/quietGyroThresh, both
 * tuned against comma2k19's vehicle-mounted, low-noise IMU) matched ZERO
 * of 50 samples in the runaway chunk — a handheld phone's natural jitter
 * routinely exceeds thresholds tuned for a rigidly-mounted one, so ZUPT
 * structurally cannot fire for exactly the "presenter holding the phone"
 * scenario the demo blackout toggle is meant for. The logged raw data
 * showed why that matters: avgRawAccelX ~3.37 m/s^2 sustained across the
 * whole chunk (a real, constant bias — almost certainly leveling-
 * calibration leakage from a not-quite-level hold, not zero-mean noise;
 * genuine jitter would average back toward zero over 50 samples, a
 * constant offset doesn't), with the model's own correction only
 * partially canceling it (net avgForwardAccel ~+0.75 m/s^2) — small
 * enough to look "reasonable" per-sample, large enough to add several
 * m/s of speed every 5s chunk, indefinitely. Fixed with a second,
 * independent safeguard that doesn't depend on any per-sample threshold
 * at all: when blackoutEntrySpeed indicates a slow/near-stationary entry
 * (below zuptMaxSpeed), cap that chunk's integration to a tight band
 * around blackoutEntrySpeed (see lowSpeedVMargin) instead of the full
 * vehicle-scale vClampMax — a phone that was at walking pace when GNSS
 * was last available has no real way to reach vehicle speeds within one
 * blackout without a genuine, large acceleration event, which this would
 * still allow (the margin isn't zero); a genuinely fast blackoutEntrySpeed
 * keeps the wide vClampMax exactly as before, so real highway
 * acceleration/deceleration during a blackout is unaffected.
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
    // Second, independent safeguard against the runaway-speed bug —
    // see class doc's third ZUPT paragraph. Only applied when
    // blackoutEntrySpeed < zuptMaxSpeed (same "slow/near-stationary
    // entry" regime ZUPT targets); a genuinely fast entry keeps the full
    // vClampMax range. 4 m/s (~14.4 km/h) on top of a walking-pace entry
    // comfortably covers a real brisk-walk-to-jog range without
    // resembling the reported bug's actual failure mode (tens of km/h
    // within a couple chunks with the raw GNSS speed never leaving ~1 m/s).
    private val lowSpeedVMargin: Float = 4.0f,
    // Below this GNSS speed, Location.getBearing() is either flagged
    // invalid (hasBearing()==false) or, worse, a stale/last-good value the
    // provider never cleared — a real bug found live: sitting still
    // indoors with a rough network fix, heading kept snapping to whatever
    // compass-0 default or last-cached bearing the provider handed back,
    // which then got trusted as the authoritative GNSS_TRACKING heading
    // every tick. 1 m/s (~3.6 km/h) is comfortably below CalibrationManager's
    // own minSpeedMps=2.0 gate for locking yaw in the first place — this is
    // a looser "is this instant's bearing even meaningful" check, not a
    // calibration-quality bar. Below it, tick() holds the last fused
    // heading instead of snapping to a bogus one; x/y/speed still update
    // from the real fix as normal.
    private val minHeadingLockSpeedMps: Float = 1.0f,
    // Visual-only catch-up duration for a chunk transition — see
    // resolveChunk's doc for the real bug this fixes (a visible kink
    // every 5s wherever the real path curved within a chunk). Exposed as
    // a constructor param (not just a private constant) so tests can
    // dial it to 0 and confirm the kink it's meant to fix actually
    // reappears without it — see FusionEngineTest.
    private val chunkBlendSeconds: Float = 0.5f,
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
    // The last real GNSS-confirmed speed before the current blackout began
    // (frozen once at the blackout transition, see tick()) — isQuiet's own
    // ZUPT gate reference. NOT the same as chunkAnchor.speed once a chunk
    // has resolved: chunkAnchor.speed becomes the *integrated* result,
    // which is exactly what can run away (see isQuiet's doc); this field
    // never gets overwritten by anything but a real GNSS fix.
    private var blackoutEntrySpeed = 0f

    private val blendSamples = maxOf(1, (blendSeconds / dt).toInt())
    private var blendRemaining = 0
    private var blendFrom = FusionState()

    // Visual-only catch-up for chunk transitions — see resolveChunk's doc
    // for the real bug this fixes (a visible kink/"zigzag" every 5s
    // whenever the real path curved within a chunk). Short on purpose:
    // long enough to not look like a jump, short enough that the display
    // is back on the true resolved path well before the next chunk.
    private val chunkBlendSamples = maxOf(1, (chunkBlendSeconds / dt).toInt())
    private var chunkBlendRemaining = 0
    private var chunkBlendFrom = FusionState()

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
     *
     * `gnssHasBearing` is the caller's `Location.hasBearing()` (always true
     * for replay, which carries real recorded courses) — combined with
     * `gnssSpeed` against [minHeadingLockSpeedMps] to decide whether this
     * tick's bearing is actually trustworthy enough to lock heading to. See
     * [minHeadingLockSpeedMps]'s doc for the real bug this guards against.
     */
    fun tick(
        calibratedAccel: FloatArray,
        calibratedGyro: FloatArray,
        available: Boolean,
        lat: Double,
        lon: Double,
        gnssSpeed: Float,
        gnssHeadingRad: Float,
        // Defaults true so every existing call site/test that doesn't care
        // about this gate (synthetic fixtures with no real Location object)
        // keeps its prior behavior; MainActivity's real tick() passes the
        // real Location.hasBearing() explicitly.
        gnssHasBearing: Boolean = true,
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
            // dropped, *then* snapshot the BLEND ramp's start.
            //
            // From chunkAnchor, NOT state: state can be mid-way through
            // its own short visual catch-up from the last chunk boundary
            // right now (see resolveChunk's doc) and so isn't necessarily
            // the true last blackout position yet — chunkAnchor always is,
            // updated immediately inside resolveChunk regardless of how
            // long the display takes to visually catch up to it.
            flushPending()
            blendRemaining = blendSamples
            blendFrom = chunkAnchor.copy()
            chunkBlendRemaining = 0  // superseded by the GNSS-reconnect blend starting now
        }
        if (!effectiveAvailable && wasAvailable) {
            // Blackout just started fresh (from GNSS_TRACKING or BLEND) —
            // anchor to the real current state and start clean. Explicit
            // transition check, not just "pending == 0": pending is ALSO
            // 0 immediately after every chunk resolution (see
            // resolveChunk), and conflating the two used to make this
            // line stomp chunkAnchor with a stale mid-blend display value
            // every 5s — found writing the chunk-transition blend below,
            // not live; see resolveChunk's doc for why this line has to
            // be unambiguous about which case it's actually handling.
            chunkAnchor = state.copy()
            // Frozen once, here, for the whole blackout — see isQuiet's
            // doc for the real runaway-speed bug this fixes (ZUPT gating
            // on the live *integrated* v instead of this).
            blackoutEntrySpeed = state.speed
            pending = 0
            chunkBlendRemaining = 0
        }

        if (effectiveAvailable) {
            val xy = toLocalXY(lat, lon)
            // Only lock heading to this fix's bearing when it's actually
            // meaningful (see minHeadingLockSpeedMps's doc) — otherwise
            // hold the last fused heading. Position/speed still update from
            // the real fix either way; only the heading source is gated.
            val headingValid = gnssHasBearing && gnssSpeed > minHeadingLockSpeedMps
            val lockedHeading = if (headingValid) gnssHeadingRad else state.heading
            val gnssState = FusionState(FusionMode.GNSS_TRACKING, xy[0], xy[1], lockedHeading, gnssSpeed)

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
            blendRemaining = 0
            pending++

            state = if (chunkBlendRemaining > 0) {
                // Easing toward a just-resolved chunk's true endpoint —
                // see resolveChunk's doc. chunkAnchor is already the
                // correct (accuracy-critical) value; only what's drawn
                // eases toward it instead of snapping.
                val w = 1f - chunkBlendRemaining / chunkBlendSamples.toFloat()
                chunkBlendRemaining--
                FusionState(
                    FusionMode.BLACKOUT,
                    (1 - w) * chunkBlendFrom.x + w * chunkAnchor.x,
                    (1 - w) * chunkBlendFrom.y + w * chunkAnchor.y,
                    wrapAngle((1 - w) * chunkBlendFrom.heading + w * chunkAnchor.heading),
                    (1 - w) * chunkBlendFrom.speed + w * chunkAnchor.speed,
                )
            } else {
                // Cheap live extrapolation between chunk boundaries — see
                // class doc. Uses the anchor's own speed/heading, not the
                // network: a display-only convenience, not part of the
                // accuracy path.
                val elapsed = pending * dt
                val extrapX = chunkAnchor.x + chunkAnchor.speed * cos(chunkAnchor.heading.toDouble()).toFloat() * elapsed
                val extrapY = chunkAnchor.y + chunkAnchor.speed * sin(chunkAnchor.heading.toDouble()).toFloat() * elapsed
                FusionState(FusionMode.BLACKOUT, extrapX, extrapY, chunkAnchor.heading, chunkAnchor.speed)
            }

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
     *
     * Real bug found reasoning through a live report (a photographed
     * "the trail zigzags/goes backwards" at a real road junction, not
     * something reproducible from a screenshot alone): this integrates
     * the TRUE per-sample heading smoothly, sample by sample, as it
     * should — but only the FINAL endpoint ever used to reach `state`
     * directly. Every tick in between (see tick()'s else-branch) instead
     * *guesses* with a straight line at the PREVIOUS chunk's frozen
     * heading, because the real corrected heading for this chunk isn't
     * known until the whole chunk (and its one model call) resolves.
     * Whenever the real path actually curves within a chunk — exactly
     * what happens at a junction — that straight-line guess drifts
     * further off the true path throughout the chunk, then this function
     * used to snap `state` straight to the true endpoint: a visible kink
     * every single chunk boundary where the road curved, at 5s intervals.
     * Fixed by not setting `state` here at all — chunkAnchor (the only
     * value anything accuracy-critical, including the NEXT chunk's own
     * integration, ever reads) still becomes the true resolved endpoint
     * immediately, but tick() eases the *displayed* state toward it over
     * chunkBlendSeconds instead of jumping, same idea as the existing
     * GNSS-reconnect blend.
     */
    private fun resolveChunk() {
        val window = rollingWindow.toTypedArray()  // always windowSize once filled
        val corrections = model.predict(window)     // (windowSize, 2)
        val startIdx = window.size - pending

        var v = chunkAnchor.speed
        var th = chunkAnchor.heading
        var x = chunkAnchor.x
        var y = chunkAnchor.y
        // Second, independent safeguard against the runaway-speed bug —
        // see class doc's third ZUPT paragraph and lowSpeedVMargin's own
        // doc. Computed once from blackoutEntrySpeed (frozen, real
        // GNSS-confirmed), not from the live v — same non-self-referential
        // reasoning as blackoutEntrySpeed itself, so this can't be defeated
        // by the very runaway it exists to bound.
        val vMaxThisChunk = if (blackoutEntrySpeed < zuptMaxSpeed) {
            minOf(vClampMax, blackoutEntrySpeed + lowSpeedVMargin)
        } else vClampMax
        for (i in startIdx until window.size) {
            if (isQuiet(window[i], blackoutEntrySpeed)) {
                // ZUPT — raw sensors say the device is at rest right now;
                // don't trust the network's correction for this instant
                // (out-of-distribution for anything but real driving), just
                // decay any stale speed and hold heading. See class doc.
                v *= 0.7f
            } else {
                val forwardAccel = window[i][0] + corrections[i][0]
                val yawRate = window[i][5] + corrections[i][1]
                v = (v + forwardAccel * dt).coerceIn(vClampMin, vMaxThisChunk)
                th = wrapAngle(th + yawRate * dt)
            }
            x += v * cos(th.toDouble()).toFloat() * dt
            y += v * sin(th.toDouble()).toFloat() * dt
        }
        val resolved = FusionState(FusionMode.BLACKOUT, x, y, th, v)

        // `state` right now is wherever the last tick's straight-line
        // guess landed (see class/tick() docs) — ease the display from
        // there toward the true resolved endpoint instead of snapping.
        chunkBlendFrom = state.copy()
        chunkAnchor = resolved
        chunkBlendRemaining = chunkBlendSamples
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
     * Gating on referenceSpeed already being low fixes the ambiguity: a
     * cruising vehicle's speed is not low, so it's exempt regardless of
     * how quiet the instantaneous accel/gyro looks.
     *
     * referenceSpeed must be blackoutEntrySpeed (the last real
     * GNSS-confirmed speed), NOT the live integrated v — a second real
     * bug, found live via `adb logcat` while reproducing a reported "speed
     * jumps to 37 km/h a second into a blackout" (real device, phone
     * essentially stationary/walking-pace, GNSS reporting ~1 m/s the whole
     * time — confirmed the logged gnssSpeed never left 0.7-1.1 m/s while
     * fusedSpeed climbed unbounded, chunk after chunk, past 37 km/h and
     * well past 130 km/h before being caught, still climbing). Gating on
     * the live v instead of a frozen reference is a one-way door: the
     * model's out-of-distribution correction (same domain-mismatch this
     * whole ZUPT mechanism exists to catch — see class doc) only has to
     * push v above zuptMaxSpeed ONCE, and ZUPT permanently stops firing
     * for the rest of the blackout right when it's needed most, letting
     * every subsequent chunk's bias integrate further unchecked — a
     * positive-feedback runaway, not a one-off glitch. Freezing the
     * reference at blackoutEntrySpeed instead breaks that loop: a phone
     * that was genuinely slow when GNSS was last available stays
     * ZUPT-eligible for the whole blackout regardless of how far the
     * integration has since (wrongly) drifted, so a runaway gets caught
     * and decayed on the very next quiet sample instead of never again.
     * The genuinely-cruising-vehicle case this gate was originally added
     * for is unaffected: that vehicle's speed was already high at the
     * moment GNSS was last available, so blackoutEntrySpeed is high too. */
    private fun isQuiet(sample: FloatArray, referenceSpeed: Float): Boolean {
        if (referenceSpeed >= zuptMaxSpeed) return false
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
        chunkBlendRemaining = 0
        chunkAnchor = FusionState()
        blackoutEntrySpeed = 0f
        wasAvailable = true
    }
}
