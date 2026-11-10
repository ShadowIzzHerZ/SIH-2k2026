package com.sih26168.deadreckoning

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Regression test for the "circling while at rest" bug found live on a
 * real device (see FusionEngine's ZUPT class doc): during a blackout, if
 * the bias-correction network hallucinates a small constant yaw-rate bias
 * for near-zero real motion — exactly what happens feeding it input
 * wildly outside its vehicle-driving training distribution, e.g. a phone
 * just being held still — integrating that bias every tick used to spin
 * the fused position around in a full circle even though the device
 * never actually moved.
 *
 * No device, no ONNX runtime, no Android Context needed: FusionEngine's
 * only real dependency is the small Predictor interface, so a fake
 * standing in for BiasCorrectionModel can deterministically reproduce the
 * exact correction shape that caused the bug and prove it's fixed,
 * runnable with `./gradlew test` on any machine.
 */
class FusionEngineTest {

    /** Always returns the same [deltaV, deltaTheta] correction for every
     * sample in the window, regardless of input — a stand-in for the real
     * network hallucinating a constant bias on out-of-distribution input. */
    private class FakePredictor(private val deltaV: Float, private val deltaTheta: Float) : Predictor {
        override fun predict(window: Array<FloatArray>): Array<FloatArray> =
            Array(window.size) { floatArrayOf(deltaV, deltaTheta) }
    }

    private val windowSize = 50 // must match FusionEngine's default

    /** 50+ ticks of real GNSS tracking at a fixed speed — fills the
     * rolling window and gives BLACKOUT a nonzero chunkAnchor.speed to
     * start from, same as a real device that was actually moving right
     * before GNSS cut out. */
    private fun warmUpGnssTracking(engine: FusionEngine, speed: Float) {
        repeat(windowSize + 5) {
            engine.tick(
                calibratedAccel = floatArrayOf(0f, 0f, 0f),
                calibratedGyro = floatArrayOf(0f, 0f, 0f),
                available = true,
                lat = 0.0, lon = 0.0,
                gnssSpeed = speed, gnssHeadingRad = 0f,
            )
        }
    }

    private fun distanceFromOrigin(engine: FusionEngine): Double =
        sqrt((engine.state.x * engine.state.x + engine.state.y * engine.state.y).toDouble())

    @Test
    fun `blackout with a stationary device and a noisy correction does not spin into a circle`() {
        // A fake network hallucinating a modest constant yaw-rate bias
        // (0.3 rad/s) on out-of-distribution input — left uncorrected this
        // traces a wide circle within a few seconds (confirmed live
        // on-device as the actual reported bug). 2 m/s: a brisk walk,
        // realistically what GNSS was last reporting right before someone
        // holding the phone stood still — comfortably under zuptMaxSpeed.
        val engine = FusionEngine(FakePredictor(deltaV = 0f, deltaTheta = 0.3f))
        warmUpGnssTracking(engine, speed = 2f)

        // Blackout begins with the device genuinely at rest: raw
        // (pre-correction) accel/gyro both ~zero.
        repeat(windowSize * 5) { // 5 chunk resolutions, 25s of simulated blackout
            engine.tick(
                calibratedAccel = floatArrayOf(0f, 0f, 0f),
                calibratedGyro = floatArrayOf(0f, 0f, 0f),
                available = false,
                lat = 0.0, lon = 0.0, gnssSpeed = 0f, gnssHeadingRad = 0f,
            )
        }

        val dist = distanceFromOrigin(engine)
        assertTrue(
            "device was at rest the whole blackout but drifted ${dist}m from where it started " +
                "(x=${engine.state.x}, y=${engine.state.y}) — ZUPT should have held position",
            dist < 5.0,
        )
    }

    @Test
    fun `blackout with a genuinely parked device and real noisy sensor data does not creep`() {
        // Regression for a real bug reported live during real-device
        // testing: "the line keeps moving even when we stop moving." The
        // first "stationary device" test above uses *exactly* zero
        // accel/gyro, which trivially passes isQuiet's threshold — not
        // representative of a real device's actual sensor noise. This test
        // reproduces the real failure shape instead: accel/gyro that FAILS
        // isQuiet's threshold (quietAccelThresh=0.35, quietGyroThresh=0.05
        // by default), plus a small constant network correction, on a
        // device that was genuinely parked (entry speed 0) before the
        // blackout — exactly the case FusionEngine's fourth ZUPT rule
        // (zuptForceRestSpeed) exists for. Without that rule, v climbs the
        // lowSpeedVMargin ceiling (4 m/s) and holds there for the whole
        // blackout — a real, sustained ~100m creep over 25s, not a small
        // rounding error.
        val engine = FusionEngine(FakePredictor(deltaV = 0.75f, deltaTheta = 0f))
        warmUpGnssTracking(engine, speed = 0f)

        repeat(windowSize * 5) { // 5 chunk resolutions, 25s of simulated blackout
            engine.tick(
                // Horizontal accel magnitude ~3.37 m/s^2 — comfortably
                // above quietAccelThresh, the exact real-device magnitude
                // logged chasing this bug live (see FusionEngine's ZUPT
                // class doc, fourth paragraph).
                calibratedAccel = floatArrayOf(3.37f, 0f, 0f),
                calibratedGyro = floatArrayOf(0f, 0f, 0f),
                available = false,
                lat = 0.0, lon = 0.0, gnssSpeed = 0f, gnssHeadingRad = 0f,
            )
        }

        val dist = distanceFromOrigin(engine)
        assertTrue(
            "device was genuinely parked the whole blackout but drifted ${dist}m from where it started " +
                "(x=${engine.state.x}, y=${engine.state.y}) — the fourth ZUPT rule should have held position " +
                "regardless of the noisy accel reading",
            dist < 5.0,
        )
    }

    @Test
    fun `blackout with genuine motion still integrates corrections normally`() {
        // Same shape of fake network, but this time the raw samples look
        // like real driving (clearly non-quiet accel/gyro) — ZUPT must NOT
        // suppress this, or real blackout dead reckoning (the validated
        // 2.9%-drift replay path) would silently stop working.
        val engine = FusionEngine(FakePredictor(deltaV = 0f, deltaTheta = 0.1f))
        warmUpGnssTracking(engine, speed = 10f)

        repeat(windowSize) { // one full chunk, 5s
            engine.tick(
                calibratedAccel = floatArrayOf(1.0f, 0f, 0f), // clearly non-quiet forward accel
                calibratedGyro = floatArrayOf(0f, 0f, 0.05f), // clearly non-quiet yaw rate
                available = false,
                lat = 0.0, lon = 0.0, gnssSpeed = 0f, gnssHeadingRad = 0f,
            )
        }

        val dist = distanceFromOrigin(engine)
        assertTrue(
            "genuine in-motion blackout should still displace the fused position substantially, got ${dist}m",
            dist > 20.0,
        )
    }

    @Test
    fun `blackout with a vehicle cruising at constant speed does not get ZUPT'd to a stop`() {
        // Regression for a second real bug, found live via the replay
        // demo: real ~110 km/h highway cruising (comma2k19, verified
        // against the raw recorded data) has instantaneous accel/yaw-rate
        // small enough to look "quiet" by magnitude alone — Newton's
        // first law means constant velocity also reads as ~zero net
        // force, same as genuinely being at rest. The first ZUPT version
        // decayed a real 30 m/s cruise toward zero and the fused trail
        // visibly reversed course mid-cruise. These numbers are the real
        // ones measured from the bundled replay_drive.json at the point
        // this happened: speed ~31 m/s, gz staying under 0.06 rad/s.
        val engine = FusionEngine(FakePredictor(deltaV = 0f, deltaTheta = 0f))
        warmUpGnssTracking(engine, speed = 31f)

        repeat(windowSize * 3) { // 15s of "quiet-looking" but genuinely fast cruising
            engine.tick(
                calibratedAccel = floatArrayOf(0.2f, 0.1f, 0f), // quiet by the old accel-only check
                calibratedGyro = floatArrayOf(0f, 0f, 0.04f),   // quiet by the old gyro-only check too
                available = false,
                lat = 0.0, lon = 0.0, gnssSpeed = 0f, gnssHeadingRad = 0f,
            )
        }

        assertTrue(
            "a vehicle genuinely cruising at ~31 m/s must not have its speed decayed toward zero " +
                "just because the instantaneous reading looks quiet — got ${engine.state.speed} m/s",
            engine.state.speed > 20f,
        )
    }

    private fun angleDiff(a: Float, b: Float): Float {
        val d = atan2(sin((a - b).toDouble()), cos((a - b).toDouble()))
        return abs(d).toFloat()
    }

    @Test
    fun `a real turn within one chunk doesn't snap into a single-tick kink`() {
        // Regression for a real bug found reasoning through a live report
        // (photos of the trail visibly zigzagging at a real road
        // junction) — see resolveChunk's doc. Between chunk boundaries,
        // the display extrapolates in a straight line at the PREVIOUS
        // chunk's frozen heading (the real corrected heading for the
        // current chunk isn't known until the whole chunk resolves).
        // Before the fix, the instant a chunk resolved, `state` snapped
        // straight to the true endpoint — however much the heading
        // actually changed within that chunk, in one single tick. A real
        // turn at a junction (curving substantially within one 5s chunk)
        // produced a visible kink every single chunk boundary.
        val turnRateCorrection = 0.2f // rad/s
        val engine = FusionEngine(FakePredictor(deltaV = 0f, deltaTheta = turnRateCorrection))
        warmUpGnssTracking(engine, speed = 15f) // chunkAnchor.heading = 0 after this

        val headings = mutableListOf(engine.state.heading)
        repeat(windowSize + 10) { // one full chunk + enough ticks to see the whole catch-up finish
            engine.tick(
                calibratedAccel = floatArrayOf(1.0f, 0f, 0f), // clearly non-quiet — a real turn, not ZUPT
                calibratedGyro = floatArrayOf(0f, 0f, 0.1f),  // + the 0.2 correction = 0.3 rad/s true turn rate
                available = false,
                lat = 0.0, lon = 0.0, gnssSpeed = 0f, gnssHeadingRad = 0f,
            )
            headings.add(engine.state.heading)
        }

        val maxTickDelta = headings.zipWithNext { a, b -> angleDiff(b, a) }.max()
        // The true total turn across the chunk is ~0.3 rad/s * 5s = 1.5
        // rad. A single-tick snap (the bug) would show ~1.5 rad in one
        // step; spread across the blend it should be a small fraction of
        // that — 0.5 rad is a generous bound, still far below a real snap.
        assertTrue(
            "a single tick's heading changed by $maxTickDelta rad — that's a snap, not a smooth catch-up " +
                "(full headings: $headings)",
            maxTickDelta < 0.5f,
        )

        // The blend has to actually finish and reach the true value, not
        // just avoid the snap — final heading should be close to the real
        // integrated turn (~1.5 rad), not stuck partway.
        val expectedFinalHeading = 0.3f * (windowSize * 0.1f)
        assertTrue(
            "final heading ${engine.state.heading} should have caught up to the true turn " +
                "($expectedFinalHeading rad) well after the blend window",
            angleDiff(engine.state.heading, expectedFinalHeading) < 0.1f,
        )
    }

    /** Regression for a real bug found live: sitting still indoors with a
     * rough network fix, Location.getBearing() is either flagged invalid
     * (hasBearing()==false) or a stale/default value the provider never
     * cleared — but GNSS_TRACKING used to snap `state.heading` straight to
     * it every tick regardless. See FusionEngine.tick's minHeadingLockSpeedMps
     * doc. */
    @Test
    fun `a low-speed or bearing-less fix does not snap heading to a bogus bearing`() {
        val engine = FusionEngine(FakePredictor(deltaV = 0f, deltaTheta = 0f))

        // Real, fast, bearing-valid fix — heading should lock straight to it.
        val realHeading = 1.0f // rad
        engine.tick(
            calibratedAccel = floatArrayOf(0f, 0f, 0f),
            calibratedGyro = floatArrayOf(0f, 0f, 0f),
            available = true,
            lat = 0.0, lon = 0.0, gnssSpeed = 10f, gnssHeadingRad = realHeading,
            gnssHasBearing = true,
        )
        assertTrue(
            "heading should have locked to the real fix's bearing, got ${engine.state.heading}",
            angleDiff(engine.state.heading, realHeading) < 0.01f,
        )

        // Same real fix location/speed, but no valid bearing this tick (a
        // bogus far-away heading stands in for whatever garbage the
        // provider would hand back) — heading must hold, not snap.
        val bogusHeading = 3.0f
        engine.tick(
            calibratedAccel = floatArrayOf(0f, 0f, 0f),
            calibratedGyro = floatArrayOf(0f, 0f, 0f),
            available = true,
            lat = 0.0, lon = 0.0, gnssSpeed = 10f, gnssHeadingRad = bogusHeading,
            gnssHasBearing = false,
        )
        assertTrue(
            "a bearing-less fix snapped heading to $bogusHeading instead of holding ~$realHeading",
            angleDiff(engine.state.heading, realHeading) < 0.01f,
        )

        // Same fix, this time bearing IS flagged valid but speed is too low
        // for it to be meaningful — must still hold, not snap.
        engine.tick(
            calibratedAccel = floatArrayOf(0f, 0f, 0f),
            calibratedGyro = floatArrayOf(0f, 0f, 0f),
            available = true,
            lat = 0.0, lon = 0.0, gnssSpeed = 0.2f, gnssHeadingRad = bogusHeading,
            gnssHasBearing = true,
        )
        assertTrue(
            "a near-stationary fix snapped heading to $bogusHeading instead of holding ~$realHeading",
            angleDiff(engine.state.heading, realHeading) < 0.01f,
        )
    }

    /** The GNSS-reconnect blend has to take the short way round the
     * circle. Blending linearly in raw radians from just under +pi to just
     * over -pi (two headings 0.28 rad apart, both pointing west) swept
     * through 0 — the displayed heading spun to due east mid-blend. */
    @Test
    fun `reconnect blend across the plus-minus pi seam stays pointing the same way`() {
        val engine = FusionEngine(FakePredictor(deltaV = 0f, deltaTheta = 0f))
        val westish = 3.0f
        repeat(windowSize + 5) {
            engine.tick(
                calibratedAccel = floatArrayOf(0f, 0f, 0f),
                calibratedGyro = floatArrayOf(0f, 0f, 0f),
                available = true,
                lat = 0.0, lon = 0.0, gnssSpeed = 10f, gnssHeadingRad = westish,
            )
        }
        repeat(10) {
            engine.tick(
                calibratedAccel = floatArrayOf(0f, 0f, 0f),
                calibratedGyro = floatArrayOf(0f, 0f, 0f),
                available = false,
                lat = 0.0, lon = 0.0, gnssSpeed = 10f, gnssHeadingRad = westish,
            )
        }
        val otherSideOfSeam = -3.0f
        repeat(25) {
            engine.tick(
                calibratedAccel = floatArrayOf(0f, 0f, 0f),
                calibratedGyro = floatArrayOf(0f, 0f, 0f),
                available = true,
                lat = 0.0, lon = 0.0, gnssSpeed = 10f, gnssHeadingRad = otherSideOfSeam,
            )
            assertTrue(
                "blend heading ${engine.state.heading} left the short arc between $westish and $otherSideOfSeam",
                angleDiff(engine.state.heading, Math.PI.toFloat()) <= 0.3f,
            )
        }
    }

    /** Records every window handed to the model, so a test can check which
     * samples a chunk resolution actually integrated. */
    private class RecordingPredictor : Predictor {
        val windows = ArrayList<Array<FloatArray>>()
        override fun predict(window: Array<FloatArray>): Array<FloatArray> {
            windows.add(window.map { it.copyOf() }.toTypedArray())
            return Array(window.size) { floatArrayOf(0f, 0f) }
        }
    }

    /** A short blackout resolved when GNSS returns must integrate exactly
     * the blackout's own samples — the same chunk fusion.py's run_fusion
     * builds (accel[t:t + chunk_len], stopping before the first available
     * sample). The window's last `pending` rows used to include the
     * reconnect tick's own (non-blackout) sample and drop the first
     * blackout sample. */
    @Test
    fun `a short blackout resolved on GNSS return integrates only its own samples`() {
        val predictor = RecordingPredictor()
        val engine = FusionEngine(predictor)
        // Tag every sample with its tick number in az so the window rows
        // can be identified (az is never read by the integration).
        var tickNo = 0
        fun tick(available: Boolean) {
            engine.tick(
                calibratedAccel = floatArrayOf(0f, 0f, tickNo.toFloat()),
                calibratedGyro = floatArrayOf(0f, 0f, 0f),
                available = available,
                lat = 0.0, lon = 0.0, gnssSpeed = 10f, gnssHeadingRad = 0f,
            )
            tickNo++
        }
        repeat(windowSize + 5) { tick(available = true) }
        val firstBlackoutTick = tickNo
        val blackoutTicks = 20
        repeat(blackoutTicks) { tick(available = false) }
        val reconnectTick = tickNo
        tick(available = true)

        assertTrue("expected exactly one chunk resolution, got ${predictor.windows.size}", predictor.windows.size == 1)
        val window = predictor.windows[0]
        assertTrue("model must always get a full window, got ${window.size}", window.size == windowSize)
        val harvested = window.takeLast(blackoutTicks).map { it[2].toInt() }
        assertTrue(
            "harvested ticks $harvested should be exactly the blackout ticks " +
                "$firstBlackoutTick..${reconnectTick - 1}",
            harvested == (firstBlackoutTick until reconnectTick).toList(),
        )
    }

    // ---- static hold (GNSS available, device not moving) ----

    private val metresPerDegLat = 111_194.9 // 6371000 m * pi / 180

    /** One GNSS-available tick at [northM] metres north of the origin. */
    private fun fixTick(
        engine: FusionEngine, northM: Double, speed: Float, stationary: Boolean, accuracyM: Float = 0f,
    ) {
        engine.tick(
            calibratedAccel = floatArrayOf(0f, 0f, 0f),
            calibratedGyro = floatArrayOf(0f, 0f, 0f),
            available = true,
            lat = northM / metresPerDegLat, lon = 0.0,
            gnssSpeed = speed, gnssHeadingRad = 0f,
            stationary = stationary, gnssAccuracyM = accuracyM,
        )
    }

    /** The wandering dot: a phone at rest whose fixes scatter by a few
     * metres. Without the hold every fix moved the position. */
    @Test
    fun `stationary fixes scattering inside the hold radius do not move the position`() {
        val engine = FusionEngine(FakePredictor(0f, 0f))
        fixTick(engine, 0.0, speed = 0f, stationary = true, accuracyM = 5f)
        val scatter = doubleArrayOf(4.0, -6.0, 8.0, -3.0, 7.5, -7.0, 2.0)
        for (m in scatter) {
            fixTick(engine, m, speed = 0f, stationary = true, accuracyM = 5f)
            assertTrue("position moved to ${engine.state.y} m on a stationary fix at $m m", abs(engine.state.y) < 0.01f)
        }
    }

    @Test
    fun `without the stationary flag every fix is followed as before`() {
        val engine = FusionEngine(FakePredictor(0f, 0f))
        fixTick(engine, 0.0, speed = 0f, stationary = false)
        fixTick(engine, 6.0, speed = 0f, stationary = false)
        assertTrue("expected the raw fix at 6 m, got ${engine.state.y}", abs(engine.state.y - 6f) < 0.01f)
    }

    @Test
    fun `a stationary fix outside the hold radius moves the hold to it`() {
        val engine = FusionEngine(FakePredictor(0f, 0f))
        fixTick(engine, 0.0, speed = 0f, stationary = true, accuracyM = 5f)
        fixTick(engine, 40.0, speed = 0f, stationary = true, accuracyM = 5f)
        assertTrue("expected the hold to move to 40 m, got ${engine.state.y}", abs(engine.state.y - 40f) < 0.01f)
        fixTick(engine, 44.0, speed = 0f, stationary = true, accuracyM = 5f)
        assertTrue("expected the new hold to stay at 40 m, got ${engine.state.y}", abs(engine.state.y - 40f) < 0.01f)
    }

    /** A cold start: the first fix is 50 m accurate and off, the next is
     * 8 m accurate. Holding the first one would pin the marker on the
     * worse position for as long as the phone stays put. */
    @Test
    fun `a clearly more accurate stationary fix replaces the held one`() {
        val engine = FusionEngine(FakePredictor(0f, 0f))
        fixTick(engine, 0.0, speed = 0f, stationary = true, accuracyM = 50f)
        fixTick(engine, 30.0, speed = 0f, stationary = true, accuracyM = 8f)
        assertTrue("expected the better fix at 30 m, got ${engine.state.y}", abs(engine.state.y - 30f) < 0.01f)
    }

    @Test
    fun `moving off ends the hold and follows the fixes again`() {
        val engine = FusionEngine(FakePredictor(0f, 0f))
        fixTick(engine, 0.0, speed = 0f, stationary = true, accuracyM = 5f)
        fixTick(engine, 3.0, speed = 0f, stationary = true, accuracyM = 5f)
        fixTick(engine, 4.0, speed = 2f, stationary = false, accuracyM = 5f)
        assertTrue("expected the raw fix at 4 m once moving, got ${engine.state.y}", abs(engine.state.y - 4f) < 0.01f)
    }

    /** Hold, then lose GNSS: the blackout starts from the held position at
     * speed 0, so the forced-rest ZUPT keeps it there whatever the network
     * says. */
    @Test
    fun `a blackout that starts from a static hold stays on the held position`() {
        val engine = FusionEngine(FakePredictor(deltaV = 1.0f, deltaTheta = 0.2f))
        repeat(windowSize + 5) { i ->
            fixTick(engine, if (i % 2 == 0) 5.0 else -5.0, speed = 0f, stationary = true, accuracyM = 5f)
        }
        val heldY = engine.state.y
        repeat(3 * windowSize) {
            engine.tick(
                calibratedAccel = floatArrayOf(0.8f, 0.3f, 0f),
                calibratedGyro = floatArrayOf(0f, 0f, 0.1f),
                available = false,
                lat = 0.0, lon = 0.0, gnssSpeed = 0f, gnssHeadingRad = 0f,
            )
        }
        val moved = sqrt(((engine.state.y - heldY) * (engine.state.y - heldY) + engine.state.x * engine.state.x).toDouble())
        assertTrue("drifted $moved m during a blackout entered at rest", moved < 0.5)
    }
}
