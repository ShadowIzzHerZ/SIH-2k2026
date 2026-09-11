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
}
