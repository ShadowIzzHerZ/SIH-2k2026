package com.sih26168.deadreckoning

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
        // (0.3 rad/s) on out-of-distribution input — left uncorrected at
        // 5 m/s this traces a ~17m-radius circle within a few seconds
        // (confirmed live on-device as the actual reported bug).
        val engine = FusionEngine(FakePredictor(deltaV = 0f, deltaTheta = 0.3f))
        warmUpGnssTracking(engine, speed = 5f)

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
}
