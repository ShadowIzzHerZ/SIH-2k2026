package com.sih26168.deadreckoning

import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests the two live-calibration gates (see CalibrationManager's class
 * doc): leveling waiting for a genuinely stationary window, and yaw
 * lock-in waiting for checkpoint estimates to agree. Both exist because a bad
 * live calibration is silent — it doesn't crash or look wrong, it just
 * tilts every downstream number for the rest of the trip — so the gates
 * themselves are the only thing standing between "held the phone wrong
 * for two seconds" and a whole demo run of quietly wrong output.
 *
 * Pure JVM, no device: Calibration/CalibrationManager are plain Kotlin
 * math with no Android dependency.
 */
class CalibrationManagerTest {

    private val g = 9.81f
    private val rng = Random(26168)

    /** A resting phone lying flat face-up. Android's TYPE_ACCELEROMETER
     * measures proper acceleration, so this reads +9.81 on z (the normal
     * force), not -9.81 — getting this backwards makes the leveling
     * rotation a 180-degree flip instead of an identity, which silently
     * mirrors the yaw math. Real resting noise is ~0.02-0.1 m/s^2. */
    private fun restingSample(noise: Float = 0.03f) = floatArrayOf(
        rng.nextFloat().minus(0.5f) * 2 * noise,
        rng.nextFloat().minus(0.5f) * 2 * noise,
        g + rng.nextFloat().minus(0.5f) * 2 * noise,
    )

    /** A phone being carried/waved around — magnitude swinging well away
     * from a steady 1g, the thing the stationarity gate must reject. */
    private fun movingSample() = floatArrayOf(
        rng.nextFloat().minus(0.5f) * 8f,
        rng.nextFloat().minus(0.5f) * 8f,
        g + rng.nextFloat().minus(0.5f) * 8f,
    )

    @Test
    fun `leveling does not commit while the phone is still being moved`() {
        val cal = CalibrationManager()
        repeat(40) { cal.addLevelSample(movingSample()) }

        assertFalse(
            "leveling committed to a gravity estimate from samples taken while the phone was moving — " +
                "that rotation would then skew every calibrated sample for the whole session",
            cal.isLeveled,
        )
        assertTrue(
            "UI should be able to say specifically that it's waiting on the user to hold still",
            cal.levelWaitingForStillness,
        )
    }

    @Test
    fun `leveling commits once the phone actually settles`() {
        val cal = CalibrationManager()
        // Fumbling with the phone first, then setting it down.
        repeat(40) { cal.addLevelSample(movingSample()) }
        assertFalse(cal.isLeveled)

        repeat(25) { cal.addLevelSample(restingSample()) }

        assertTrue("leveling should commit once a genuinely still window arrives", cal.isLeveled)
        assertTrue("a still window should count as converged, not best-effort", cal.levelConverged)
    }

    /** A phone mounted in a car that's already driving: orientation is
     * rock steady, but |accel| swings by m/s^2 as it accelerates, brakes
     * and corners. Leveling is perfectly estimable here — gravity's
     * direction in the phone frame never moves. */
    private fun drivingSample() = floatArrayOf(
        rng.nextFloat().minus(0.5f) * 4f,   // longitudinal accel/braking
        rng.nextFloat().minus(0.5f) * 3f,   // cornering
        g + rng.nextFloat().minus(0.5f) * 1f,
    )

    @Test
    fun `leveling works for a phone already mounted in a moving vehicle`() {
        // Regression: gating on |accel| stability instead of orientation
        // stability made the replay demo (a real car already at speed)
        // wait out the full timeout and self-flag as degraded, on a
        // window that was perfectly good. Caught on a real device.
        val cal = CalibrationManager()
        repeat(25) { cal.addLevelSample(drivingSample()) }

        assertTrue("a steadily-mounted phone in a moving car must still level", cal.isLeveled)
        assertTrue("...and count as converged, not a degraded fallback", cal.levelConverged)
    }

    @Test
    fun `leveling falls back to best-effort rather than stalling forever`() {
        // A phone that never settles — e.g. handheld the whole time, or
        // heavy vibration through a mount. A demo must not hang on this.
        val cal = CalibrationManager(maxLevelSamples = 60)
        repeat(60) { cal.addLevelSample(movingSample()) }

        assertTrue("leveling must give up and commit rather than stall the demo", cal.isLeveled)
        assertFalse("...but must report itself as a degraded estimate", cal.levelConverged)
    }

    /** A car speeding up, braking and weaving, with the course starting
     * at `course0`. Kept across feedYaw calls so the GPS speed/course it
     * reports stays continuous. */
    private class Drive(course0: Double, val dt: Double = 0.1) {
        var t = 0.0
        var v = 12.0
        var crs = course0
    }

    /** Feeds yaw samples from `drive`, with the leveled phone frame rotated
     * by `trueYawOffset` from the vehicle frame (yawRotation(trueYawOffset)
     * takes phone to vehicle). */
    private fun feedYaw(cal: CalibrationManager, n: Int, drive: Drive, trueYawOffset: Float, noise: Float = 0.05f) {
        val c = cos(trueYawOffset.toDouble())
        val s = sin(trueYawOffset.toDouble())
        repeat(n) {
            val aLon = 1.5 * sin(2 * Math.PI * drive.t / 6.0)
            val yawRate = 0.15 * sin(2 * Math.PI * drive.t / 5.0)
            drive.v += aLon * drive.dt
            drive.crs += yawRate * drive.dt
            drive.t += drive.dt
            val aLat = drive.v * yawRate
            // Vehicle-frame [forward, left] accel expressed in the phone frame.
            val ax = (c * aLon + s * aLat).toFloat() + rng.nextFloat().minus(0.5f) * noise
            val ay = (-s * aLon + c * aLat).toFloat() + rng.nextFloat().minus(0.5f) * noise
            cal.addYawSample(floatArrayOf(ax, ay, g), gpsSpeed = drive.v.toFloat(), gpsHeadingRad = drive.crs.toFloat())
        }
    }

    private fun leveledManager(
        minConfidentSamples: Int = 30,
        maxConfidentSamples: Int = 90,
        yawWindowSamples: Int = 60,
    ): CalibrationManager {
        val cal = CalibrationManager(
            minConfidentSamples = minConfidentSamples,
            maxConfidentSamples = maxConfidentSamples,
            yawWindowSamples = yawWindowSamples,
        )
        repeat(25) { cal.addLevelSample(restingSample()) }
        require(cal.isLeveled)
        return cal
    }

    @Test
    fun `yaw locks in on a consistent straight-line stretch`() {
        val cal = leveledManager()
        val trueOffset = 0.6f  // phone mounted ~34 degrees off the vehicle's forward axis

        feedYaw(cal, n = 60, Drive(course0 = 1.0), trueYawOffset = trueOffset)

        assertTrue("a clean straight-line stretch should lock yaw in", cal.isReady)
        assertTrue("...and report itself converged", cal.yawConverged)
        assertEquals("recovered yaw offset should match the one baked into the samples", trueOffset, cal.psiYaw, 0.15f)
    }

    @Test
    fun `the locked-in yaw does not depend on the course driven`() {
        // Regression: the estimate used to compare the leveled phone-frame
        // accel angle with the world-frame GPS course, returning the mount
        // offset plus the course instead of the mount offset alone.
        for (course in listOf(-2.5, 0.0, 1.6, 3.0)) {
            val cal = leveledManager()
            feedYaw(cal, n = 90, Drive(course0 = course), trueYawOffset = 0.6f)
            assertTrue(cal.isReady)
            assertTrue(cal.yawConverged)
            assertEquals("course $course", 0.6f, cal.psiYaw, 0.1f)
        }
    }

    @Test
    fun `an early inconsistent stretch delays lock-in past the naive early window`() {
        // The phone shifts in its mount early in the collection window. A
        // fixed minConfidentSamples cutoff would lock in on the mixture;
        // settledness should make it wait for the later stretch to
        // actually settle instead.
        val cal = leveledManager(minConfidentSamples = 20, maxConfidentSamples = 200, yawWindowSamples = 25)
        val drive = Drive(course0 = 0.2)
        feedYaw(cal, n = 20, drive, trueYawOffset = 0.6f)  // would satisfy minConfidentSamples alone
        feedYaw(cal, n = 90, drive, trueYawOffset = -0.9f) // enough of a stretch to fully flush the early window

        assertTrue(cal.isReady)
        assertEquals(
            "should settle on the later stretch's true offset, not something blended with the early value",
            -0.9f, cal.psiYaw, 0.2f,
        )
    }

    @Test
    fun `yaw falls back to best-effort at the safety cap rather than stalling`() {
        val cal = leveledManager(minConfidentSamples = 20, maxConfidentSamples = 50)
        // A turn that doesn't resolve into any lasting straight stretch
        // before the safety cap — must still commit to something rather
        // than stall the demo, but must own up to being degraded.
        val drive = Drive(course0 = 0.2)
        feedYaw(cal, n = 20, drive, trueYawOffset = 0.6f)
        feedYaw(cal, n = 20, drive, trueYawOffset = -0.3f)
        feedYaw(cal, n = 20, drive, trueYawOffset = 0.9f)

        assertTrue("yaw must eventually commit rather than stall the demo forever", cal.isReady)
        assertFalse("...but must report itself as a degraded estimate", cal.yawConverged)
        assertTrue("degraded calibration should be surfaceable to the UI", cal.isRoughCalibration)
    }

    @Test
    fun `a clean calibration is not flagged as rough`() {
        val cal = leveledManager()
        feedYaw(cal, n = 60, Drive(course0 = 1.0), trueYawOffset = 0.6f)

        assertTrue(cal.isReady)
        assertFalse("a fully converged calibration must not be labelled rough", cal.isRoughCalibration)
    }

    @Test
    fun `steady cruising alone does not lock in a converged yaw`() {
        // Constant speed on a straight road: GPS implies no accel to match
        // the measured accel against, so there is nothing to estimate from.
        val cal = leveledManager(minConfidentSamples = 20, maxConfidentSamples = 60)
        repeat(60) {
            cal.addYawSample(floatArrayOf(0.4f, -0.3f, g), gpsSpeed = 15f, gpsHeadingRad = 1.0f)
        }
        assertTrue("must still commit at the safety cap", cal.isReady)
        assertFalse("...but not as a converged estimate", cal.yawConverged)
        assertEquals(0f, cal.psiYaw, 0f)
    }

    @Test
    fun `yaw ignores samples below the confident speed threshold`() {
        val cal = leveledManager()
        // Stopped at a light: GPS heading is meaningless, accel is noise.
        repeat(100) {
            cal.addYawSample(floatArrayOf(0.01f, 0.01f, g), gpsSpeed = 0.3f, gpsHeadingRad = 2.0f)
        }
        assertFalse("stationary samples must not count toward yaw alignment", cal.isReady)
    }
}
