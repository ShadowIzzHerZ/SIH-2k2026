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
 * lock-in waiting for two estimates to agree. Both exist because a bad
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

    /**
     * Feeds a physically consistent drive to the yaw stage: the vehicle's
     * speed follows [speedAt] and it turns at [yawRateAt]; the phone lies
     * flat (so leveling is the identity) rotated [trueYawOffset] from the
     * vehicle's forward axis. Leveled horizontal accel is then the
     * vehicle-frame [dv/dt, v * yaw_rate] rotated back by the offset —
     * exactly the relation Calibration.estimateForwardAxis fits.
     */
    private fun feedDrive(
        cal: CalibrationManager,
        n: Int,
        trueYawOffset: Float,
        startTick: Int = 0,
        noise: Float = 0.1f,
        speedAt: (Int) -> Float = { t -> 12f + 4f * sin(t * 0.05).toFloat() },
        yawRateAt: (Int) -> Float = { t -> 0.15f * sin(t * 0.021).toFloat() },
    ) {
        val dt = 0.1f
        for (i in 0 until n) {
            val t = startTick + i
            val v = speedAt(t)
            val lon = (speedAt(t + 1) - speedAt(t - 1)) / (2 * dt)
            val wz = yawRateAt(t)
            val lat = v * wz
            // vehicle = Rz(offset) * leveled  =>  leveled = Rz(-offset) * vehicle
            val c = cos(-trueYawOffset.toDouble()).toFloat()
            val sn = sin(-trueYawOffset.toDouble()).toFloat()
            val ax = c * lon - sn * lat + rng.nextFloat().minus(0.5f) * 2 * noise
            val ay = sn * lon + c * lat + rng.nextFloat().minus(0.5f) * 2 * noise
            cal.addYawSample(floatArrayOf(ax, ay, g), floatArrayOf(0f, 0f, wz), v)
        }
    }

    private fun leveledManager(
        maxYawSamples: Int = 600,
        maxYawWaitTicks: Int = 100_000,
    ): CalibrationManager {
        val cal = CalibrationManager(maxYawSamples = maxYawSamples, maxYawWaitTicks = maxYawWaitTicks)
        repeat(25) { cal.addLevelSample(restingSample()) }
        require(cal.isLeveled)
        return cal
    }

    private fun angleDiff(a: Float, b: Float): Float {
        var d = (a - b).toDouble()
        while (d > Math.PI) d -= 2 * Math.PI
        while (d < -Math.PI) d += 2 * Math.PI
        return kotlin.math.abs(d).toFloat()
    }

    @Test
    fun `yaw recovers the mounting offset from speed changes and turns`() {
        for (offset in listOf(0f, 0.6f, -1.2f, 2.5f, 3.1f, -2.9f)) {
            val cal = leveledManager()
            feedDrive(cal, n = 400, trueYawOffset = offset)
            assertTrue("offset $offset: a real drive should lock yaw in", cal.isReady)
            assertTrue("offset $offset: ...and report it converged", cal.yawConverged)
            assertTrue(
                "offset $offset: recovered ${cal.psiYaw}",
                angleDiff(offset, cal.psiYaw) < 0.05f,
            )
        }
    }

    @Test
    fun `yaw does not depend on which way the vehicle is heading`() {
        // The old estimate compared phone-frame accel with compass
        // headings, so the same mounting gave a different answer on a
        // north-bound and an east-bound road. The new one never reads a
        // heading at all, so the same drive in any direction gives the
        // same psi by construction; this pins that the API has no heading.
        val a = leveledManager(); feedDrive(a, n = 300, trueYawOffset = 0.8f)
        val b = leveledManager(); feedDrive(b, n = 300, trueYawOffset = 0.8f)
        assertEquals(a.psiYaw, b.psiYaw, 0.05f)
    }

    @Test
    fun `yaw needs the offline fit's 100 usable samples before locking`() {
        val cal = leveledManager()
        feedDrive(cal, n = 90, trueYawOffset = 0.6f)
        assertFalse(cal.isReady)
        assertTrue(cal.progressPercent in 50..99)
    }

    @Test
    fun `a steady straight cruise does not lock in on noise`() {
        // Nothing to fit: constant speed, no turning, only sensor noise.
        val cal = leveledManager(maxYawSamples = 300)
        feedDrive(cal, n = 250, trueYawOffset = 0.6f, noise = 0.3f, speedAt = { 20f }, yawRateAt = { 0f })
        assertFalse("pure noise must not be accepted as a converged forward axis", cal.isReady)
        feedDrive(cal, n = 100, trueYawOffset = 0.6f, noise = 0.3f, speedAt = { 20f }, yawRateAt = { 0f })
        assertTrue("...but the safety cap still ends calibration", cal.isReady)
        assertFalse(cal.yawConverged)
        assertTrue(cal.isRoughCalibration)
        assertTrue("an incoherent fit is not trusted at all", cal.yawFit!!.coherence < 0.5f)
    }

    @Test
    fun `a walk finishes calibration after the wait cap instead of stalling`() {
        // Every bundled demo replay is a walk under 2 m/s: no sample ever
        // counts toward the forward axis.
        val cal = leveledManager(maxYawWaitTicks = 300)
        feedDrive(cal, n = 299, trueYawOffset = 0.6f, speedAt = { 1.3f })
        assertFalse(cal.isReady)
        feedDrive(cal, n = 1, trueYawOffset = 0.6f, speedAt = { 1.3f })
        assertTrue("calibration must not sit at 'calibrating' forever on a walk", cal.isReady)
        assertEquals(0f, cal.psiYaw)
        assertTrue(cal.isRoughCalibration)
    }

    @Test
    fun `ticks without GNSS count toward the wait cap too`() {
        val cal = leveledManager(maxYawWaitTicks = 50)
        repeat(50) { cal.yawGnssLost() }
        assertTrue(cal.isReady)
        assertTrue(cal.isRoughCalibration)
    }

    @Test
    fun `after a rough start, a real drive refines yaw and clears the rough flag`() {
        val cal = leveledManager(maxYawWaitTicks = 300)
        feedDrive(cal, n = 300, trueYawOffset = -1.0f, speedAt = { 1.3f })  // walk out to the car
        assertTrue(cal.isReady && cal.isRoughCalibration)
        feedDrive(cal, n = 400, trueYawOffset = -1.0f, startTick = 300)
        assertTrue("psi ${cal.psiYaw} should follow the drive", angleDiff(-1.0f, cal.psiYaw) < 0.05f)
        assertFalse("a coherent whole-drive estimate upgrades the calibration", cal.isRoughCalibration)
    }

    @Test
    fun `yaw keeps refining after lock-in`() {
        val cal = leveledManager()
        // Lock in on a short, noisy start, then keep driving cleanly.
        feedDrive(cal, n = 140, trueYawOffset = 1.4f, noise = 0.6f)
        assertTrue(cal.isReady)
        val early = angleDiff(1.4f, cal.psiYaw)
        feedDrive(cal, n = 2000, trueYawOffset = 1.4f, startTick = 140, noise = 0.1f)
        val late = angleDiff(1.4f, cal.psiYaw)
        assertTrue("late $late should be at least as good as early $early", late <= early + 0.01f)
        assertTrue(late < 0.03f)
    }

    @Test
    fun `no derivative is taken across a GNSS gap`() {
        val cal = leveledManager()
        feedDrive(cal, n = 200, trueYawOffset = 0.6f)
        val before = cal.yawFit!!.psi
        cal.yawGnssLost()
        // GNSS returns 15 m/s faster than it left (the car accelerated in
        // a tunnel). Across the gap that would read as a huge dv/dt.
        feedDrive(cal, n = 30, trueYawOffset = 0.6f, startTick = 200, speedAt = { 27f }, yawRateAt = { 0f })
        assertTrue(angleDiff(before, cal.psiYaw) < 0.05f)
        assertTrue(angleDiff(0.6f, cal.psiYaw) < 0.05f)
    }

    @Test
    fun `yaw samples are ignored before leveling`() {
        val cal = CalibrationManager()
        repeat(500) { cal.addYawSample(floatArrayOf(1f, 0f, g), floatArrayOf(0f, 0f, 0.1f), 15f) }
        assertFalse(cal.isLeveled)
        assertFalse(cal.isReady)
    }

    @Test
    fun `a non-finite speed is treated as a gap`() {
        val cal = leveledManager()
        feedDrive(cal, n = 200, trueYawOffset = 0.6f)
        cal.addYawSample(floatArrayOf(0f, 0f, g), floatArrayOf(0f, 0f, 0f), Float.NaN)
        feedDrive(cal, n = 200, trueYawOffset = 0.6f, startTick = 200)
        assertTrue(angleDiff(0.6f, cal.psiYaw) < 0.05f)
    }

    @Test
    fun `progress keeps moving during the yaw wait and reads 100 only when ready`() {
        // From the calibration-fix change: a full ring that keeps waiting
        // reads as a hang.
        val cal = leveledManager(maxYawWaitTicks = 300)
        feedDrive(cal, n = 299, trueYawOffset = 0f, speedAt = { 0f })
        assertFalse(cal.isReady)
        assertTrue("progress should keep moving while it waits", cal.progressPercent in 91..99)
        feedDrive(cal, n = 1, trueYawOffset = 0f, speedAt = { 0f })
        assertTrue(cal.isReady)
        assertEquals(100, cal.progressPercent)
    }
}
