package com.sih26168.deadreckoning

import kotlin.math.PI
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Regression test for a real bug found live on-device: a blackout trail
 * heading off in a very different direction from the real one, traced back
 * to MainActivity.readSample() and ReplayDataSource both doing only
 * Math.toRadians(bearing) on a real compass bearing (0=N, 90=E, clockwise)
 * and handing the result to code that expects this app's own heading
 * convention (0=east, ccw+ — FusionState.heading, MapMatcher's headingRad).
 * Those two conventions aren't a fixed rotation apart, they're a
 * reflection — compassDegToMathRad/mathRadToCompassDeg are the only
 * correct way to move between them (see Calibration.kt's doc for the full
 * story). Fixed values here, not just a round-trip check, so a future
 * change that swaps compassDegToMathRad back to a plain toRadians() (or
 * introduces the same bug at a new call site) fails loudly instead of
 * silently reproducing the live bug.
 */
class CalibrationHeadingConventionTest {

    private val tol = 1e-4f

    @Test
    fun `compass north converts to math 90 degrees`() {
        // North (compass 0 deg) is +y in the [east, north] local frame this
        // app integrates in, i.e. 90 deg (PI/2) in the 0=east ccw+ convention.
        assertEquals((PI / 2).toFloat(), Calibration.compassDegToMathRad(0f), tol)
    }

    @Test
    fun `compass east converts to math 0 degrees`() {
        // East (compass 90 deg) is +x in the local frame, i.e. math heading 0.
        assertEquals(0f, Calibration.compassDegToMathRad(90f), tol)
    }

    @Test
    fun `compass south converts to math negative 90 degrees`() {
        assertEquals((-PI / 2).toFloat(), Calibration.compassDegToMathRad(180f), tol)
    }

    @Test
    fun `compass west converts to math 180 degrees`() {
        // atan2 wraps to (-pi, pi], so west (compass 270) lands on +/-pi.
        val result = Calibration.compassDegToMathRad(270f)
        assertEquals(PI.toFloat(), kotlin.math.abs(result), tol)
    }

    @Test
    fun `mathRadToCompassDeg is the exact inverse across a full circle`() {
        var deg = 0f
        while (deg < 360f) {
            val mathRad = Calibration.compassDegToMathRad(deg)
            val back = Calibration.mathRadToCompassDeg(mathRad)
            // Compare on the unit circle, not the raw degree value, so the
            // 0/360 wrap boundary doesn't produce a false failure.
            assertEquals(kotlin.math.cos(Math.toRadians(deg.toDouble())).toFloat(),
                kotlin.math.cos(Math.toRadians(back.toDouble())).toFloat(), tol)
            assertEquals(kotlin.math.sin(Math.toRadians(deg.toDouble())).toFloat(),
                kotlin.math.sin(Math.toRadians(back.toDouble())).toFloat(), tol)
            deg += 15f
        }
    }

    @Test
    fun `a plain degrees-to-radians conversion is NOT this app's heading convention`() {
        // The exact bug this whole file guards against: Math.toRadians(bearing)
        // alone (no axis swap) disagrees with the real conversion everywhere
        // except the degenerate points where compass and math conventions
        // happen to coincide (0 and 180). At 90 (east) they're 90 degrees
        // apart, not equal — confirming the two are genuinely different
        // conventions, not just the same value under a different name.
        val bearingDeg = 90f
        val naive = Math.toRadians(bearingDeg.toDouble()).toFloat()
        val correct = Calibration.compassDegToMathRad(bearingDeg)
        assertEquals((PI / 2).toFloat(), kotlin.math.abs(naive - correct), tol)
    }
}
