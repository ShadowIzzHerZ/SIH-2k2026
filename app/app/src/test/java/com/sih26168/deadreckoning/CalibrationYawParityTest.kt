package com.sih26168.deadreckoning

import kotlin.math.cos
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Calibration.estimateYawMisalignment against src/calibration.py's
 * estimate_yaw_misalignment on the same inputs. The expected values were
 * produced by running the Python function itself; it keeps samples above
 * np.percentile(mag, 75) (linear interpolation) and takes np.median
 * (mean of the two middle values for an even count).
 */
class CalibrationYawParityTest {

    /** One leveled-accel sample of magnitude `mag` pointing at `angleRad`,
     * paired with a due-east GPS heading, so each sample's angle diff is
     * exactly -angleRad. */
    private fun samples(mags: List<Float>, angles: List<Float>): Pair<List<FloatArray>, List<FloatArray>> {
        val accel = mags.zip(angles).map { (m, a) -> floatArrayOf(m * cos(a), m * sin(a)) }
        val heading = mags.map { floatArrayOf(1f, 0f) }
        return accel to heading
    }

    @Test
    fun `even count of kept samples takes the mean of the two middle diffs`() {
        // np.percentile([1..8], 75) = 6.25 -> keeps mags 7 and 8 (diffs
        // -0.1, -0.3) -> np.median = -0.2.
        val (accel, heading) = samples(
            (1..8).map { it.toFloat() },
            listOf(1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 0.1f, 0.3f),
        )
        assertEquals(-0.2f, Calibration.estimateYawMisalignment(accel, heading), 1e-5f)
    }

    @Test
    fun `small window still keeps the top sample like numpy does`() {
        // np.percentile([1,2,3,4], 75) = 3.25 -> keeps mag 4 (diff -0.3).
        // Taking sorted[(n * 0.75).toInt()] = 4 as the cutoff kept nothing
        // and silently returned 0.
        val (accel, heading) = samples(
            listOf(1f, 2f, 3f, 4f),
            listOf(1.0f, -1.0f, 0.1f, 0.3f),
        )
        assertEquals(-0.3f, Calibration.estimateYawMisalignment(accel, heading), 1e-5f)
    }
}
