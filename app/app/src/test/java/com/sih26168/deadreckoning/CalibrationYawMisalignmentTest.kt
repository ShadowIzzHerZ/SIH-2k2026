package com.sih26168.deadreckoning

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Calibration.estimateYawMisalignment, same cases as
 * tests/test_calibration.py. The old version compared the leveled
 * phone-frame accel angle with the world-frame GPS course, so it returned
 * the mount offset plus the average course: a perfectly aligned phone on a
 * north-bound road came out at 90 degrees.
 */
class CalibrationYawMisalignmentTest {

    private class Segment(
        val accelXY: List<FloatArray>,
        val headingXY: List<FloatArray>,
        val speed: List<Float>,
        val aLon: DoubleArray,
        val dt: Float,
    )

    /** Leveled-phone horizontal accel of a car that speeds up, brakes and
     * turns, where yawRotation(psiTrue) takes the phone frame to the
     * vehicle frame; plus its GPS course (world-frame unit vectors) and
     * speed. */
    private fun driveSegment(psiTrue: Double, course0: Double, n: Int = 600, dt: Double = 0.1, noise: Double = 0.0): Segment {
        val rng = Random(0)
        val accel = ArrayList<FloatArray>(n)
        val heading = ArrayList<FloatArray>(n)
        val speed = ArrayList<Float>(n)
        val aLonAll = DoubleArray(n)
        var v = 12.0
        var crs = course0
        val c = cos(psiTrue)
        val s = sin(psiTrue)
        for (i in 0 until n) {
            val t = i * dt
            val aLon = 1.5 * sin(2 * PI * t / 20.0)       // accelerate, then brake
            val yawRate = 0.15 * sin(2 * PI * t / 13.0)   // weave left and right
            v += aLon * dt
            crs += yawRate * dt
            val aLat = v * yawRate
            // yawRotation(-psiTrue) applied to [forward, left]
            val ax = c * aLon + s * aLat + (rng.nextDouble() - 0.5) * 2 * noise
            val ay = -s * aLon + c * aLat + (rng.nextDouble() - 0.5) * 2 * noise
            accel.add(floatArrayOf(ax.toFloat(), ay.toFloat()))
            heading.add(floatArrayOf(cos(crs).toFloat(), sin(crs).toFloat()))
            speed.add(v.toFloat())
            aLonAll[i] = aLon
        }
        return Segment(accel, heading, speed, aLonAll, dt.toFloat())
    }

    private fun angleErr(a: Double, b: Double) = abs(atan2(sin(a - b), cos(a - b)))

    @Test
    fun `recovers the mount offset when driving along world x`() {
        val seg = driveSegment(Math.toRadians(30.0), course0 = 0.0)
        val psi = Calibration.estimateYawMisalignment(seg.accelXY, seg.headingXY, seg.speed, seg.dt)
        assertEquals(Math.toRadians(30.0), psi.toDouble(), Math.toRadians(1.0))
        // Rotating by psi puts braking back on -x, not flipped onto +x.
        val r = Calibration.yawRotation(psi)
        for (i in seg.accelXY.indices) {
            val a = seg.accelXY[i]
            assertEquals(seg.aLon[i], (r[0] * a[0] + r[1] * a[1]).toDouble(), 0.05)
        }
    }

    @Test
    fun `estimate does not depend on the course driven`() {
        for (psiDeg in listOf(30.0, -100.0, 170.0)) {
            for (courseDeg in listOf(90.0, 180.0, -135.0, 17.0)) {
                val seg = driveSegment(Math.toRadians(psiDeg), Math.toRadians(courseDeg), noise = 0.3)
                val psi = Calibration.estimateYawMisalignment(seg.accelXY, seg.headingXY, seg.speed, seg.dt)
                assertTrue(
                    "psi=$psiDeg course=$courseDeg gave ${Math.toDegrees(psi.toDouble())}",
                    angleErr(psi.toDouble(), Math.toRadians(psiDeg)) < Math.toRadians(2.0),
                )
                val spread = Calibration.yawMisalignmentSpread(seg.accelXY, seg.headingXY, seg.speed, seg.dt)
                assertTrue("a clean drive should fit coherently, spread=$spread", spread < 0.1f)
            }
        }
    }

    @Test
    fun `an aligned phone on a north-bound road is not 90 degrees off`() {
        val seg = driveSegment(0.0, course0 = PI / 2)
        val psi = Calibration.estimateYawMisalignment(seg.accelXY, seg.headingXY, seg.speed, seg.dt)
        assertEquals(0.0, psi.toDouble(), Math.toRadians(1.0))
    }

    @Test
    fun `no usable samples gives zero, not a guess`() {
        val n = 400
        val accel = List(n) { floatArrayOf(0.4f, 0.3f) }
        val heading = List(n) { floatArrayOf(1f, 0f) }
        for (v in listOf(15f, 0.5f)) {  // steady cruise / crawling
            val speed = List(n) { v }
            assertEquals(0f, Calibration.estimateYawMisalignment(accel, heading, speed, 0.1f), 0f)
            assertEquals(1f, Calibration.yawMisalignmentSpread(accel, heading, speed, 0.1f), 0f)
        }
        assertEquals(0f, Calibration.estimateYawMisalignment(accel.take(2), heading.take(2), List(2) { 15f }, 0.1f), 0f)
    }
}
