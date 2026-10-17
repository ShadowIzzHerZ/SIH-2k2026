package com.sih26168.deadreckoning

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * Warm start from an earlier drive in the same mount (CalibrationPrior),
 * and the gate that keeps the prior until this drive's own fit is
 * trustworthy.
 */
class CalibrationWarmStartTest {

    private val g = 9.81f
    private val rng = Random(7)

    private fun angleDiff(a: Float, b: Float): Float {
        var d = (a - b).toDouble()
        while (d > PI) d -= 2 * PI
        while (d < -PI) d += 2 * PI
        return abs(d).toFloat()
    }

    private fun flat() = floatArrayOf(rng.nextFloat() * 0.04f - 0.02f, rng.nextFloat() * 0.04f - 0.02f, g)

    /** A phone lying flat, rotated [offset] from the car's forward axis
     * (same construction as CalibrationManagerTest.feedDrive). */
    private fun feedDrive(cal: CalibrationManager, n: Int, offset: Float, start: Int = 0) {
        val speedAt = { t: Int -> 12f + 4f * sin(t * 0.05).toFloat() }
        for (i in 0 until n) {
            val t = start + i
            val v = speedAt(t)
            val lon = (speedAt(t + 1) - speedAt(t - 1)) / 0.2f
            val wz = 0.15f * sin(t * 0.021).toFloat()
            val lat = v * wz
            val c = cos(-offset.toDouble()).toFloat(); val s = sin(-offset.toDouble()).toFloat()
            val ax = c * lon - s * lat + rng.nextFloat() * 0.2f - 0.1f
            val ay = s * lon + c * lat + rng.nextFloat() * 0.2f - 0.1f
            cal.addYawSample(floatArrayOf(ax, ay, g), floatArrayOf(0f, 0f, wz), v)
        }
    }

    private fun leveled(prior: CalibrationPrior?): CalibrationManager {
        val cal = CalibrationManager(prior = prior)
        repeat(25) { cal.addLevelSample(flat()) }
        require(cal.isLeveled)
        return cal
    }

    private val up = floatArrayOf(0f, 0f, 1f)

    // ------------------------------------------------------------ JSON

    @Test
    fun `prior survives a JSON round trip`() {
        val p = CalibrationPrior(floatArrayOf(0.6f, 0f, 0.8f), 1.25f, 412)
        val back = CalibrationPrior.fromJson(p.toJson())!!
        assertEquals(p.psi, back.psi, 1e-6f)
        assertEquals(p.samples, back.samples)
        for (i in 0 until 3) assertEquals(p.gravity[i], back.gravity[i], 1e-6f)
    }

    @Test
    fun `bad saved priors mean no warm start`() {
        val good = JSONObject(CalibrationPrior(up, 0.5f, 300).toJson())
        assertNull(CalibrationPrior.fromJson(null))
        assertNull(CalibrationPrior.fromJson("not json"))
        assertNull(CalibrationPrior.fromJson(JSONObject(good.toString()).put("version", 99).toString()))
        assertNull(CalibrationPrior.fromJson(JSONObject(good.toString()).put("samples", 0).toString()))
        assertNull(CalibrationPrior.fromJson(JSONObject(good.toString()).put("psi", "x").toString()))
        assertNull(CalibrationPrior.fromJson(JSONObject(good.toString()).put("gravity", org.json.JSONArray(listOf(0, 0))).toString()))
        assertNull(CalibrationPrior.fromJson(JSONObject(good.toString()).put("gravity", org.json.JSONArray(listOf(0, 0, 3))).toString()))
        // Slightly off-unit is normalised, and psi is wrapped.
        val p = CalibrationPrior.fromJson(
            JSONObject(good.toString()).put("gravity", org.json.JSONArray(listOf(0, 0, 1.1))).put("psi", 7.0).toString(),
        )!!
        assertEquals(1f, p.gravity[2], 1e-6f)
        assertEquals((7.0 - 2 * PI).toFloat(), p.psi, 1e-5f)
    }

    // ------------------------------------------------------- warm start

    @Test
    fun `same mount is ready straight after leveling with the prior's forward axis`() {
        val cal = leveled(CalibrationPrior(up, 0.9f, 500))
        assertTrue(cal.isReady)
        assertTrue(cal.usingPrior)
        assertFalse(cal.isRoughCalibration)
        assertEquals(0.9f, cal.psiYaw, 1e-6f)
    }

    @Test
    fun `a different mount ignores the prior`() {
        // Gravity 20 degrees away: the phone was mounted differently.
        val tilted = floatArrayOf(sin(0.35).toFloat(), 0f, cos(0.35).toFloat())
        val cal = leveled(CalibrationPrior(tilted, 0.9f, 500))
        assertFalse(cal.isReady)
        assertFalse(cal.usingPrior)
    }

    @Test
    fun `no prior is the old path`() {
        val cal = leveled(null)
        assertFalse(cal.isReady)
        feedDrive(cal, 400, offset = 0.4f)
        assertTrue(cal.isReady && !cal.usingPrior)
        assertTrue(angleDiff(0.4f, cal.psiYaw) < 0.05f)
    }

    @Test
    fun `a short stretch only nudges the prior, a long one takes over`() {
        // Prior 0.2 rad off this drive's truth (0.6), from 500 samples.
        val cal = leveled(CalibrationPrior(up, 0.4f, 500))
        feedDrive(cal, 150, offset = 0.6f)
        assertTrue(cal.usingPrior)
        assertTrue("150 samples against 500: still near the prior, got ${cal.psiYaw}", cal.psiYaw in 0.4f..0.5f)
        feedDrive(cal, 400, offset = 0.6f, start = 150)
        assertFalse("this drive now outweighs the prior", cal.usingPrior)
        assertTrue(angleDiff(0.6f, cal.psiYaw) < 0.05f)
    }

    @Test
    fun `an old prior never outweighs more than a minute of new driving`() {
        val cal = leveled(CalibrationPrior(up, 0.0f, 1_000_000))
        feedDrive(cal, 650, offset = 0.5f)
        assertFalse(cal.usingPrior)
        assertTrue(angleDiff(0.5f, cal.psiYaw) < 0.05f)
    }

    @Test
    fun `a phone turned in the same mount overrides the prior early`() {
        val cal = leveled(CalibrationPrior(up, 0.6f + 1.6f, 500))  // ~90 degrees off
        feedDrive(cal, 160, offset = 0.6f)
        assertFalse("a clear contradiction must not wait 30 s", cal.usingPrior)
        assertTrue(angleDiff(0.6f, cal.psiYaw) < 0.1f)
    }

    // ---------------------------------------------------------- saving

    @Test
    fun `a long good drive produces a prior to save, a short one does not`() {
        val cal = leveled(null)
        feedDrive(cal, 200, offset = -0.7f)
        assertTrue(cal.isReady)
        assertNull("too little data to be worth saving", cal.currentPrior())
        feedDrive(cal, 200, offset = -0.7f, start = 200)
        val p = cal.currentPrior()
        assertNotNull(p)
        assertTrue(angleDiff(-0.7f, p!!.psi) < 0.05f)
        assertEquals(1f, p.gravity[2], 0.01f)
        // ...and starting from it is ready at once on the next drive.
        val next = leveled(p)
        assertTrue(next.isReady && next.usingPrior)
    }

    @Test
    fun `nothing is saved before this drive adds enough, after a skip, or from a timed-out guess`() {
        val usingPrior = leveled(CalibrationPrior(up, 0.4f, 500))
        assertNull("the prior alone is not re-saved", usingPrior.currentPrior())
        feedDrive(usingPrior, 250, offset = 0.4f)
        assertNull(usingPrior.currentPrior())
        feedDrive(usingPrior, 100, offset = 0.4f, start = 250)
        val blended = usingPrior.currentPrior()!!
        assertTrue(blended.samples > 500)
        assertTrue(angleDiff(0.4f, blended.psi) < 0.05f)

        val skipped = leveled(null)
        skipped.skipForTesting()
        feedDrive(skipped, 500, offset = 0.4f)
        assertNull(skipped.currentPrior())

        val walk = CalibrationManager(maxYawWaitTicks = 50)
        repeat(25) { walk.addLevelSample(flat()) }
        repeat(60) { walk.addYawSample(flat(), floatArrayOf(0f, 0f, 0f), 1.2f) }
        assertTrue(walk.isReady && walk.isRoughCalibration)
        assertNull(walk.currentPrior())
    }

    // ------------------------------------------------- the bundled drive

    private fun asset(name: String): File =
        listOf(File("src/main/assets/$name"), File("app/app/src/main/assets/$name"))
            .firstOrNull { it.exists() } ?: error("can't find $name")

    @Test
    fun `on the bundled short drive a warm start avoids the early wander`() {
        val ref = JSONObject(javaClass.classLoader!!.getResourceAsStream("fwdfix_reference.json")!!.bufferedReader().readText())
            .getJSONObject("replay_drive.json")
        val psiRef = ref.getDouble("psi").toFloat()
        val rows = JSONObject(asset("replay_drive.json").readText()).getJSONArray("rows")
        val accel = List(rows.length()) { i -> rows.getJSONObject(i).let { floatArrayOf(it.getDouble("ax").toFloat(), it.getDouble("ay").toFloat(), it.getDouble("az").toFloat()) } }
        val gyro = List(rows.length()) { i -> rows.getJSONObject(i).let { floatArrayOf(it.getDouble("gx").toFloat(), it.getDouble("gy").toFloat(), it.getDouble("gz").toFloat()) } }
        val speed = FloatArray(rows.length()) { rows.getJSONObject(it).getDouble("speed").toFloat() }
        val gAll = Calibration.estimateGravityVector(accel)

        fun run(prior: CalibrationPrior?): Float {
            // Worst error against the offline answer over the drive, once ready.
            val cal = CalibrationManager(prior = prior)
            var worst = 0f
            for (i in accel.indices) {
                if (!cal.isLeveled) cal.addLevelSample(accel[i]) else cal.addYawSample(accel[i], gyro[i], speed[i])
                if (cal.isReady) worst = maxOf(worst, angleDiff(cal.psiYaw, psiRef))
            }
            return worst
        }
        val cold = run(null)
        val warm = run(CalibrationPrior(gAll, psiRef, 598))
        assertTrue("cold start wanders (worst ${Math.toDegrees(cold.toDouble())} deg)", cold > Math.toRadians(20.0))
        println("worst error on the bundled drive: cold ${Math.toDegrees(cold.toDouble())} deg, warm ${Math.toDegrees(warm.toDouble())} deg")
        assertTrue("warm start stays close (worst ${Math.toDegrees(warm.toDouble())} deg)", warm < Math.toRadians(8.0))
    }
}
