package com.sih26168.deadreckoning

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.PI
import kotlin.math.abs

/**
 * The Kotlin forward-axis fit against the Python one it ports
 * (experiments/drift/lib.py's calibrate_fwdfix), on the app's own bundled
 * real drive. scripts/fwdfix_reference.py wrote the expected numbers in
 * src/test/resources/fwdfix_reference.json.
 */
class CalibrationForwardAxisTest {

    private class Drive(val accel: List<FloatArray>, val gyro: List<FloatArray>, val speed: FloatArray)

    private fun asset(name: String): File =
        listOf(File("src/main/assets/$name"), File("app/app/src/main/assets/$name"), File("app/src/main/assets/$name"))
            .firstOrNull { it.exists() } ?: error("can't find asset $name from ${File(".").absolutePath}")

    private fun loadDrive(name: String): Drive {
        val rows = JSONObject(asset(name).readText()).getJSONArray("rows")
        val accel = ArrayList<FloatArray>(); val gyro = ArrayList<FloatArray>()
        val speed = FloatArray(rows.length())
        for (i in 0 until rows.length()) {
            val r = rows.getJSONObject(i)
            accel += floatArrayOf(r.getDouble("ax").toFloat(), r.getDouble("ay").toFloat(), r.getDouble("az").toFloat())
            gyro += floatArrayOf(r.getDouble("gx").toFloat(), r.getDouble("gy").toFloat(), r.getDouble("gz").toFloat())
            speed[i] = r.getDouble("speed").toFloat()
        }
        return Drive(accel, gyro, speed)
    }

    private val reference: JSONObject by lazy {
        val text = javaClass.classLoader!!.getResourceAsStream("fwdfix_reference.json")!!.bufferedReader().readText()
        JSONObject(text)
    }

    private fun angleDiff(a: Double, b: Double): Double {
        var d = a - b
        while (d > PI) d -= 2 * PI
        while (d < -PI) d += 2 * PI
        return abs(d)
    }

    @Test
    fun `smoothing and gradient match numpy`() {
        val x = doubleArrayOf(1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 9.0, 6.0, 0.0, 2.0, 5.0, 8.0)
        // np.convolve(x, ones(10), "same") / np.convolve(ones_like(x), ones(10), "same")
        val smooth = doubleArrayOf(4.0, 4.5, 4.2857142857, 4.875, 5.0, 4.5, 4.6, 4.7, 5.3, 5.0, 5.0, 4.7142857143, 5.0)
        assertArrayEquals(smooth, Calibration.smoothEdgeNormalized(x), 1e-9)
        // np.gradient(x, 0.1)
        val grad = doubleArrayOf(30.0, 5.0, 20.0, 15.0, -5.0, -10.0, 10.0, 15.0, -45.0, -20.0, 25.0, 30.0, 30.0)
        assertArrayEquals(grad, Calibration.gradient(x, 0.1), 1e-9)
    }

    @Test
    fun `batch fit matches the Python version on the bundled real drive`() {
        val ref = reference.getJSONObject("replay_drive.json")
        val d = loadDrive("replay_drive.json")
        // The same whole-drive leveling the Python version uses.
        val r = Calibration.levelingRotation(Calibration.estimateGravityVector(d.accel))
        val refR = ref.getJSONArray("r_level")
        for (i in 0 until 9) assertEquals(refR.getDouble(i), r[i].toDouble(), 1e-5)
        val fit = Calibration.estimateForwardAxis(
            d.accel.map { Calibration.matVec3(r, it) }, d.gyro.map { Calibration.matVec3(r, it) }, d.speed,
        )
        assertNotNull(fit)
        assertEquals(ref.getInt("ok_samples"), fit!!.samples)
        assertTrue(
            "kotlin ${fit.psi} vs python ${ref.getDouble("psi")}",
            angleDiff(fit.psi.toDouble(), ref.getDouble("psi")) < 1e-4,
        )
        assertTrue("a real drive is coherent, got ${fit.coherence}", fit.coherence > 0.8f)
    }

    @Test
    fun `streaming fit matches the batch fit on the real drive`() {
        val d = loadDrive("replay_drive.json")
        val r = Calibration.levelingRotation(Calibration.estimateGravityVector(d.accel))
        val la = d.accel.map { Calibration.matVec3(r, it) }
        val lg = d.gyro.map { Calibration.matVec3(r, it) }
        val batch = Calibration.estimateForwardAxis(la, lg, d.speed)!!
        val est = ForwardAxisEstimator()
        for (i in d.speed.indices) est.add(d.accel[i], d.gyro[i], d.speed[i], r)
        val live = est.fit(r)!!
        // Only the ~11 samples at the drive's two ends differ.
        assertTrue(abs(batch.samples - live.samples) <= 11)
        assertTrue(
            "live ${Math.toDegrees(live.psi.toDouble())} vs batch ${Math.toDegrees(batch.psi.toDouble())} deg",
            angleDiff(live.psi.toDouble(), batch.psi.toDouble()) < Math.toRadians(1.0),
        )
    }

    @Test
    fun `live calibration on the real drive lands on the offline forward axis`() {
        val ref = reference.getJSONObject("replay_drive.json").getDouble("psi")
        val d = loadDrive("replay_drive.json")
        val cal = CalibrationManager()
        var i = 0
        while (!cal.isLeveled) { cal.addLevelSample(d.accel[i]); i++ }
        while (i < d.speed.size) { cal.addYawSample(d.accel[i], d.gyro[i], d.speed[i]); i++ }
        assertTrue(cal.isReady)
        assertTrue(cal.yawConverged)
        // Leveling over the drive and the raw-frame sum mean the live
        // answer converges on the offline one; only the drive's edge
        // samples and the last partial second differ.
        assertTrue(
            "live ${Math.toDegrees(cal.psiYaw.toDouble())} vs offline ${Math.toDegrees(ref)} deg",
            angleDiff(cal.psiYaw.toDouble(), ref) < Math.toRadians(1.5),
        )
    }

    @Test
    fun `streaming fit gives the same answer whichever leveling it is read with later`() {
        // The raw-frame sum means re-leveling doesn't need the samples
        // again: fit(r) after the fact equals feeding leveled samples.
        val d = loadDrive("replay_drive.json")
        val early = Calibration.levelingRotation(Calibration.estimateGravityVector(d.accel.take(20)))
        val whole = Calibration.levelingRotation(Calibration.estimateGravityVector(d.accel))
        val est = ForwardAxisEstimator()
        for (i in d.speed.indices) est.add(d.accel[i], d.gyro[i], d.speed[i], early)
        val batch = Calibration.estimateForwardAxis(
            d.accel.map { Calibration.matVec3(whole, it) }, d.gyro.map { Calibration.matVec3(whole, it) }, d.speed,
        )!!
        assertTrue(angleDiff(est.fit(whole)!!.psi.toDouble(), batch.psi.toDouble()) < Math.toRadians(1.0))
        // And reading it with the 2 s window's leveling really is different
        // (why the app re-levels over the drive).
        assertTrue(angleDiff(est.fit(early)!!.psi.toDouble(), batch.psi.toDouble()) > Math.toRadians(5.0))
    }

    @Test
    fun `calibrated forward accel tracks GNSS speed change on the real drive`() {
        // The point of the fix: after calibration, the x axis really is
        // the car's forward axis. With psi = 0 (the old answer on this
        // drive was near-arbitrary) the correlation is negative.
        val d = loadDrive("replay_drive.json")
        val r = Calibration.levelingRotation(Calibration.estimateGravityVector(d.accel))
        val psi = reference.getJSONObject("replay_drive.json").getDouble("psi").toFloat()
        val dvdt = Calibration.gradient(Calibration.smoothEdgeNormalized(DoubleArray(d.speed.size) { d.speed[it].toDouble() }), 0.1)
        fun corr(p: Float): Double {
            val fwd = Calibration.smoothEdgeNormalized(DoubleArray(d.accel.size) { Calibration.calibrateSample(d.accel[it], r, p)[0].toDouble() })
            val mx = fwd.average(); val my = dvdt.average()
            var sxy = 0.0; var sxx = 0.0; var syy = 0.0
            for (k in fwd.indices) { sxy += (fwd[k] - mx) * (dvdt[k] - my); sxx += (fwd[k] - mx) * (fwd[k] - mx); syy += (dvdt[k] - my) * (dvdt[k] - my) }
            return sxy / Math.sqrt(sxx * syy)
        }
        assertTrue(corr(psi) > 0.6)
        assertTrue(corr(0f) < 0.0)
    }

    @Test
    fun `too few usable samples gives no fit`() {
        val n = 100  // the offline floor is strictly more than 100
        val accel = List(n) { floatArrayOf(1f, 0f, 9.8f) }
        val gyro = List(n) { floatArrayOf(0f, 0f, 0f) }
        val speed = FloatArray(n) { 10f + it * 0.1f }
        assertNull(Calibration.estimateForwardAxis(accel, gyro, speed))
        assertNotNull(Calibration.estimateForwardAxis(accel + accel, gyro + gyro, FloatArray(2 * n) { 10f + it * 0.1f }))
        // Below 3 m/s nothing counts.
        assertNull(Calibration.estimateForwardAxis(accel + accel, gyro + gyro, FloatArray(2 * n) { 2.9f }))
    }

    @Test
    fun `walking demo replays never count toward the forward axis`() {
        for (name in listOf("replay_demo1.json", "replay_demo_2142.json")) {
            assertEquals(0, reference.getJSONObject(name).getInt("ok_samples"))
        }
    }
}
