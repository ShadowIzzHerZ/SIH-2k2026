package com.sih26168.deadreckoning

import org.junit.Assert.assertEquals
import org.junit.Test

class GnssSpeedCheckTest {

    private val metresPerDegLat = 111_194.9

    /** Feeds [ticks] ticks at 10 Hz, the fix refreshing once a second like
     * a phone's GPS, and returns the last filtered speed. */
    private fun run(check: GnssSpeedCheck, ticks: Int, speed: Float, northAt: (Double) -> Double): Float {
        var out = speed
        for (i in 0 until ticks) {
            val fixTime = (i / 10).toDouble()
            out = check.filter(northAt(fixTime) / metresPerDegLat, 0.0, speed)
        }
        return out
    }

    /** The Galaxy F17 indoors: 2.5 m/s reported, fixes going nowhere. */
    @Test
    fun `speed with fixes that stay put reads as zero`() {
        val jitter = doubleArrayOf(0.0, 1.5, -1.0, 2.0, 0.5, -1.5, 1.0, 0.0)
        val out = run(GnssSpeedCheck(), 80, 2.5f) { t -> jitter[t.toInt() % jitter.size] }
        assertEquals(0f, out, 0f)
    }

    @Test
    fun `a real walk keeps its speed`() {
        val out = run(GnssSpeedCheck(), 80, 1.4f) { t -> 1.4 * t }
        assertEquals(1.4f, out, 0f)
    }

    @Test
    fun `a slow car keeps its speed`() {
        val out = run(GnssSpeedCheck(), 80, 3.0f) { t -> 3.0 * t }
        assertEquals(3.0f, out, 0f)
    }

    @Test
    fun `road speeds are never checked`() {
        val out = run(GnssSpeedCheck(), 80, 15f) { 0.0 }
        assertEquals(15f, out, 0f)
    }

    @Test
    fun `nothing is zeroed before a full window`() {
        val out = run(GnssSpeedCheck(), 50, 2.5f) { 0.0 }
        assertEquals(2.5f, out, 0f)
    }

    /** Too short a reported path to tell noise from a real crawl. */
    @Test
    fun `a crawl too short to judge keeps its speed`() {
        val out = run(GnssSpeedCheck(), 80, 0.6f) { 0.0 }
        assertEquals(0.6f, out, 0f)
    }

    @Test
    fun `reset starts a new window`() {
        val check = GnssSpeedCheck()
        run(check, 80, 2.5f) { 0.0 }
        check.reset()
        assertEquals(2.5f, run(check, 10, 2.5f) { 0.0 }, 0f)
    }

    /** Standing with noise, then walking off: the speed comes back once
     * the fixes cover enough of the reported path. */
    @Test
    fun `speed returns after the phone really starts moving`() {
        val check = GnssSpeedCheck()
        run(check, 80, 2.5f) { 0.0 }
        var out = 0f
        for (i in 0 until 60) {
            val t = (i / 10).toDouble()
            out = check.filter(2.5 * t / metresPerDegLat, 0.0, 2.5f)
        }
        assertEquals(2.5f, out, 0f)
    }
}
