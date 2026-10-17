package com.sih26168.deadreckoning

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DriveSelfCheckTest {

    private val lines = mutableListOf<String>()
    private fun check() = DriveSelfCheck({ lines += it })

    private fun gnss(c: DriveSelfCheck, x: Float, e: EnsembleModel? = null) =
        c.tick(FusionMode.GNSS_TRACKING, floatArrayOf(x, 0f), floatArrayOf(x, 0f), e)

    @Test
    fun `demo blackout reports live drift every second and a summary`() {
        val c = check()
        repeat(5) { gnss(c, it * 1f) }
        // 10 s at 10 m/s; dead reckoning drifts sideways 0.05 m per tick.
        for (i in 1..100) {
            val x = 4f + i * 1f
            c.tick(FusionMode.BLACKOUT, floatArrayOf(x, i * 0.05f), floatArrayOf(x, 0f), null)
        }
        gnss(c, 105f)
        val live = lines.filter { it.contains("[live") }
        assertEquals(10, live.size)
        assertTrue(live.last(), live.last().contains("error 5.0 m"))
        val r = c.reports.single()
        assertTrue(r.truthDuringBlackout)
        assertEquals(10f, r.seconds, 1e-4f)
        assertEquals(99f, r.distanceM, 0.5f)       // GNSS path between the first and last blackout tick
        assertEquals(5f, r.endErrorM!!, 0.01f)     // DR end (104, 5) vs GNSS at that tick (104, 0)
        assertEquals(5.05f, r.driftPercent!!, 0.05f)
        assertTrue(lines.last(), lines.last().startsWith("blackout ended after 10.0 s"))
    }

    @Test
    fun `real blackout is scored against the first fix after it`() {
        val c = check()
        gnss(c, 0f)
        for (i in 1..50) c.tick(FusionMode.BLACKOUT, floatArrayOf(i * 2f, 0f), null, null)
        assertTrue("no live line without truth", lines.none { it.contains("[live") })
        // Two ticks without a fix after the blackout (BLEND but GNSS not back yet).
        repeat(2) { c.tick(FusionMode.BLEND, floatArrayOf(100f, 0f), null, null) }
        assertTrue(c.reports.isEmpty())
        c.tick(FusionMode.BLEND, floatArrayOf(100f, 0f), floatArrayOf(108f, 6f), null)
        val r = c.reports.single()
        assertFalse(r.truthDuringBlackout)
        assertEquals(98f, r.distanceM, 0.01f)       // dead-reckoned path
        assertEquals(10f, r.endErrorM!!, 0.01f)
        assertEquals(10.2f, r.driftPercent!!, 0.1f)
    }

    @Test
    fun `a blackout with no fix afterwards is still reported`() {
        val c = check()
        gnss(c, 0f)
        for (i in 1..20) c.tick(FusionMode.BLACKOUT, floatArrayOf(i * 1f, 0f), null, null)
        repeat(DriveSelfCheck.MAX_WAIT_FOR_FIX_TICKS + 1) { c.tick(FusionMode.BLEND, floatArrayOf(20f, 0f), null, null) }
        val r = c.reports.single()
        assertNull(r.endErrorM)
        assertNull(r.driftPercent)
        assertTrue(lines.last().contains("unknown"))
    }

    @Test
    fun `a new blackout before a fix closes the previous one`() {
        val c = check()
        gnss(c, 0f)
        repeat(10) { c.tick(FusionMode.BLACKOUT, floatArrayOf(it * 1f, 0f), null, null) }
        c.tick(FusionMode.BLEND, floatArrayOf(10f, 0f), null, null)
        repeat(10) { c.tick(FusionMode.BLACKOUT, floatArrayOf(10f + it, 0f), null, null) }
        assertEquals(1, c.reports.size)
        c.tick(FusionMode.GNSS_TRACKING, floatArrayOf(20f, 0f), floatArrayOf(20f, 0f), null)
        assertEquals(2, c.reports.size)
    }

    @Test
    fun `chunks are logged with the model that answered and counted per blackout`() {
        val slow = object : Predictor {
            override fun predict(window: Array<FloatArray>) = Array(window.size) { floatArrayOf(0f, 0f) }
        }
        val fast = object : Predictor {
            override fun predict(window: Array<FloatArray>) = Array(window.size) { floatArrayOf(0f, 0f) }
        }
        val e = EnsembleModel(slow, fast)
        val w = Array(50) { FloatArray(6) }
        val c = check()
        gnss(c, 0f, e)
        c.tick(FusionMode.BLACKOUT, floatArrayOf(1f, 0f), null, e)
        e.predict(w, 12f)
        c.tick(FusionMode.BLACKOUT, floatArrayOf(2f, 0f), null, e)
        e.predict(w, 3f)
        c.tick(FusionMode.BLACKOUT, floatArrayOf(3f, 0f), null, e)
        c.tick(FusionMode.GNSS_TRACKING, floatArrayOf(3f, 0f), floatArrayOf(3f, 0f), e)
        assertTrue(lines.any { it == "chunk 1: fast model, start speed 12.0 m/s" })
        assertTrue(lines.any { it == "chunk 2: slow model, start speed 3.0 m/s" })
        val r = c.reports.single()
        assertEquals(1, r.fastChunks)
        assertEquals(1, r.slowChunks)
        assertEquals(0, r.fallbacks)
    }

    @Test
    fun `ensemble counts chunks by role, including zero-correction chunks`() {
        val boom = object : Predictor {
            override fun predict(window: Array<FloatArray>): Array<FloatArray> = throw IllegalStateException()
        }
        val ok = object : Predictor {
            override fun predict(window: Array<FloatArray>) = Array(window.size) { floatArrayOf(0f, 0f) }
        }
        val w = Array(50) { FloatArray(6) }
        val e = EnsembleModel(ok, boom)
        e.predict(w, 10f)  // fast throws -> slow
        e.predict(w, 2f)
        assertEquals(2, e.slowChunks); assertEquals(0, e.fastChunks); assertEquals(1, e.fallbackCount)
        val dead = EnsembleModel(boom, boom)
        dead.predict(w, 10f)
        assertEquals(1, dead.zeroChunks); assertEquals(1, dead.chunks)
        assertEquals(10f, dead.lastStartSpeed)
    }

    @Test
    fun `nothing is logged while GNSS tracks normally`() {
        val c = check()
        repeat(100) { gnss(c, it * 1f) }
        assertTrue(lines.isEmpty())
        assertTrue(c.reports.isEmpty())
    }
}
