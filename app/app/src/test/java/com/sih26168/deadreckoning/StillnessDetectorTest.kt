package com.sih26168.deadreckoning

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StillnessDetectorTest {

    private fun feed(d: StillnessDetector, n: Int, accel: (Int) -> FloatArray, gyro: FloatArray = floatArrayOf(0.005f, 0f, 0.003f)) {
        for (i in 0 until n) d.add(accel(i), gyro)
    }

    @Test
    fun `not still before a full window`() {
        val d = StillnessDetector()
        feed(d, 19, { floatArrayOf(0.02f, 0.01f, 9.81f) })
        assertFalse(d.isStill)
    }

    @Test
    fun `phone on a table is still`() {
        val d = StillnessDetector()
        feed(d, 30, { i -> floatArrayOf(0.02f, 0.01f, 9.81f + 0.03f * (i % 2)) })
        assertTrue(d.isStill)
    }

    @Test
    fun `walking shake is not still`() {
        val d = StillnessDetector()
        feed(d, 30, { i -> floatArrayOf(0f, 0f, 9.81f + if (i % 2 == 0) 1.5f else -1.5f) })
        assertFalse(d.isStill)
    }

    @Test
    fun `steady turning is not still even with a quiet accelerometer`() {
        val d = StillnessDetector()
        feed(d, 30, { floatArrayOf(0f, 0f, 9.81f) }, gyro = floatArrayOf(0f, 0f, 0.3f))
        assertFalse(d.isStill)
    }

    @Test
    fun `reset forgets the window`() {
        val d = StillnessDetector()
        feed(d, 30, { floatArrayOf(0f, 0f, 9.81f) })
        d.reset()
        assertFalse(d.isStill)
    }

    @Test
    fun `speed is zeroed only while still and slow`() {
        assertEquals(0f, speedWhileStill(2.8f, true), 0f)
        assertEquals(2.8f, speedWhileStill(2.8f, false), 0f)
        assertEquals(15f, speedWhileStill(15f, true), 0f)
    }
}
