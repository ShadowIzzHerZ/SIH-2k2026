package com.sih26168.deadreckoning

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** How RoadMapView splits its trail into polylines (startOfNextTrailLine). */
class TrailLineTest {

    @Test
    fun `keeps extending the current line below the cap`() {
        assertNull(startOfNextTrailLine(listOf(1, 2, 3), modeChanged = false, maxPoints = 4, modeChangeCarry = 3))
    }

    @Test
    fun `a mode change starts at the marker's last position`() {
        assertEquals(listOf(9), startOfNextTrailLine(listOf(1, 2), modeChanged = true, maxPoints = 4, modeChangeCarry = 9))
        assertEquals(emptyList<Int>(), startOfNextTrailLine(emptyList<Int>(), modeChanged = true, maxPoints = 4, modeChangeCarry = null))
    }

    @Test
    fun `a full line continues from its own last point`() {
        assertEquals(listOf(4), startOfNextTrailLine(listOf(1, 2, 3, 4), modeChanged = false, maxPoints = 4, modeChangeCarry = 4))
    }

    @Test
    fun `a long single-mode run stays bounded per line and loses no points`() {
        val maxPoints = 50
        val lines = mutableListOf<ArrayList<Int>>()
        var current = ArrayList<Int>()
        var mode: String? = null
        for (p in 0 until 1234) {
            val next = startOfNextTrailLine(current, "gnss" != mode, maxPoints, current.lastOrNull())
            if (next != null) {
                mode = "gnss"
                current = next
                lines.add(current)
            }
            current.add(p)
        }
        assertTrue(lines.all { it.size <= maxPoints })
        // Each line after the first starts on the previous line's last
        // point so the drawn track has no gaps; dropping that repeat, it's
        // every point exactly once, in order.
        lines.zipWithNext { a, b -> assertEquals(a.last(), b.first()) }
        val joined = lines.first() + lines.drop(1).flatMap { it.drop(1) }
        assertEquals((0 until 1234).toList(), joined)
    }
}
