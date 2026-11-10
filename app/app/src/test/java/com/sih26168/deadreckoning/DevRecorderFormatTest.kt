package com.sih26168.deadreckoning

import java.util.Locale
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Test

/** DevRecorder's CSV rows must parse the same whatever the phone's locale. */
class DevRecorderFormatTest {

    private val originalLocale = Locale.getDefault()

    @After
    fun restoreLocale() = Locale.setDefault(originalLocale)

    @Test
    fun `rows use a dot decimal separator on a comma-decimal locale`() {
        Locale.setDefault(Locale.GERMANY)
        val row = formatRecordingRow(
            1.5, floatArrayOf(0.1f, -0.2f, 9.81f), floatArrayOf(0f, 0f, 0.05f),
            hasFix = true, lat = 31.2576, lon = 75.7065, speed = 12.5f, bearingDeg = 90f,
        )
        assertEquals(
            "1.500,0.10000,-0.20000,9.81000,0.00000,0.00000,0.05000,31.2576000,75.7065000,12.500,90.00\n",
            row,
        )
        assertEquals(11, row.trimEnd().split(",").size)
    }

    @Test
    fun `a missing fix leaves the four GPS columns blank`() {
        Locale.setDefault(Locale.FRANCE)
        val row = formatRecordingRow(
            0.0, floatArrayOf(0f, 0f, 0f), floatArrayOf(0f, 0f, 0f),
            hasFix = false, lat = 0.0, lon = 0.0, speed = 0f, bearingDeg = 0f,
        )
        assertEquals("0.000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,,,,\n", row)
    }
}
